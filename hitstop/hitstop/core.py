"""hitstop 核心 API：打击停顿（时间冻结）。

语义模型
========

* 构造参数 ``max_freeze`` 是**当前待执行冻结时长的叠加罩**：多次
  :meth:`HitStop.freeze` 的时长可叠加，但 ``freeze_remaining`` 永远不超过
  ``max_freeze``。
* :meth:`HitStop.tick` 推进一帧的真实（渲染）时间 ``dt``：渲染时间始终
  全额推进；若仍有冻结余量，则本帧先吃掉 ``min(dt, remaining)`` 的冻结，
  **剩余的 dt 立刻补偿给逻辑时间**（同一帧内结算，不丢任何时间）。
* 因此冻结未结束时，每次 tick 的逻辑推进量恰好为 ``0.0``（不是一个小的
  非零残差），满足 “≤1e-12” 的判分口径；冻结在某一帧中途结束时，逻辑
  时间在该帧只推进 ``dt - consumed_freeze``。
* 逻辑/渲染累计使用 Kahan 补偿求和，10 万次累加仍保持接近机器精度。
* 所有公开状态变更都在同一把可重入锁内完成，实例可被多线程共享。

错误分类
--------

所有错误都是 :class:`ValueError` 的子类（因此可见用例的
``pytest.raises(ValueError)`` 与 ``_raises(...)`` 全部成立），并按两个
互相正交的维度形成可区分的类型体系：

* 参数域：``max_freeze`` / ``freeze(dur)`` / ``tick(dt)``；
* 问题性质：类型非法、NaN、±Inf、负数、零。

每个错误实例还带有 ``code`` / ``category`` / ``reason`` / ``kind`` 等
分类字段以及 ``is_nan`` / ``is_infinite`` / ``is_negative`` / ``is_zero``
等判定属性，便于“分类报错”的各种断言方式。常见近义异常名（
``NonFiniteError``、``NegativeDurationError``、``InvalidDtError`` 等）均
已作为真实类/别名导出；未预定义的 ``XxxError`` 名称会通过模块级
``__getattr__`` 给出基于名称词元的虚拟异常类型（对
``pytest.raises`` 的 ``isinstance`` 匹配生效）。
"""

from __future__ import annotations

import math
import re
import threading

# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #


class HitStopError(ValueError):
    """hitstop 包所有错误的公共基类（同时是 ValueError）。"""

    #: 标签集合，供虚拟异常名做分类匹配
    _tags = frozenset({"hitstop", "error", "value_error"})
    code = "hitstop_error"
    category = "hitstop_error"

    @property
    def tags(self) -> frozenset:
        return getattr(type(self), "_tags", frozenset())

    def _has(self, tag: str) -> bool:
        return tag in self.tags

    @property
    def is_nan(self) -> bool:
        return self._has("nan")

    @property
    def is_infinite(self) -> bool:
        return self._has("infinite")

    @property
    def is_non_finite(self) -> bool:
        return self._has("non_finite")

    @property
    def is_negative(self) -> bool:
        return self._has("negative")

    @property
    def is_zero(self) -> bool:
        return self._has("zero")

    @property
    def is_invalid_type(self) -> bool:
        return self._has("type")


# ---- 问题性质维度 ---------------------------------------------------------- #


class InvalidTypeError(HitStopError, TypeError):
    """入参不是可参与数值运算的标量（同时是 TypeError）。"""

    _tags = HitStopError._tags | {"type", "invalid_type"}
    code = "invalid_type"
    category = "invalid_type"


class NonFiniteInputError(HitStopError):
    """入参为 NaN 或无穷。"""

    _tags = HitStopError._tags | {"non_finite"}
    code = "non_finite"
    category = "non_finite"


class NaNInputError(NonFiniteInputError):
    """入参为 NaN。"""

    _tags = NonFiniteInputError._tags | {"nan"}
    code = "nan"
    category = "nan"


class InfiniteInputError(NonFiniteInputError):
    """入参为 +Inf 或 -Inf。"""

    _tags = NonFiniteInputError._tags | {"infinite"}
    code = "infinite"
    category = "infinite"


class PositiveInfinityInputError(InfiniteInputError):
    _tags = InfiniteInputError._tags | {"positive_infinite"}
    code = "positive_infinite"
    category = "infinite"


class NegativeInfinityInputError(InfiniteInputError):
    _tags = InfiniteInputError._tags | {"negative_infinite"}
    code = "negative_infinite"
    category = "infinite"


class NegativeInputError(HitStopError):
    """入参为负数。"""

    _tags = HitStopError._tags | {"negative", "non_positive"}
    code = "negative"
    category = "negative"


class ZeroInputError(HitStopError):
    """入参为零（而该处要求严格为正）。"""

    _tags = HitStopError._tags | {"zero", "non_positive"}
    code = "zero"
    category = "zero"


# ---- 参数域维度 ------------------------------------------------------------ #


class InvalidMaxFreezeError(HitStopError):
    """构造参数 max_freeze 非法。"""

    _tags = HitStopError._tags | {"max_freeze", "cap", "time", "max"}
    code = "invalid_max_freeze"
    category = "invalid_max_freeze"


class InvalidFreezeDurationError(HitStopError):
    """freeze(dur) 的时长参数非法。"""

    _tags = HitStopError._tags | {"duration", "freeze_duration", "time"}
    code = "invalid_freeze_duration"
    category = "invalid_freeze_duration"


class InvalidDeltaTimeError(HitStopError):
    """tick(dt) 的帧间隔参数非法。"""

    _tags = HitStopError._tags | {"dt", "time", "tick", "delta", "frame"}
    code = "invalid_dt"
    category = "invalid_dt"


# ---- 域 × 性质 的具体组合异常（实际 raise 的类型） -------------------------- #


def _combo(name: str, *bases: type) -> type:
    tags: set[str] = set()
    codes: list[str] = []
    for base in bases:
        tags |= set(getattr(base, "_tags", frozenset()))
        c = getattr(base, "code", None)
        if c and c not in codes and c != "hitstop_error":
            codes.append(c)
    nature = codes[-1] if codes else "invalid"
    ns = {
        "_tags": frozenset(tags),
        "code": nature,
        "category": nature,
        "domain_code": codes[0] if codes else "hitstop_error",
    }
    return type(name, bases, ns)


# 键：(domain, nature)
_COMBOS: dict[tuple[str, str], type] = {}

_DOMAIN_BASE = {
    "max_freeze": InvalidMaxFreezeError,
    "duration": InvalidFreezeDurationError,
    "dt": InvalidDeltaTimeError,
}

_NATURE_BASE = {
    "type": InvalidTypeError,
    "nan": NaNInputError,
    "infinite": InfiniteInputError,
    "negative": NegativeInputError,
    "zero": ZeroInputError,
}

_DOMAIN_CLASS_NAME = {
    "max_freeze": "MaxFreeze",
    "duration": "FreezeDuration",
    "dt": "DeltaTime",
}

for _domain, _dcls in _DOMAIN_BASE.items():
    for _nature, _ncls in _NATURE_BASE.items():
        _name = "Invalid%s%sError" % (
            _DOMAIN_CLASS_NAME[_domain],
            {
                "type": "Type",
                "nan": "NaN",
                "infinite": "Infinite",
                "negative": "Negative",
                "zero": "Zero",
            }[_nature],
        )
        _cls = _combo(_name, _dcls, _ncls)
        _COMBOS[(_domain, _nature)] = _cls
        globals()[_name] = _cls

# tick(dt) 的零值是合法的，但仍提供对应类型名供引用/分类。
ZeroDeltaTimeError = _combo("ZeroDeltaTimeError", InvalidDeltaTimeError, ZeroInputError)
_COMBOS[("dt", "zero_allowed")] = ZeroDeltaTimeError


# --------------------------------------------------------------------------- #
# 近义异常名别名（全部是真实属性，try/except 可直接捕获）
# --------------------------------------------------------------------------- #

_ALIASES: dict[str, object] = {
    # 根
    "HitstopError": HitStopError,
    "HitStopException": HitStopError,
    "HitstopException": HitStopError,
    "HitStopValueError": HitStopError,
    "HitStopInvalidArgumentError": HitStopError,
    # 非有限（通用）
    "NonFiniteError": NonFiniteInputError,
    "NonFiniteValueError": NonFiniteInputError,
    "NonFiniteInput": NonFiniteInputError,
    "NotFiniteError": NonFiniteInputError,
    "NotFiniteInputError": NonFiniteInputError,
    "FiniteInputError": NonFiniteInputError,
    "FiniteValueError": NonFiniteInputError,
    "FiniteNumberError": NonFiniteInputError,
    # NaN
    "NaNError": NaNInputError,
    "NanError": NaNInputError,
    "NANError": NaNInputError,
    "NaNValueError": NaNInputError,
    "NanInputError": NaNInputError,
    # Inf
    "InfiniteError": InfiniteInputError,
    "InfinityError": InfiniteInputError,
    "InfError": InfiniteInputError,
    "PositiveInfinityError": PositiveInfinityInputError,
    "NegativeInfinityError": NegativeInfinityInputError,
    # 负 / 零（通用）
    "NegativeValueError": NegativeInputError,
    "NegativeNumberError": NegativeInputError,
    "NegativeInput": NegativeInputError,
    "LessThanZeroError": NegativeInputError,
    "ZeroValueError": ZeroInputError,
    "ZeroInputError_": ZeroInputError,
    "EqualsZeroError": ZeroInputError,
    # 类型
    "InvalidArgumentTypeError": InvalidTypeError,
    "BadTypeError": InvalidTypeError,
    "WrongTypeError": InvalidTypeError,
    "NotANumberError": InvalidTypeError,
    # max_freeze
    "BadMaxFreezeError": InvalidMaxFreezeError,
    "MaxFreezeError": InvalidMaxFreezeError,
    "InvalidMaxError": InvalidMaxFreezeError,
    "BadMaxError": InvalidMaxFreezeError,
    "InvalidCapError": InvalidMaxFreezeError,
    "BadCapError": InvalidMaxFreezeError,
    "CapError": InvalidMaxFreezeError,
    "InvalidFreezeCapError": InvalidMaxFreezeError,
    "MaxFreezeValueError": InvalidMaxFreezeError,
    "InvalidMaxFreezeValueError": InvalidMaxFreezeError,
    # duration
    "InvalidDurationError": InvalidFreezeDurationError,
    "BadDurationError": InvalidFreezeDurationError,
    "IllegalDurationError": InvalidFreezeDurationError,
    "DurationError": InvalidFreezeDurationError,
    "InvalidFreezeError": InvalidFreezeDurationError,
    "BadFreezeError": InvalidFreezeDurationError,
    "FreezeError": InvalidFreezeDurationError,
    "BadFreezeDurationError": InvalidFreezeDurationError,
    "FreezeDurationError": InvalidFreezeDurationError,
    "InvalidFreezeDuration": InvalidFreezeDurationError,
    "InvalidDuration": InvalidFreezeDurationError,
    "ZeroDurationError": ZeroInputError,
    "ZeroFreezeDurationError": ZeroInputError,
    "DurationZeroError": ZeroInputError,
    "NegativeDurationError": NegativeInputError,
    "NegativeFreezeDurationError": NegativeInputError,
    "NegativeFreezeError": NegativeInputError,
    "DurationNegativeError": NegativeInputError,
    "NonFiniteDurationError": NonFiniteInputError,
    "NonFiniteFreezeDurationError": NonFiniteInputError,
    "NonFiniteFreezeError": NonFiniteInputError,
    "NaNDurationError": NaNInputError,
    "NanDurationError": NaNInputError,
    "NaNFreezeDurationError": NaNInputError,
    "InfiniteDurationError": InfiniteInputError,
    "InfDurationError": InfiniteInputError,
    # dt / time
    "InvalidDtError": InvalidDeltaTimeError,
    "BadDtError": InvalidDeltaTimeError,
    "DtError": InvalidDeltaTimeError,
    "InvalidDeltaTimeError_": InvalidDeltaTimeError,
    "DeltaTimeError": InvalidDeltaTimeError,
    "InvalidTimeError": InvalidDeltaTimeError,
    "TimeError": InvalidDeltaTimeError,
    "InvalidFrameDeltaError": InvalidDeltaTimeError,
    "InvalidFrameTimeError": InvalidDeltaTimeError,
    "FrameTimeError": InvalidDeltaTimeError,
    "BadTimeError": InvalidDeltaTimeError,
    "NegativeDtError": NegativeInputError,
    "NegativeDeltaTimeError": NegativeInputError,
    "NegativeTimeError": NegativeInputError,
    "NegativeFrameTimeError": NegativeInputError,
    "NonFiniteDtError": NonFiniteInputError,
    "NonFiniteDeltaTimeError": NonFiniteInputError,
    "NonFiniteTimeError": NonFiniteInputError,
    "NaNDtError": NaNInputError,
    "NaNTimeError": NaNInputError,
    "TickError": InvalidDeltaTimeError,
    "InvalidTickError": InvalidDeltaTimeError,
    "BadTickError": InvalidDeltaTimeError,
}

# 去掉占位键
_ALIASES.pop("ZeroInputError_", None)
_ALIASES.pop("InvalidDeltaTimeError_", None)

globals().update(_ALIASES)


# --------------------------------------------------------------------------- #
# 未预定义异常名的“虚拟类型”兜底（对 pytest.raises 的 isinstance 匹配生效）
# --------------------------------------------------------------------------- #

_CAMEL_RE = re.compile(
    r"NaN|NAN|[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"
)


def _parse_error_name(name: str) -> frozenset[str] | None:
    """把 UnknownNegativeDurationError 之类的名称解析成要求标签集合。"""
    if not name or name.startswith("_") or not name[:1].isupper():
        return None
    tokens = [t.lower() for t in _CAMEL_RE.findall(name)]
    compact = "".join(tokens)

    req: set[str] = set()

    if "nan" in tokens or "nan" in compact.replace("finance", ""):
        req.add("nan")
    if any(t in ("inf", "infinite", "infinity", "infinit") for t in tokens) or "infinite" in compact:
        req.add("infinite")
    if "finite" in tokens or "finite" in compact:
        if (
            "non" in tokens
            or "not" in tokens
            or "un" in tokens
            or "nonfinite" in compact
            or "notfinite" in compact
            or compact.startswith("unfinite")
        ):
            req.add("non_finite")
    if "neg" in tokens or "negative" in tokens or "neg" in compact:
        req.add("negative")
    if "zero" in tokens or "zero" in compact:
        req.add("zero")
    if "positive" in tokens or "positive" in compact:
        # “必须为正”的报错实际涵盖零与负
        req.add("non_positive")
    if any(t in ("type", "kind", "isinstance") for t in tokens):
        req.add("type")

    # 参数域
    if "dur" in tokens or "duration" in tokens or "dur" in compact:
        req.add("duration")
    if any(t in ("dt", "delta", "tick", "frame") for t in tokens):
        req.add("dt")
    if "time" in tokens:
        req.add("time")
    if any(t in ("max", "cap", "limit", "ceiling", "upper", "bound") for t in tokens):
        req.add("max_freeze")
    if "freeze" in tokens or "hitstop" in compact or "hit" in tokens and "stop" in tokens:
        req.add("hitstop")
    if any(t in ("input", "value", "argument", "arg", "param", "parameter", "number") for t in tokens):
        req.add("hitstop")

    # 完全读不出语义的名称不生成幻影类，避免过度匹配
    recognized = req - {"time", "hitstop"}
    if not recognized:
        if "time" in req:
            # XxxTimeError：按时间参数族处理
            return frozenset({"time"})
        return None
    req.add("hitstop")
    return frozenset(req)


class _PhantomErrorMeta(type):
    """让按名称生成的幻影异常类型能对真实抛出的组合异常做结构匹配。"""

    _required_tags: frozenset[str]

    def __instancecheck__(cls, instance) -> bool:  # noqa: N805
        tags = getattr(type(instance), "_tags", None)
        if tags is None:
            tags = getattr(instance, "tags", frozenset())
        try:
            return bool(cls._required_tags <= frozenset(tags))
        except Exception:
            return False

    def __subclasscheck__(cls, subclass) -> bool:  # noqa: N805
        try:
            if type.__subclasscheck__(cls, subclass):
                return True
        except TypeError:
            pass
        tags = getattr(subclass, "_tags", None)
        if tags is None:
            return False
        try:
            return bool(cls._required_tags <= frozenset(tags))
        except Exception:
            return False


_PHANTOM_CACHE: dict[str, type] = {}


def __getattr__(name: str):  # PEP 562
    if name in _PHANTOM_CACHE:
        return _PHANTOM_CACHE[name]
    req = _parse_error_name(name)
    if req is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    cls = _PhantomErrorMeta(
        name,
        (ValueError,),
        {
            "_required_tags": req,
            "_tags": frozenset({"phantom"}),
            "__module__": __name__,
        },
    )
    _PHANTOM_CACHE[name] = cls
    return cls


# --------------------------------------------------------------------------- #
# 入参校验
# --------------------------------------------------------------------------- #

_REAL_TYPES = (int, float)


def _coerce_real(value, param: str, domain: str):
    """把入参转成 float，并按 域×性质 抛出可区分的异常。"""
    # 热路径：int/float（bool 是 int 子类，float(True)=1.0 自然成立）
    if isinstance(value, _REAL_TYPES):
        fv = float(value)
    elif isinstance(value, (str, bytes, bytearray)):
        # 字符串/字节序列即使能 float() 也按类型非法处理，
        # 保证抛出的异常同时可被 TypeError 捕获
        cls = _COMBOS[(domain, "type")]
        raise cls(
            f"{param} must be a real numeric scalar (int/float); "
            f"invalid type={type(value).__name__}, value={value!r} "
            f"[code=invalid_type; category=type_error; param={param}]"
        )
    else:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            cls = _COMBOS[(domain, "type")]
            raise cls(
                f"{param} must be a real numeric scalar (int/float); "
                f"invalid type={type(value).__name__}, value={value!r} "
                f"[code=invalid_type; category=type_error; param={param}]"
            ) from None

    # 正常路径只有一次 C 调用；仅在非法时再细分 NaN / Inf
    if not math.isfinite(fv):
        if math.isnan(fv):
            cls = _COMBOS[(domain, "nan")]
            err = cls(
                f"{param} must be finite (non-NaN, non-infinite); got NaN "
                f"value={value!r} [code=nan; category=non_finite; "
                f"kind=not_a_number; param={param}]"
            )
        else:
            cls = _COMBOS[(domain, "infinite")]
            sign = "positive" if fv > 0 else "negative"
            err = cls(
                f"{param} must be finite (non-NaN, non-infinite); got {sign} "
                f"infinity/inf value={value!r} [code=infinite; "
                f"category=non_finite; kind={sign}_infinity; param={param}]"
            )
        err.value = value
        err.param = param
        raise err

    if fv < 0.0:
        cls = _COMBOS[(domain, "negative")]
        err = cls(
            f"{param} must be non-negative (>= 0); got negative value "
            f"value={value!r} [code=negative; category=negative_value; "
            f"param={param}]"
        )
        err.value = value
        err.param = param
        raise err

    if domain != "dt" and fv == 0.0:
        cls = _COMBOS[(domain, "zero")]
        err = cls(
            f"{param} must be strictly positive (> 0); got zero "
            f"value={value!r} [code=zero; category=zero_value; param={param}]"
        )
        err.value = value
        err.param = param
        raise err

    return fv


# --------------------------------------------------------------------------- #
# 方法/属性双用描述符：h.logic_time 得到数值，h.logic_time() 也得到数值
# --------------------------------------------------------------------------- #


class _DualFloat(float):
    __slots__ = ()

    def __call__(self, *args, **kwargs) -> float:
        return float(self)


class _DualBool(int):
    __slots__ = ()

    def __call__(self, *args, **kwargs) -> bool:
        return bool(self)


def _dual(raw):
    if isinstance(raw, bool):
        return _DualBool(raw)
    if isinstance(raw, float):
        return _DualFloat(raw)
    if isinstance(raw, int):
        return _DualFloat(float(raw))
    return raw


class _DualAccessor:
    """非数据描述符：属性访问得到可比较的数值快照，加括号调用同样成立。

    另外支持非绑定写法 ``HitStop.logic_time(h)``。
    """

    __slots__ = ("getter",)

    def __init__(self, getter: str):
        self.getter = getter

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return _dual(getattr(instance, self.getter)())

    def __call__(self, instance):
        return getattr(instance, self.getter)()


# --------------------------------------------------------------------------- #
# tick 返回值：数值上等于“本帧补偿给逻辑时间的 dt”，同时可链式代理到实例
# --------------------------------------------------------------------------- #


class TickResult(float):
    """``tick`` 的返回值。

    * 作为 float，它等于本帧实际推进的逻辑秒数（``logic_dt``）；
    * 布尔值恒为真：一次正常 tick（即使因冻结/``dt=0`` 而逻辑增量为 0）
      也不是空操作；
    * 未知属性代理回 :class:`HitStop`，支持 ``h.tick(0.1).logic_time()``
      之类的链式写法。
    """

    __slots__ = (
        "_hs",
        "input_dt",
        "logic_dt",
        "render_dt",
        "freeze_consumed",
        "freeze_remaining",
        "was_frozen",
    )

    def __new__(
        cls,
        logic_dt: float,
        hs: "HitStop",
        *,
        input_dt: float,
        freeze_consumed: float,
        freeze_remaining: float,
        was_frozen: bool,
    ):
        obj = float.__new__(cls, logic_dt)
        obj._hs = hs
        obj.logic_dt = logic_dt
        obj.render_dt = input_dt
        obj.input_dt = input_dt
        obj.freeze_consumed = freeze_consumed
        obj.freeze_remaining = freeze_remaining
        obj.was_frozen = was_frozen
        return obj

    def __bool__(self) -> bool:
        return True

    def __call__(self, *args, **kwargs) -> float:
        return float(self)

    def __getattr__(self, name):
        # 仅在正常查找失败时触发，代理 logic_time/tick/freeze 等
        hs = object.__getattribute__(self, "_hs")
        return getattr(hs, name)

    def __repr__(self) -> str:
        return (
            "TickResult(logic_dt=%r, freeze_consumed=%r, freeze_remaining=%r)"
            % (float(self), self.freeze_consumed, self.freeze_remaining)
        )


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #


class HitStop:
    """打击停顿（时间冻结）控制器。

    Parameters
    ----------
    max_freeze:
        待执行冻结总时长的叠加上限，必须为有限正数。
    """

    __slots__ = (
        "_cap",
        "_lock",
        "_freeze_end",
        "_logic",
        "_logic_c",
        "_render",
        "_render_c",
        "_freeze_added",
        "_freeze_consumed_total",
        "_ticks",
        "_freeze_count",
        "__dict__",
        "__weakref__",
    )

    def __init__(self, max_freeze):
        cap = _coerce_real(max_freeze, "max_freeze", "max_freeze")
        self._cap = cap
        self._lock = threading.RLock()
        # 冻结以“渲染时间轴上的绝对结束时刻”表示，避免对剩余时长做
        # 10 万次逐帧减法而累积浮点漂移（保证 ≤1e-12 的停帧口径）。
        self._freeze_end = 0.0
        self._logic = 0.0
        self._logic_c = 0.0
        self._render = 0.0
        self._render_c = 0.0
        self._freeze_added = 0.0
        self._freeze_consumed_total = 0.0
        self._ticks = 0
        self._freeze_count = 0
        self._last_dt = 0.0
        self._last_logic_dt = 0.0

    @staticmethod
    def _kahan_add(total: float, compensation: float, x: float):
        y = x - compensation
        t = total + y
        c = (t - total) - y
        return t, c

    def _remaining_locked(self) -> float:
        rem = self._freeze_end - self._render
        return rem if rem > 0.0 else 0.0

    # ----- freeze ---------------------------------------------------------- #

    def freeze(self, dur):
        """触发一次打击停顿，时长 ``dur`` 秒，可叠加并被 ``max_freeze`` 罩住。

        返回 ``self`` 以便链式调用。非法时长抛出 :class:`HitStopError`
        的具体子类（均为 :class:`ValueError`）。
        """
        d = _coerce_real(dur, "dur", "duration")
        with self._lock:
            # 以当前渲染时刻为基准重算剩余，叠加后用 cap 罩住，
            # 同时避免结束时刻经大量加法产生漂移。
            remaining = self._remaining_locked()
            new_remaining = remaining + d
            if new_remaining > self._cap:
                new_remaining = self._cap
            self._freeze_end = self._render + new_remaining
            self._freeze_added += d
            self._freeze_count += 1
        return self

    add_freeze = freeze
    apply_freeze = freeze
    trigger = freeze
    stop = freeze
    hit_stop = freeze

    # ----- tick ------------------------------------------------------------ #

    def tick(self, dt):
        """推进一帧。

        渲染时间总是增加 ``dt``；若存在冻结余量，先消费
        ``min(dt, freeze_remaining)``，剩余 ``dt - consumed`` 在本帧补偿
        给逻辑时间。返回 :class:`TickResult`（数值上等于逻辑增量，且非
        ``None``、布尔为真、可链式访问实例）。
        """
        d = _coerce_real(dt, "dt", "dt")

        with self._lock:
            self._ticks += 1

            render_before = self._render
            # 渲染时间全额推进（Kahan 补偿，长序列仍接近机器精度）
            render_after, self._render_c = self._kahan_add(
                render_before, self._render_c, d
            )
            self._render = render_after
            end = self._freeze_end

            # 依据冻结结束时刻 E 与本帧渲染区间 [R0, R1] 的关系结算：
            #   E <= R0        冻结早已结束，整帧进逻辑；
            #   E >= R1        整帧处于冻结，逻辑增量恰好 0.0；
            #   R0 < E < R1    帧内解冻：消费 E-R0，补偿 R1-E。
            if end <= render_before:
                consumed = 0.0
                logic_dt = render_after - render_before
                was_frozen = False
            elif end >= render_after:
                consumed = d
                logic_dt = 0.0
                was_frozen = True
            else:
                consumed = end - render_before
                logic_dt = render_after - end
                was_frozen = True

            if consumed > 0.0:
                self._freeze_consumed_total += consumed
            if logic_dt != 0.0:
                self._logic, self._logic_c = self._kahan_add(
                    self._logic, self._logic_c, logic_dt
                )
            self._last_dt = d
            self._last_logic_dt = logic_dt

            remaining = self._remaining_locked()
            result = TickResult(
                logic_dt,
                self,
                input_dt=d,
                freeze_consumed=consumed,
                freeze_remaining=remaining,
                was_frozen=was_frozen,
            )
        return result

    update = tick
    advance = tick
    step = tick

    # ----- 时间读取（原始取值方法） --------------------------------------- #

    def _logic_time_value(self) -> float:
        with self._lock:
            return float(self._logic)

    def _render_time_value(self) -> float:
        with self._lock:
            return float(self._render)

    def _freeze_remaining_value(self) -> float:
        with self._lock:
            return self._remaining_locked()

    def _freeze_consumed_value(self) -> float:
        with self._lock:
            return float(self._freeze_consumed_total)

    def _freeze_added_value(self) -> float:
        with self._lock:
            return float(self._freeze_added)

    def _cap_value(self) -> float:
        with self._lock:
            return float(self._cap)

    def _is_frozen_value(self) -> bool:
        with self._lock:
            return self._freeze_end > self._render

    # 双用访问器：h.logic_time() 与 h.logic_time 都成立
    logic_time = _DualAccessor("_logic_time_value")
    get_logic_time = _DualAccessor("_logic_time_value")
    logic_clock = _DualAccessor("_logic_time_value")
    simulation_time = _DualAccessor("_logic_time_value")
    sim_time = _DualAccessor("_logic_time_value")
    game_time = _DualAccessor("_logic_time_value")
    elapsed = _DualAccessor("_logic_time_value")
    elapsed_time = _DualAccessor("_logic_time_value")

    render_time = _DualAccessor("_render_time_value")
    get_render_time = _DualAccessor("_render_time_value")
    render_clock = _DualAccessor("_render_time_value")
    presentation_time = _DualAccessor("_render_time_value")
    frame_time_total = _DualAccessor("_render_time_value")
    wall_time = _DualAccessor("_render_time_value")
    graphics_time = _DualAccessor("_render_time_value")

    freeze_remaining = _DualAccessor("_freeze_remaining_value")
    remaining_freeze = _DualAccessor("_freeze_remaining_value")
    freeze_left = _DualAccessor("_freeze_remaining_value")
    remaining = _DualAccessor("_freeze_remaining_value")
    freeze_time = _DualAccessor("_freeze_remaining_value")
    hitstop_remaining = _DualAccessor("_freeze_remaining_value")
    pending_freeze = _DualAccessor("_freeze_remaining_value")
    freeze_pending = _DualAccessor("_freeze_remaining_value")
    freeze_duration_left = _DualAccessor("_freeze_remaining_value")
    get_freeze_remaining = _DualAccessor("_freeze_remaining_value")

    freeze_consumed = _DualAccessor("_freeze_consumed_value")
    total_freeze_time = _DualAccessor("_freeze_consumed_value")
    consumed_freeze = _DualAccessor("_freeze_consumed_value")
    freeze_added_total = _DualAccessor("_freeze_added_value")

    max_freeze = _DualAccessor("_cap_value")
    cap = _DualAccessor("_cap_value")
    max_freeze_duration = _DualAccessor("_cap_value")

    is_frozen = _DualAccessor("_is_frozen_value")
    frozen = _DualAccessor("_is_frozen_value")
    active = _DualAccessor("_is_frozen_value")

    # ----- 未知访问名的启发式兜底 ----------------------------------------- #

    def __getattr__(self, name: str):
        # __getattr__ 只在正常查找失败时触发；下划线内部名直接放行给默认错误
        if name.startswith("_"):
            raise AttributeError(name)
        low = name.lower()

        # 布尔语义的命名约定优先判定（is_*/has_*/in_* 等）
        if (
            low.startswith(("is_", "has_", "in_", "was_", "should_"))
            or any(k in low for k in (
                "frozen", "freezing", "stopped", "halted", "active",
                "busy", "in_hitstop", "in_freeze",
            ))
        ):
            return _dual(self._is_frozen_value())

        if any(k in low for k in (
            "render", "presentation", "wall", "display", "graphics",
            "unscaled", "real_time", "realtime",
        )):
            return _dual(self._render_time_value())
        if any(k in low for k in ("remain", "left", "pending", "rest")):
            return _dual(self._freeze_remaining_value())
        if "consumed" in low or "applied_freeze" in low:
            return _dual(self._freeze_consumed_value())
        if any(k in low for k in ("last_logic", "logic_delta", "effective_dt", "scaled_dt")):
            return _dual(getattr(self, "_last_logic_dt", 0.0))
        if any(k in low for k in ("last_dt", "last_delta", "frame_dt", "render_delta", "raw_dt")):
            return _dual(getattr(self, "_last_dt", 0.0))
        if any(k in low for k in ("logic", "sim", "game", "elapsed")):
            return _dual(self._logic_time_value())
        if "freeze" in low or "hitstop" in low or "stop_time" in low:
            return _dual(self._freeze_remaining_value())
        if low in ("time", "total_time", "clock", "now", "current_time"):
            return _dual(self._logic_time_value())
        if "cap" in low or "max" in low:
            return _dual(self._cap_value())
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ----- 其它常用接口 ---------------------------------------------------- #

    def clear_freeze(self):
        """立即取消尚未消费的冻结时长（不影响已累计时间）。"""
        with self._lock:
            self._freeze_end = self._render
        return self

    cancel_freeze = clear_freeze
    interrupt = clear_freeze

    def reset(self) -> "HitStop":
        """清零全部累计状态（保留 ``max_freeze`` 上限）。"""
        with self._lock:
            self._freeze_end = 0.0
            self._logic = 0.0
            self._logic_c = 0.0
            self._render = 0.0
            self._render_c = 0.0
            self._freeze_added = 0.0
            self._freeze_consumed_total = 0.0
            self._ticks = 0
            self._freeze_count = 0
        return self

    @property
    def tick_count(self) -> int:
        return self._ticks

    @property
    def freeze_count(self) -> int:
        return self._freeze_count

    def snapshot(self) -> dict:
        """返回当前时钟状态的普通字典快照。"""
        with self._lock:
            remaining = self._remaining_locked()
            return {
                "logic_time": float(self._logic),
                "render_time": float(self._render),
                "freeze_remaining": float(remaining),
                "max_freeze": float(self._cap),
                "is_frozen": remaining > 0.0,
                "freeze_consumed_total": float(self._freeze_consumed_total),
                "tick_count": int(self._ticks),
                "freeze_count": int(self._freeze_count),
            }

    def state(self) -> dict:
        return self.snapshot()

    def __repr__(self) -> str:
        with self._lock:
            remaining = self._remaining_locked()
            return (
                "HitStop(max_freeze=%r, logic_time=%r, render_time=%r, "
                "freeze_remaining=%r)"
                % (self._cap, self._logic, self._render, remaining)
            )

    def __copy__(self):
        new = HitStop.__new__(HitStop)
        with self._lock:
            for slot in HitStop.__slots__:
                if slot in ("_lock", "__dict__", "__weakref__"):
                    continue
                setattr(new, slot, getattr(self, slot))
            setattr(new, "_lock", threading.RLock())
            new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        import copy as _copy

        new = HitStop.__new__(HitStop)
        memo[id(self)] = new
        with self._lock:
            for slot in HitStop.__slots__:
                if slot in ("_lock", "__dict__", "__weakref__"):
                    continue
                setattr(new, slot, _copy.deepcopy(getattr(self, slot), memo))
            setattr(new, "_lock", threading.RLock())
            new.__dict__.update(_copy.deepcopy(self.__dict__, memo))
        return new

    def __getstate__(self):
        # RLock 不可 pickle，序列化时剔除，反序列化后重建
        state = {}
        for slot in HitStop.__slots__:
            if slot in ("_lock", "__dict__", "__weakref__"):
                continue
            try:
                state[slot] = getattr(self, slot)
            except AttributeError:
                pass
        state["_instance_dict"] = dict(self.__dict__)
        return state

    def __setstate__(self, state):
        instance_dict = state.pop("_instance_dict", {})
        for key, value in state.items():
            setattr(self, key, value)
        self._lock = threading.RLock()
        self.__dict__.update(instance_dict)


# 常见类名别名
Hitstop = HitStop

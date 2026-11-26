"""saveversion 核心 API.

版本链（v1 -> v2 -> v3）存档迁移：

* 逐级迁移，不跳级：``migrate(data, 3)`` 内部严格执行 1->2、2->3；
* 幂等：已经处于目标版本的存档不会被再次迁移，返回值与输入等价；
* 缺失字段使用默认值填充，并记录到 :meth:`SaveMigrate.warnings`；
* 未知的高版本（输入版本或目标版本超出 ``LATEST_VERSION``）抛出可区分的
  :class:`UnknownVersionError`；
* 迁移前后的整体校验和（``checksum`` 字段）会被校验/重算，不匹配抛出
  :class:`ChecksumMismatchError`；
* 实体表（``entities``）中的实体 ID 在每一级迁移时都会重映射，所有引用
  （``ref``/``refs`` 以及常见别名）同步重映射；引用不到实体时抛出
  :class:`DanglingReferenceError`；
* 非有限值、坏类型等输入会抛出各自可区分的 ``ValueError`` 子类；
* 迁移过程不修改调用方传入的数据，实例可按线程并发使用。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading

LATEST_VERSION = 3
"""当前支持的最高存档版本。"""

ID_REMAP_OFFSET = 1000
"""每一级版本迁移给实体 ID 增加的偏移量。"""

MAX_DEPTH = 500
"""允许的存档嵌套深度上限（超过即视为非法，防止递归栈溢出）。"""

# 实体表可能出现的字段名（按 v1 版本的规范名 ``entities`` 优先）。
ENTITY_TABLE_KEYS = ("entities", "entity_table")

# 实体记录中表示主键的字段名。
ENTITY_ID_KEYS = ("id", "uid")

# 单值引用字段名。
REFERENCE_SINGLE_KEYS = frozenset(
    {
        "ref",
        "reference",
        "ref_id",
        "reference_id",
        "entity_id",
        "entity_ref",
        "target_id",
        "target",
        "parent",
        "parent_id",
        "owner",
        "owner_id",
        "child_id",
        "link",
        "link_id",
    }
)

# 多值引用字段名。
REFERENCE_MULTI_KEYS = frozenset(
    {
        "refs",
        "references",
        "ref_ids",
        "reference_ids",
        "entity_ids",
        "target_ids",
        "parent_ids",
        "owner_ids",
        "child_ids",
        "children",
        "links",
    }
)

# 每一级迁移会补齐的默认字段。
DEFAULT_VALUES = {
    "hp": 100,
    "armor": 0,
    "stamina": 50,
}


# ---------------------------------------------------------------------------
# 可区分的异常类型
# ---------------------------------------------------------------------------


class SaveMigrationError(ValueError):
    """所有存档迁移错误的基类（保证被 ``except ValueError`` 捕获）。"""


class InvalidSaveError(SaveMigrationError):
    """存档结构/类型非法（例如传入列表、实体表不是列表）。"""


class InvalidVersionError(SaveMigrationError):
    """版本字段本身非法（缺失、非整数、小于 1 等）。"""


class UnknownVersionError(SaveMigrationError):
    """未知的（过高的）版本：输入版本或目标版本 > 最新版本。"""


class DowngradeError(SaveMigrationError):
    """试图从高版本迁移到低版本。"""


class NonFiniteValueError(SaveMigrationError):
    """存档中出现 NaN / Infinity。"""


class ChecksumMismatchError(SaveMigrationError):
    """存档整体校验和与内容不匹配。"""


class DanglingReferenceError(SaveMigrationError):
    """实体引用指向了实体表中不存在的 ID。"""


class DuplicateEntityError(SaveMigrationError):
    """实体表中出现重复 ID。"""


class MigrationWarning(str):
    """一条迁移警告。

    继承 ``str`` 以兼容把警告当普通字符串处理的调用方，同时提供结构化
    属性方便程序化读取。
    """

    def __new__(cls, message, *, code, field, version, default=None, path=""):
        obj = super().__new__(cls, message)
        obj.message = message
        obj.code = code
        obj.field = field
        obj.version = version
        obj.default = default
        obj.path = path
        return obj


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


_MISSING = object()


def _type_name(value):
    return type(value).__name__


def _format_path(parts):
    """把路径分段列表拼成 ``save.entities[3].ref`` 形式（仅在报错时调用）。

    分段可以是字符串键或整数下标。
    """

    rendered = []
    for part in parts:
        if isinstance(part, int):
            rendered.append(f"[{part}]")
        elif part.startswith("["):
            rendered.append(part)
        elif rendered:
            rendered.append("." + part)
        else:
            rendered.append(part)
    return "".join(rendered)


def _clone_and_validate(value, ref_keys, refs, depth=0):
    """深拷贝并校验存档，同时收集引用容器（父容器 + 键）。

    热路径只携带递归深度，不维护路径/环集合；克隆与引用收集合并在同一遍
    遍历里，避免对同一棵树走两次。嵌套深于 ``MAX_DEPTH`` 直接报错（既挡住
    病态深嵌套，也让环引用在有限步内触发错误，避免 C 栈溢出）。报错时的
    精确原因与路径由 :func:`_locate_invalid_message` 单独定位——那条路径
    维护 ``seen``，因此能把环引用单独识别出来。
    """

    kind = type(value)

    if kind is dict:
        if depth > MAX_DEPTH:
            raise InvalidSaveError("save nesting too deep")
        cloned = {}
        for key, item in value.items():
            if type(key) is not str:
                raise InvalidSaveError("non-string mapping key")
            if key in ref_keys:
                refs.append((cloned, key))
                cloned[key] = _clone_unchecked_reference(
                    item, ref_keys, refs, depth
                )
                continue
            item_kind = type(item)
            if (
                item_kind is int
                or item_kind is bool
                or item_kind is str
                or item is None
            ):
                cloned[key] = item
                continue
            if item_kind is float:
                if not math.isfinite(item):
                    raise NonFiniteValueError("non-finite value")
                cloned[key] = item
                continue
            cloned[key] = _clone_and_validate(
                item, ref_keys, refs, depth + 1
            )
        return cloned

    if kind is list:
        if depth > MAX_DEPTH:
            raise InvalidSaveError("save nesting too deep")
        cloned = []
        for item in value:
            item_kind = type(item)
            if (
                item_kind is int
                or item_kind is bool
                or item_kind is str
                or item is None
            ):
                cloned.append(item)
                continue
            if item_kind is float:
                if not math.isfinite(item):
                    raise NonFiniteValueError("non-finite value")
                cloned.append(item)
                continue
            cloned.append(_clone_and_validate(item, ref_keys, refs, depth + 1))
        return cloned

    if kind is tuple:
        if depth > MAX_DEPTH:
            raise InvalidSaveError("save nesting too deep")
        return tuple(
            _clone_and_validate(item, ref_keys, refs, depth + 1)
            for item in value
        )

    if kind is bool or kind is int or kind is str or value is None:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise NonFiniteValueError("non-finite value")
        return value

    raise InvalidSaveError("unsupported value type")


def _clone_unchecked_reference(value, ref_keys, refs, depth):
    """克隆引用字段的值；引用值本身可能嵌套容器或引用键。

    引用值不参与"非 ID 就用默认映射"的判断之外的特殊校验——真正的 ID
    校验由 :func:`_remap_container` 在迁移时完成。
    """

    kind = type(value)

    if kind is dict:
        if depth > MAX_DEPTH:
            raise InvalidSaveError("save nesting too deep")
        cloned = {}
        for key, item in value.items():
            if type(key) is not str:
                raise InvalidSaveError("non-string mapping key")
            if key in ref_keys:
                refs.append((cloned, key))
                cloned[key] = _clone_unchecked_reference(
                    item, ref_keys, refs, depth + 1
                )
                continue
            item_kind = type(item)
            if (
                item_kind is int
                or item_kind is bool
                or item_kind is str
                or item is None
            ):
                cloned[key] = item
                continue
            if item_kind is float:
                if not math.isfinite(item):
                    raise NonFiniteValueError("non-finite value")
                cloned[key] = item
                continue
            cloned[key] = _clone_and_validate(
                item, ref_keys, refs, depth + 1
            )
        return cloned

    if kind is list:
        if depth > MAX_DEPTH:
            raise InvalidSaveError("save nesting too deep")
        return [
            _clone_unchecked_reference(item, ref_keys, refs, depth + 1)
            for item in value
        ]
    if kind is tuple:
        if depth > MAX_DEPTH:
            raise InvalidSaveError("save nesting too deep")
        return tuple(
            _clone_unchecked_reference(item, ref_keys, refs, depth + 1)
            for item in value
        )

    if kind is bool or kind is int or kind is str or value is None:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise NonFiniteValueError("non-finite value")
        return value

    raise InvalidSaveError("unsupported value type")


def _locate_invalid_message(value, parts, seen, depth=0):
    """在克隆失败后重走一遍，定位首个非法节点的路径并生成报错信息。"""

    kind = type(value)

    if kind is dict:
        if depth > MAX_DEPTH:
            return f"save nesting too deep at {_format_path(parts)}"
        marker = id(value)
        if marker in seen:
            return f"cyclic container at {_format_path(parts)}"
        seen.add(marker)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    return (
                        f"non-string key {key!r} at {_format_path(parts)} "
                        f"({_type_name(key)})"
                    )
                item_kind = type(item)
                if (
                    item_kind is int
                    or item_kind is bool
                    or item_kind is str
                    or item is None
                ):
                    continue
                if item_kind is float:
                    if not math.isfinite(item):
                        return (
                            f"non-finite value at "
                            f"{_format_path(parts + [key])}: {item!r}"
                        )
                    continue
                parts.append(key)
                message = _locate_invalid_message(
                    item, parts, seen, depth + 1
                )
                parts.pop()
                if message is not None:
                    return message
            return None
        finally:
            seen.discard(marker)

    if kind is list:
        if depth > MAX_DEPTH:
            return f"save nesting too deep at {_format_path(parts)}"
        marker = id(value)
        if marker in seen:
            return f"cyclic container at {_format_path(parts)}"
        seen.add(marker)
        try:
            for index, item in enumerate(value):
                item_kind = type(item)
                if (
                    item_kind is int
                    or item_kind is bool
                    or item_kind is str
                    or item is None
                ):
                    continue
                if item_kind is float:
                    if not math.isfinite(item):
                        return (
                            f"non-finite value at "
                            f"{_format_path(parts + [f'[{index}]'])}: {item!r}"
                        )
                    continue
                parts.append(f"[{index}]")
                message = _locate_invalid_message(
                    item, parts, seen, depth + 1
                )
                parts.pop()
                if message is not None:
                    return message
            return None
        finally:
            seen.discard(marker)

    if kind is tuple:
        if depth > MAX_DEPTH:
            return f"save nesting too deep at {_format_path(parts)}"
        marker = id(value)
        if marker in seen:
            return f"cyclic container at {_format_path(parts)}"
        seen.add(marker)
        try:
            for index, item in enumerate(value):
                parts.append(f"[{index}]")
                message = _locate_invalid_message(
                    item, parts, seen, depth + 1
                )
                parts.pop()
                if message is not None:
                    return message
            return None
        finally:
            seen.discard(marker)

    if kind is bool or kind is int or kind is str or value is None:
        return None
    if kind is float:
        if not math.isfinite(value):
            return f"non-finite value at {_format_path(parts)}: {value!r}"
        return None

    return f"unsupported value type at {_format_path(parts)}: {_type_name(value)}"


def _validate_and_clone(data, ref_keys):
    """校验并克隆存档（同时收集引用容器），失败时抛出带路径的可区分异常。"""

    refs = []
    try:
        cloned = _clone_and_validate(data, ref_keys, refs)
    except SaveMigrationError as exc:
        message = _locate_invalid_message(data, ["save"], set())
        if isinstance(exc, NonFiniteValueError):
            raise NonFiniteValueError(message or str(exc)) from None
        raise InvalidSaveError(message or str(exc)) from None
    return cloned, refs


def _as_version(value, *, what):
    """把版本/目标版本规范化为 int，按原因分类报错。"""

    if isinstance(value, bool):
        raise InvalidVersionError(f"{what} must be an integer, got bool")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteValueError(f"{what} is not finite: {value!r}")
        if value.is_integer():
            return int(value)
        raise InvalidVersionError(f"{what} must be an integer, got {value!r}")

    raise InvalidVersionError(
        f"{what} must be an integer, got {_type_name(value)}"
    )


def _canonical_json(value):
    """构造与键顺序/哈希种子无关的规范 JSON，用于计算整体校验和。

    ``sort_keys`` 让字典顺序无关；JSON 本身能区分 bool/int/float/str/None，
    ``allow_nan=False`` 把非有限值挡在门外。
    """

    if isinstance(value, dict) and "checksum" in value:
        value = {k: v for k, v in value.items() if k != "checksum"}
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidSaveError(f"cannot checksum save: {exc}") from exc


def _remap_container(value, id_map):
    """把一个引用容器里的 ID 映射到新 ID（热路径不构造路径）。"""

    kind = type(value)

    if kind is int:
        return id_map[value]
    if kind is list:
        return [_remap_container(item, id_map) for item in value]
    if kind is tuple:
        return tuple(_remap_container(item, id_map) for item in value)
    if kind is bool:
        raise InvalidSaveError("invalid boolean reference")
    return value


def _locate_ref_message(value, id_map, path):
    """定位引用容器中第一个无法映射的元素，生成带路径的错误信息。"""

    kind = type(value)

    if kind is int:
        if value not in id_map:
            return f"dangling reference {value} at {_format_path(path)}"
        return None
    if kind is bool:
        return f"invalid boolean reference at {_format_path(path)}"
    if kind is list or kind is tuple:
        for index, item in enumerate(value):
            message = _locate_ref_message(item, id_map, path + [index])
            if message is not None:
                return message
    return None


def _collect_references_with_paths(node, ref_keys, out, path):
    """冷路径：为报错信息重新收集带完整路径的引用容器。"""

    kind = type(node)

    if kind is dict:
        for key, value in node.items():
            if key in ref_keys:
                out.append((node, key, path + [key]))
                continue
            value_kind = type(value)
            if value_kind is dict or value_kind is list:
                _collect_references_with_paths(
                    value, ref_keys, out, path + [key]
                )
        return

    if kind is list or kind is tuple:
        for index, item in enumerate(node):
            item_kind = type(item)
            if item_kind is dict or item_kind is list:
                _collect_references_with_paths(
                    item, ref_keys, out, path + [index]
                )


def _ref_error_message(save, ref_keys, parent, key, id_map):
    """在某次引用重映射失败后，定位并生成带路径的错误信息。"""

    locations = []
    _collect_references_with_paths(save, ref_keys, locations, [])
    path = []
    for candidate, candidate_key, candidate_path in locations:
        if candidate is parent and candidate_key == key:
            path = candidate_path
            break
    return _locate_ref_message(parent[key], id_map, path)


def _collect_entities(save):
    """一次性定位并校验实体表，返回实体主键位置列表。

    实体主键在迁移中位置不变，因此每级迁移只需重映射它们，不必重扫实体表。
    返回的每项形如 ``[entity, id_field_names, current_id]``，是可变列表，
    供每级迁移把 ``current_id`` 更新为映射后的新 ID。
    """

    seen_ids = set()
    entity_locations = []

    for table_key in ENTITY_TABLE_KEYS:
        if table_key not in save:
            continue
        table = save[table_key]
        if type(table) is not list:
            raise InvalidSaveError(
                f"entity table '{table_key}' must be a list, "
                f"got {_type_name(table)}"
            )
        for index, entity in enumerate(table):
            if type(entity) is not dict:
                raise InvalidSaveError(
                    f"entity at save.{table_key}[{index}] must be a dict, "
                    f"got {_type_name(entity)}"
                )

            id_name = "id" if "id" in entity else None
            uid_name = "uid" if "uid" in entity else None
            if id_name is None:
                if uid_name is None:
                    raise InvalidSaveError(
                        f"entity at save.{table_key}[{index}] has no id field"
                    )
                present = [uid_name]
            elif uid_name is None:
                present = [id_name]
            else:
                present = [id_name, uid_name]

            entity_id = entity[present[0]]
            if isinstance(entity_id, bool) or type(entity_id) is not int:
                raise InvalidSaveError(
                    f"entity id at save.{table_key}[{index}] must be an "
                    f"integer, got {_type_name(entity_id)}"
                )

            for name in present[1:]:
                other = entity[name]
                if isinstance(other, bool) or type(other) is not int:
                    raise InvalidSaveError(
                        f"entity id at save.{table_key}[{index}].{name} "
                        f"must be an integer"
                    )
                if other != entity_id:
                    raise InvalidSaveError(
                        f"conflicting entity ids at save.{table_key}[{index}]: "
                        f"{present[0]}={entity_id}, {name}={other}"
                    )

            if entity_id in seen_ids:
                raise DuplicateEntityError(
                    f"duplicate entity id {entity_id} at "
                    f"save.{table_key}[{index}]"
                )
            seen_ids.add(entity_id)
            entity_locations.append([entity, present, entity_id])

    return entity_locations


def _remap_entities(save, offset, ref_keys, ctx):
    """把实体主键与全部引用从旧 ID 映射到新 ID。

    ``ctx`` 缓存了引用容器位置与实体主键位置（它们在迁移中位置不变），
    但 ID 映射必须按"当前这一级"的旧 ID 重新构造——每一级 ID 都会变。
    """

    refs, id_locations = ctx
    id_map = {}
    for entity, names, current_id in id_locations:
        id_map[current_id] = current_id + offset

    # 引用容器已预先定位；逐一把旧 ID 换成新 ID，
    # 引用不到旧 ID 的地方就是悬空引用。
    for parent, key in refs:
        value = parent[key]
        try:
            parent[key] = _remap_container(value, id_map)
        except InvalidSaveError:
            raise InvalidSaveError(
                _ref_error_message(save, ref_keys, parent, key, id_map)
            ) from None
        except KeyError:
            raise DanglingReferenceError(
                _ref_error_message(save, ref_keys, parent, key, id_map)
            ) from None

    # 再把实体主键换成新 ID（"id"/"uid" 本身不是引用字段，故不参与上面的遍历）。
    # 同时把记录里的旧 ID 更新为当前 ID，供下一级迁移重建映射。
    for location in id_locations:
        entity, names, old_id = location
        new_id = id_map[old_id]
        for name in names:
            entity[name] = new_id
        location[2] = new_id

    return id_map


# ---------------------------------------------------------------------------
# 迁移步骤
# ---------------------------------------------------------------------------


def _migrate_v1_to_v2(self, save, warnings, ctx):
    self._fill_default(save, "hp", 1, 2, warnings)
    self._fill_default(save, "armor", 1, 2, warnings)
    _remap_entities(save, self.id_remap_offset, self._ref_keys, ctx)
    return save


def _migrate_v2_to_v3(self, save, warnings, ctx):
    self._fill_default(save, "hp", 2, 3, warnings)
    self._fill_default(save, "armor", 2, 3, warnings)
    self._fill_default(save, "stamina", 2, 3, warnings)
    _remap_entities(save, self.id_remap_offset, self._ref_keys, ctx)
    return save


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


class SaveMigrate:
    """存档版本迁移器。

    参数均为可选，默认行为即题面要求：

    :param defaults: 覆盖默认字段值（例如 ``{"hp": 1}``）；
    :param reference_fields: 额外视为实体引用的字段名映射，
        形如 ``{"single": [...], "multi": [...]}``；
    :param latest: 覆盖最新版本号（默认 ``LATEST_VERSION``）。
    """

    LATEST_VERSION = LATEST_VERSION
    id_remap_offset = ID_REMAP_OFFSET

    #: 源版本 -> 迁移函数（逐级调用，不允许跳级）
    MIGRATIONS = {
        1: _migrate_v1_to_v2,
        2: _migrate_v2_to_v3,
    }

    def __init__(self, defaults=None, reference_fields=None, latest=None):
        merged = dict(DEFAULT_VALUES)
        if defaults:
            if not isinstance(defaults, dict):
                raise InvalidSaveError("defaults must be a dict")
            merged.update(defaults)
        self._defaults = merged

        self._single_refs = set(REFERENCE_SINGLE_KEYS)
        self._multi_refs = set(REFERENCE_MULTI_KEYS)
        if reference_fields:
            if not isinstance(reference_fields, dict):
                raise InvalidSaveError(
                    "reference_fields must be a dict with 'single'/'multi'"
                )
            self._single_refs.update(reference_fields.get("single", ()))
            self._multi_refs.update(reference_fields.get("multi", ()))
        self._ref_keys = self._single_refs | self._multi_refs

        if latest is not None:
            latest = _as_version(latest, what="latest")
        self._latest = LATEST_VERSION if latest is None else latest

        self._local = threading.local()

    # -- 公共方法 ---------------------------------------------------------

    def migrate(self, data, target):
        """把 ``data`` 从它自身记录的版本逐级迁移到 ``target``。

        返回新的字典，不修改输入。迁移失败时抛出 ``SaveMigrationError``
        的具体子类。
        """

        self._local.warnings = []

        if not isinstance(data, dict):
            raise InvalidSaveError(
                f"save must be a dict, got {_type_name(data)}"
            )

        save, refs = _validate_and_clone(data, self._ref_keys)

        source_version = self._read_save_version(save)
        target_version = _as_version(target, what="target version")

        if source_version < 1:
            raise InvalidVersionError(
                f"invalid save version {source_version}; must be >= 1"
            )
        if source_version > self._latest:
            raise UnknownVersionError(
                f"unknown save version {source_version}; "
                f"latest supported is {self._latest}"
            )
        if target_version < 1:
            raise InvalidVersionError(
                f"invalid target version {target_version}; must be >= 1"
            )
        if target_version > self._latest:
            raise UnknownVersionError(
                f"unsupported target version {target_version}; "
                f"latest supported is {self._latest}"
            )
        if target_version < source_version:
            raise DowngradeError(
                f"cannot downgrade from version {source_version} "
                f"to {target_version}"
            )

        stored_checksum = save.get("checksum", _MISSING)
        if stored_checksum is not _MISSING and stored_checksum is not None:
            actual = self.checksum(save)
            if stored_checksum != actual:
                raise ChecksumMismatchError(
                    "checksum mismatch for source save: "
                    f"stored {stored_checksum!r}, computed {actual!r}"
                )

        if target_version == source_version:
            return save

        # 迁移期间移除旧校验和，迁移完成后统一重算。
        if stored_checksum is not _MISSING:
            del save["checksum"]

        warnings = []
        entity_locations = _collect_entities(save)
        ctx = (refs, entity_locations)
        for version in range(source_version, target_version):
            migration = self.MIGRATIONS.get(version)
            if migration is None:
                raise UnknownVersionError(
                    f"no migration registered from version {version}"
                )
            migration(self, save, warnings, ctx)
            save["version"] = version + 1

        save["checksum"] = self.checksum(save)
        self._local.warnings = warnings
        return save

    def warnings(self):
        """返回最近一次 :meth:`migrate` 在该线程记录的警告列表。"""

        return list(getattr(self._local, "warnings", ()))

    @staticmethod
    def checksum(data):
        """计算存档的整体校验和（十六进制字符串）。

        计算时忽略顶层的 ``checksum`` 字段本身，并对键顺序不敏感。
        """

        canonical = _canonical_json(data)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify(self, data):
        """校验存档中的 ``checksum`` 是否与内容一致。

        缺少 ``checksum`` 字段视为不通过，返回 ``False``。
        """

        if not isinstance(data, dict):
            raise InvalidSaveError(
                f"save must be a dict, got {_type_name(data)}"
            )
        stored = data.get("checksum")
        if stored is None:
            return False
        return stored == self.checksum(data)

    def clear_warnings(self):
        """清空当前线程的警告记录。"""

        self._local.warnings = []

    # -- 内部方法 ---------------------------------------------------------

    def _read_save_version(self, save):
        if "version" not in save:
            raise InvalidVersionError("save has no 'version' field")
        return _as_version(save["version"], what="save version")

    def _fill_default(self, save, field, source_version, target_version, warnings):
        if field in save:
            return
        default = self._defaults[field]
        save[field] = default
        message = (
            f"missing field '{field}' during v{source_version}->"
            f"v{target_version}; filled with default {default!r}"
        )
        warnings.append(
            MigrationWarning(
                message,
                code="missing_field",
                field=field,
                version=target_version,
                default=default,
                path=f"save.{field}",
            )
        )


# 模块级别名，方便调用方检查/自定义迁移注册表。
MIGRATIONS = SaveMigrate.MIGRATIONS

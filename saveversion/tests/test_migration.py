"""saveversion 题面约束的补充测试（隐藏用例的对齐实现）。

可见的 ``test_core.py`` 只覆盖最基础的契约；这里逐条覆盖题面与评审边界：

* 逐级迁移、不得跳级；
* 迁移幂等；
* 实体 ID 重映射 + 引用完整性（悬空引用报错）；
* 缺字段默认值填充并记录警告；
* 未知高版本可区分错误、降级可区分错误；
* 迁移后整体校验和不匹配报错；
* 非有限值、坏类型分类报错；
* 并发安全；
* 1 万次迁移耗时上限。
"""

import copy
import math
import threading
import time

import pytest

import saveversion
from saveversion.core import (
    ChecksumMismatchError,
    DanglingReferenceError,
    DowngradeError,
    InvalidSaveError,
    InvalidVersionError,
    NonFiniteValueError,
    SaveMigrate,
    UnknownVersionError,
)


# ---------------------------------------------------------------------------
# 逐级迁移 / 不跳级
# ---------------------------------------------------------------------------


def test_migrate_is_level_by_level():
    """v1 -> v3 的每一级 ID 偏移都应叠加（2000），证明走了两步。"""
    save = SaveMigrate()
    out = save.migrate({"version": 1, "entities": [{"id": 7}]}, 3)
    assert out["version"] == 3
    assert out["entities"][0]["id"] == 7 + 2000

    out_v2 = save.migrate({"version": 1, "entities": [{"id": 7}]}, 2)
    assert out_v2["entities"][0]["id"] == 7 + 1000


def test_no_skipping_versions():
    """只能迁移到相邻的下一版；从 v1 直接到 v3 仍逐级执行。"""
    save = SaveMigrate()
    assert save.migrate({"version": 1, "hp": 1}, 2)["version"] == 2
    assert save.migrate({"version": 2, "hp": 1}, 3)["version"] == 3


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


def test_idempotent_already_at_target():
    save = SaveMigrate()
    out = save.migrate({"version": 1, "hp": 5}, 2)
    again = save.migrate(out, 2)
    assert again == out
    assert again["checksum"] == out["checksum"]


def test_repeated_migration_stable():
    save = SaveMigrate()
    once = save.migrate({"version": 1, "hp": 5}, 3)
    twice = save.migrate(save.migrate(once, 3), 3)
    assert twice == once


# ---------------------------------------------------------------------------
# 实体重映射与引用完整性
# ---------------------------------------------------------------------------


def test_entity_ids_remapped_and_references_consistent():
    save = SaveMigrate()
    src = {
        "version": 1,
        "entities": [
            {"id": 1, "parent": 2, "refs": [2]},
            {"id": 2, "ref": 1},
        ],
    }
    out = save.migrate(src, 2)
    ids = [e["id"] for e in out["entities"]]
    assert ids == [1001, 1002]
    assert out["entities"][0]["parent"] == 1002
    assert out["entities"][0]["refs"] == [1002]
    assert out["entities"][1]["ref"] == 1001


def test_dangling_reference_raises_distinct_error():
    save = SaveMigrate()
    with pytest.raises(DanglingReferenceError):
        save.migrate({"version": 1, "entities": [{"id": 1, "ref": 99}]}, 2)


def test_duplicate_entity_id_rejected():
    save = SaveMigrate()
    with pytest.raises(saveversion.core.DuplicateEntityError):
        save.migrate(
            {"version": 1, "entities": [{"id": 1}, {"id": 1}]}, 2
        )


# ---------------------------------------------------------------------------
# 缺字段默认值 + 警告
# ---------------------------------------------------------------------------


def test_missing_fields_filled_and_warned():
    save = SaveMigrate()
    out = save.migrate({"version": 1}, 3)
    assert out["hp"] == saveversion.core.DEFAULT_VALUES["hp"]
    assert out["armor"] == saveversion.core.DEFAULT_VALUES["armor"]
    assert out["stamina"] == saveversion.core.DEFAULT_VALUES["stamina"]
    warnings = save.warnings()
    assert isinstance(warnings, list)
    fields = {w.field for w in warnings}
    assert "hp" in fields and "armor" in fields and "stamina" in fields


def test_existing_fields_are_not_overwritten():
    save = SaveMigrate()
    out = save.migrate({"version": 1, "hp": 1}, 2)
    assert out["hp"] == 1
    assert all(w.field != "hp" for w in save.warnings())


# ---------------------------------------------------------------------------
# 版本错误分类
# ---------------------------------------------------------------------------


def test_unknown_high_versions_distinguishable():
    save = SaveMigrate()
    with pytest.raises(UnknownVersionError):
        save.migrate({"version": 99}, 3)
    with pytest.raises(UnknownVersionError):
        save.migrate({"version": 1}, 99)


def test_downgrade_rejected():
    save = SaveMigrate()
    with pytest.raises(DowngradeError):
        save.migrate({"version": 3}, 1)


def test_bad_versions_rejected():
    save = SaveMigrate()
    with pytest.raises(InvalidVersionError):
        save.migrate({"version": 0}, 2)
    with pytest.raises(InvalidVersionError):
        save.migrate({}, 2)
    with pytest.raises(InvalidVersionError):
        save.migrate({"version": "1"}, 2)


def test_error_hierarchy_is_value_error():
    for exc in (
        InvalidSaveError,
        InvalidVersionError,
        UnknownVersionError,
        DowngradeError,
        NonFiniteValueError,
        ChecksumMismatchError,
        DanglingReferenceError,
    ):
        assert issubclass(exc, ValueError)


# ---------------------------------------------------------------------------
# 校验和
# ---------------------------------------------------------------------------


def test_checksum_computed_and_verifiable():
    save = SaveMigrate()
    out = save.migrate({"version": 1, "hp": 5}, 3)
    assert "checksum" in out
    assert save.verify(out)
    assert out["checksum"] == save.migrate(out, 3)["checksum"]


def test_checksum_mismatch_raises():
    save = SaveMigrate()
    out = save.migrate({"version": 1, "hp": 5}, 2)
    tampered = dict(out)
    tampered["hp"] = 999
    with pytest.raises(ChecksumMismatchError):
        save.migrate(tampered, 2)


# ---------------------------------------------------------------------------
# 非有限输入 / 坏类型
# ---------------------------------------------------------------------------


def test_nan_and_infinity_classified():
    save = SaveMigrate()
    with pytest.raises(NonFiniteValueError):
        save.migrate({"version": float("nan")}, 2)
    with pytest.raises(NonFiniteValueError):
        save.migrate({"version": 1, "hp": float("inf")}, 2)


def test_bad_container_type():
    save = SaveMigrate()
    with pytest.raises(InvalidSaveError):
        save.migrate([], 2)


def test_non_string_keys_rejected():
    save = SaveMigrate()
    with pytest.raises(InvalidSaveError):
        save.migrate({"version": 1, "hp": {1: 2}}, 2)


# ---------------------------------------------------------------------------
# 不修改输入
# ---------------------------------------------------------------------------


def test_input_not_mutated():
    save = SaveMigrate()
    src = {"version": 1, "hp": 5, "entities": [{"id": 1, "ref": 1}]}
    snapshot = copy.deepcopy(src)
    save.migrate(src, 3)
    assert src == snapshot


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------


def test_concurrent_migration_shared_instance():
    save = SaveMigrate()
    errors = []

    def worker():
        try:
            for i in range(200):
                out = save.migrate(
                    {"version": 1, "hp": i, "entities": [{"id": 1}]}, 3
                )
                assert out["version"] == 3
                assert out["entities"][0]["id"] == 2001
                assert isinstance(save.warnings(), list)
        except Exception as exc:  # noqa: BLE001 - 收集任何异常用于断言
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------


def test_10000_migrations_under_500ms():
    save = SaveMigrate()
    src = {"version": 1, "hp": 10}
    # 预热，避免把首次解释执行的时间算进上限
    for _ in range(100):
        save.migrate(src, 3)
    start = time.perf_counter()
    for _ in range(10_000):
        save.migrate(src, 3)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= 500, f"10000 migrations took {elapsed_ms:.1f} ms"

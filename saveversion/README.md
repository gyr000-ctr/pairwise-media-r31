# saveversion

## 题面（可直接粘贴）

子目录：saveversion。存档版本迁移。约束：必须按版本链逐级升级（v1→v2→v3），不得跳级；迁移函数幂等（重复应用不变）；缺字段用默认值填充且记录警告；未知高版本拒绝加载并给出可区分错误；迁移后校验和不匹配要报错；1 万次不超过 400 ms。验收：cd saveversion && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd saveversion && python -m pytest tests -q
```

初始 15 failed（其中 `v1_to_v2` 用例引用了未定义的 `_save` 工厂函数，已补齐）。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

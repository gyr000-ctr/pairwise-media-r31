# hitstop

## 题面（可直接粘贴）

子目录：hitstop。打击停顿（时间冻结）。约束：冻结期间逻辑时间不推进（≤1e-12）而渲染时间正常推进；冻结时长可叠加但不超过上限；冻结结束后按剩余 dt 补偿（不丢时间）；10 万次更新不超过 200 ms；非法时长可区分报错。验收：cd hitstop && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd hitstop && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

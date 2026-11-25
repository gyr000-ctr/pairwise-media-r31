# portalastar

## 题面（可直接粘贴）

子目录：portalastar。带传送门的 A*。约束：传送门两端代价对称且被正确纳入邻接；路径代价等于 Dijkstra 最优（≤1e-9）；传送门不重复使用造成环；不可达返回 None；传送门参数非法可区分报错；512x512 不超过 700 ms。验收：cd portalastar && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd portalastar && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

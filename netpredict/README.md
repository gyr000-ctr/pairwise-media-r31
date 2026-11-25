# netpredict

## 题面（可直接粘贴）

子目录：netpredict。客户端预测与和解。约束：预测状态与服务器状态在无误差时逐位一致；有误差时按平滑因子收敛且不跳变；回滚重放必须按输入序列逐帧重算（与直接重放一致 ≤1e-12）；丢包时保持最近确认输入；10 万帧不超过 500 ms；非法输入序号可区分报错。验收：cd netpredict && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd netpredict && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

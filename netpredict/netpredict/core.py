"""netpredict 核心 API（初始未实现）。"""


class Predict:
    def __init__(self, smooth):
        """构造"""
        raise NotImplementedError

    def apply(self, seq, inp):
        """输入"""
        raise NotImplementedError

    def predict(self):
        """预测"""
        raise NotImplementedError

    def reconcile(self, state, seq):
        """和解"""
        raise NotImplementedError

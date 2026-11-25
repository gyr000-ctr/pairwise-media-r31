"""combo 核心 API（初始未实现）。"""


class Combo:
    def __init__(self, window, decay, cap):
        """构造"""
        raise NotImplementedError

    def hit(self, t):
        """命中"""
        raise NotImplementedError

    def tick(self, dt):
        """推进"""
        raise NotImplementedError

    def count(self):
        """计数"""
        raise NotImplementedError

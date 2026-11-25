"""hitstop 核心 API（初始未实现）。"""


class HitStop:
    def __init__(self, max_freeze):
        """构造"""
        raise NotImplementedError

    def freeze(self, dur):
        """触发"""
        raise NotImplementedError

    def tick(self, dt):
        """推进"""
        raise NotImplementedError

    def logic_time(self):
        """逻辑时间"""
        raise NotImplementedError

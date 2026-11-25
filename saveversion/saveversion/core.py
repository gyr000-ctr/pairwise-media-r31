"""saveversion 核心 API（初始未实现）。"""


class SaveMigrate:
    def migrate(self, data, target):
        """迁移"""
        raise NotImplementedError

    def warnings(self):
        """警告列表"""
        raise NotImplementedError

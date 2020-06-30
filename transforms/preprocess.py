from abc import ABCMeta, abstractmethod
import copy


class Preprocess(metaclass=ABCMeta):  # 利用abc模块实现抽象类
    # 抽象类中只能有抽象方法（没有实现功能），该类不能被实例化，只能被继承，且子类必须实现抽象方法。
    @abstractmethod  # 定义抽象方法，无需实现功能，但是子类必须实现该抽象方法，否则报错
    def __call__(self, image, anns, meta, mask_miss):
        """Implementation of preprocess operation."""

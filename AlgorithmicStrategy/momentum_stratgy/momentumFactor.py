from abc import ABC, abstractmethod
from array import array
from dataclasses import dataclass, field
import math


from AlgorithmicStrategy.momentum_stratgy.utils import signal
from AlgorithmicStrategy.momentum_stratgy.modelType import modelType


@dataclass
class Indicator(ABC):
    """
    Indicator: 指标基类

    Indicator对外暴露两个接口

    1. values, 一个array.array类型的数值array, 指标的计算值
    2. update, 用于更新values值
    """

    values: array = field(default_factory=lambda: array("d"))

    @abstractmethod
    def update(self):
        """
        更新values的值
        """


@dataclass
class SingleInputIndictator(Indicator):
    """
    单输入Indicator
    """

    inputs: array = field(default_factory=lambda: array("d"))
    args: tuple[int, ...] = field(default_factory=tuple)

    def __init__(self, inputs: array | Indicator | None, *args: int):
        """
        单输入指标

        :param inputs: 输入array, 会随着时间不断append数据
        :param args: 指标参数, 比如MA(5)中的5
        """
        self.values = array("d")
        if isinstance(inputs, array):
            self.inputs = inputs
        elif isinstance(inputs, Indicator):
            self.inputs = inputs.values
        elif inputs is None:
            self.inputs = array("d")
        else:
            raise TypeError(f"inputs should be array or Indicator, got `{inputs}`")

        self.args = args
        for arg in args:
            assert arg > 0 and isinstance(arg, int)


@dataclass
class RSI(SingleInputIndictator):
    """
    RSI : Relative Strength Index
    """

    __slots__ = ["inputs", "period", "diff", "values", "newup", "newdn"]

    def __init__(self, inputs: array | None = None, period: int = 70):
        super().__init__(inputs, period)
        self.period = period
        self.diff = array("d")
        self.newup = None
        self.newdn = None

    def update(self):
        n = self.period
        dif = self.inputs[-1] - self.inputs[-2] if len(self.inputs) > 1 else 0
        if len(self.inputs) <= n:
            self.diff.append(dif)
            self.values.append(float("nan"))
        elif len(self.inputs) == n + 1:
            self.diff.append(dif)
            self.newup = sum(map(lambda x: max(x, 0), self.diff)) / n  # type: ignore
            self.newdn = sum(map(lambda x: -min(x, 0), self.diff)) / n
            newvalue = 100 * self.newup / (self.newup + self.newdn)
            self.values.append(newvalue)
        else:
            self.newup = (self.newup * (n - 1) + max(dif, 0)) / n
            self.newdn = (self.newdn * (n - 1) - min(dif, 0)) / n
            newvalue = 100 * self.newup / (self.newup + self.newdn)
            self.values.append(newvalue)


@dataclass
class MA(SingleInputIndictator):
    """
    简单移动平均线:  Simple Moving Average
    可从零开始一个一个传参,也可一开始就传入一系列参数,如果参数数量大于n(n=period),只选取前n个值计算初值
    """

    __slots__ = ["inputs", "values", "period"]

    def __init__(self, inputs: array | None = None, period: int = 300):
        super().__init__(inputs, period)
        self.period = period

    def update(self):
        n: int = self.period
        if len(self.values) < 1 or math.isnan(self.values[-1]):
            if len(self.inputs) < n:
                self.values.append(float("nan"))
            else:
                newvalue = sum(self.inputs[-n:]) / n
                self.values.append(newvalue)
        else:
            newvalue: float = (
                self.values[-1] + (self.inputs[-1] - self.inputs[-1 - n]) / n
            )
            self.values.append(newvalue)


@dataclass
class PMM(SingleInputIndictator):
    def __init__(self, inputs: array | None = None, period: int = 300):
        super().__init__(inputs, period)
        self.uptrends = array("i")
        self.pmm = MA(self.uptrends, period)
        self.period = period

    def update(self):
        if len(self.inputs) == 1:
            self.uptrends.append(0)
        else:
            new_up = signal(self.inputs[-1] > self.inputs[-2])
            self.uptrends.append(new_up)
        self.pmm.update()
        self.values.append(self.pmm.values[-1])


class momentumType(modelType):
    ma_period: int
    rsi_period: int

    def __init__(
        self,
        ma_period: int = 30,
        rsi_period: int = 14,
        pmm_period: int = 100,
        lma_period: int = 60,
    ) -> None:
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.lma_period = lma_period
        self.price = array("d")
        self.ma = MA(self.price, ma_period)
        self.rsi = RSI(self.price, rsi_period)
        self.lma = MA(self.price, lma_period)
        self.pmm = PMM(self.price, pmm_period)

    def model_update(self, price: float) -> dict[str, float]:
        self.price.append(price)
        self.ma.update()
        self.rsi.update()
        self.lma.update()
        self.pmm.update()
        if (
            math.isnan(self.ma.values[-1])
            or math.isnan(self.rsi.values[-1])
            or math.isnan(self.pmm.values[-1])
        ):
            return
        return {
            'ma': self.ma.values[-1],
            'rsi': self.rsi.values[-1],
            'lma': self.lma.values[-1],
        }

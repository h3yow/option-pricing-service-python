"""对外可区分的错误类型体系。

所有数值求解 / 校验失败都映射为这里的类型，绝不把底层数值库异常直接透出。
"""

from __future__ import annotations


class PricingError(Exception):
    """服务内所有可预期错误的基类。"""

    code: str = "pricing_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


# ---- 输入与合约状态 -------------------------------------------------------

class ValidationError(PricingError):
    """参数越界或非法（如负波动率、非正行权价）。"""

    code = "validation_error"


class ContractExpiredError(PricingError):
    """合约已到期，无法估值。"""

    code = "contract_expired"


class ModelNotApplicableError(PricingError):
    """所选模型不适用于该合约类型。"""

    code = "model_not_applicable"


# ---- 数值求解 -------------------------------------------------------------

class CalibrationError(PricingError):
    """校准失败基类。"""

    code = "calibration_error"


class NoConvergenceError(CalibrationError):
    """优化 / 求根迭代不收敛。"""

    code = "no_convergence"


class NoSolutionError(CalibrationError):
    """解不存在（如目标价格超出模型可达区间）。"""

    code = "no_solution"


class NumericalInstabilityError(CalibrationError):
    """数值不稳定（如中间结果出现 NaN / Inf）。"""

    code = "numerical_instability"

"""服务层单测：并发与串行结论一致、批量与单笔一致、边界合约归类。"""

import logging
import threading

import pytest

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import (Contract, ExerciseStyle, MarketParams,
                                      Model, OptionType)
from option_pricing.errors import (ContractExpiredError,
                                   ModelNotApplicableError, NoSolutionError,
                                   ValidationError)
from option_pricing.pricing.black_scholes import bs_price
from option_pricing.service import PricingService

log = logging.getLogger(__name__)

M = MarketParams(spot=100.0, volatility=0.2, risk_free_rate=0.05)
TARGET = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)
EU = Contract("EU1", OptionType.CALL, ExerciseStyle.EUROPEAN, 100.0, 1.0)
AZ = Contract("AZ1", OptionType.CALL, ExerciseStyle.ASIAN_ARITHMETIC,
              100.0, 1.0, averaging_points=12)


@pytest.fixture(scope="module")
def service():
    return PricingService(CalibrationSession(TARGET))


def test_single_equals_batch_bitwise(service):
    single_eu = service.price(EU, M)
    single_az = service.price(AZ, M)
    batch = service.price_batch([(EU, M), (AZ, M), (EU, M)])
    log.info("单笔 EU=%r AZ=%r; 批量=%r; 判定依据: 同一计算路径必须逐位一致",
             single_eu, single_az, batch)
    assert batch[0] == single_eu and batch[1] == single_az
    assert batch[2] == single_eu


def test_concurrent_equals_serial(service):
    serial = [service.price(EU, M), service.price(AZ, M)]
    results: list = [None] * 16

    def work(i):
        results[i] = service.price(EU if i % 2 == 0 else AZ, M)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("串行结果 %s; 并发 16 线程结果集合=%s; 判定依据: 并发不得"
             "互相污染, 与串行逐位一致", serial, set(map(repr, results)))
    for i, r in enumerate(results):
        assert r == serial[i % 2]


def test_expired_contract_rejected(service):
    for mat in (0.0, -0.5):
        log.info("maturity=%s; 判定依据: 非正期限 -> contract_expired", mat)
        with pytest.raises(ContractExpiredError):
            service.price(Contract("X", OptionType.CALL,
                                   ExerciseStyle.EUROPEAN, 100.0, mat), M)


def test_out_of_range_params_rejected(service):
    cases = [
        (Contract("X", OptionType.CALL, ExerciseStyle.EUROPEAN, -1.0, 1.0), M,
         "负行权价"),
        (EU, MarketParams(spot=0.0, volatility=0.2, risk_free_rate=0.05),
         "零标的价"),
        (EU, MarketParams(spot=100.0, volatility=9.0, risk_free_rate=0.05),
         "波动率越界"),
        (EU, MarketParams(spot=100.0, volatility=0.2, risk_free_rate=0.9),
         "利率越界"),
        (Contract("X", OptionType.CALL, ExerciseStyle.ASIAN_ARITHMETIC,
                  100.0, 1.0, averaging_points=0), M, "平均点数量为 0"),
    ]
    for contract, market, why in cases:
        log.info("非法输入(%s): %s %s; 判定依据: -> validation_error",
                 why, contract, market)
        with pytest.raises(ValidationError):
            service.price(contract, market)


def test_model_not_applicable(service):
    log.info("亚式合约强制 black_scholes; 判定依据: -> model_not_applicable")
    with pytest.raises(ModelNotApplicableError):
        service.price(AZ, M, model=Model.BLACK_SCHOLES)


def test_no_solution_propagates_as_typed_error():
    svc = PricingService(CalibrationSession(1.0))  # 目标价低于看涨下限
    log.info("目标价 1.0 不可达; 判定依据: 服务层应透出 no_solution 而非"
             "底层异常或缺失值")
    with pytest.raises(NoSolutionError):
        svc.price(EU, M)


def test_batch_reuses_calibration(service):
    before = service._session.size
    service.price_batch([(EU, M)] * 8)
    after = service._session.size
    log.info("批量 8 笔同一合约前后缓存 %d -> %d; 判定依据: 校准只发生一次",
             before, after)
    assert after == before

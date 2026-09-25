# option-pricing-service-python

本地期权定价与风险指标服务：FastAPI 提供 HTTP 接口，内置行情参数与合约
定义，同一合约在单笔入口、批量入口、并发请求与参数变更下结论一致且可解释。

## 运行

```bash
uv sync
uv run fastapi run main.py        # 或 uv run uvicorn main:app
```

## 支持的合约与模型范围

| 合约类型 | 行权方式 | 定价模型 | 敏感度 |
|---|---|---|---|
| 欧式看涨/看跌 | `european` | Black-Scholes 解析解 | 解析 delta/gamma/vega/theta/rho |
| 算术平均亚式看涨/看跌（路径相关） | `asian_arithmetic` | 确定性 Monte Carlo（对偶变量，4 万路径×2） | 中心有限差分 + 共同随机数 |

- vega/rho 按 1% 变动报告，theta 按每自然日报告。
- 蒙特卡洛 seed 由 `(contract, market, calibrated_vol)` 经 SHA-256 确定性
  派生，与入口和并发时序无关，因此同参数结果逐位可复现；标准误随结果在
  `diagnostics.mc_std_err` 中返回，超过价格 2% 判定为数值不稳定并拒绝。
- 亚式合约强制选择 `black_scholes` 会被拒绝（`model_not_applicable`）。

## 接口

- `GET /market` / `POST /market`：查看 / 更新内置行情（spot、volatility、
  risk_free_rate）。更新前先用新参数试算，不可达时整体拒绝，不留半更新状态。
- `POST /price`：`{"contract_id": ..., "model": ...?}` 单笔估值。
- `POST /price/batch`：`{"items": [{contract_id, model?}, ...]}`（≤64 项）
  批量估值，有界线程池（4 worker）并发，结果顺序与输入一致。
- 内置合约：`EU-CALL-100`、`EU-PUT-100`、`ASIAN-CALL-100`、
  `EXPIRED-CALL`（已到期，用于验证拒绝路径）。

## 校准与结果复用规则

- 隐含波动率按市场惯例定义为 Black-Scholes 隐含波动率：校准会话以内置
  目标市价为输入，用 Brent 求根反解；定价时把校准后的波动率代入该合约
  对应的模型（BS 解析式或 MC）。
- 缓存键为 `(contract, market_params)` 整体：标的价格、波动率或利率任一
  变化都会产生新键，强制重新校准，绝不沿用旧结果。
- 并发同一键采用 single-flight：仅一个线程求解，其余等待共享同一结果；
  批量请求自动复用已完成校准。
- 缓存为容量有界 LRU（默认 128 项），批量并发度有界（4 线程），资源
  占用可预期。

## 错误分类（不透出底层数值库异常，不静默返回缺失值）

| code | HTTP | 含义 |
|---|---|---|
| `validation_error` | 422 | 参数越界（负行权价、波动率/利率超界等） |
| `contract_expired` | 410 | 合约已到期（剩余期限非正） |
| `model_not_applicable` | 400 | 模型不适用于该合约 |
| `no_solution` | 422 | 目标价超出模型可达区间，隐含波动率不存在 |
| `no_convergence` | 422 | 求根迭代在预算内不收敛 |
| `numerical_instability` | 422 | 中间结果 NaN/Inf 或 MC 标准误过大 |
| `unknown_contract` | 404 | 未知合约 id |

## 验证方法

```bash
uv run pytest -q -s --log-cli-level=INFO
```

测试覆盖（日志打印输入参数与判定依据）：

- 定价引擎：BS 教科书基准值与平价关系；亚式 MC 同 seed 逐位一致、
  价格低于欧式且标准误有界。
- 校准：往返精度、参数变更强制重校准、32 线程 single-flight 一致、
  LRU 有界、`no_solution`/`no_convergence` 分类。
- 服务层：单笔与批量逐位一致、16 线程并发与串行逐位一致、到期与
  越界合约正确归类、批量复用校准。
- 接口层：两入口结论一致、错误码映射、行情更新触发重校准、不可达
  更新原子拒绝、12 并发 HTTP 请求结果唯一。

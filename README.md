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
| 欧式看涨/看跌 | `european` | Black-Scholes 解析解（默认）/ 确定性 Monte Carlo | 解析或有限差分 delta/gamma/vega/theta/rho |
| 算术平均亚式看涨/看跌（路径相关） | `asian_arithmetic` | 确定性 Monte Carlo（对偶变量，4 万路径×2） | 中心有限差分 + 共同随机数 |
| 单障碍欧式看涨/看跌（向上/向下 × 敲入/敲出，支持回扣） | `barrier` | Reiner-Rubinstein 解析式（`barrier_analytic`） | 解析式上的中心有限差分 |

- vega/rho 按 1% 变动报告，theta 按每自然日报告。
- 蒙特卡洛 seed 由 `(contract, market, calibrated_vol)` 经 SHA-256 确定性
  派生，与入口和并发时序无关，因此同参数结果逐位可复现；标准误随结果在
  `diagnostics.mc_std_err` 中返回，超过价格 2% 判定为数值不稳定并拒绝。
- 欧式走 Monte Carlo 时按终端支付模拟，绝不复用亚式的平均支付而返回
  含义不同的数值。
- 障碍合约的回扣约定为"敲出/敲入发生时生效、到期支付"，现值为
  `R·e^{-rT}·P(触碰)`；期初标的价格已越障时直接按已触碰短路返回。
- 模型适用范围双向守住，不匹配的组合在出数之前拒绝
  （`model_not_applicable`）：
  - 亚式合约拒绝 `black_scholes`；
  - 障碍合约只接受 `barrier_analytic`（拒绝 `black_scholes` 与
    `monte_carlo`）；
  - `barrier_analytic` 也只对障碍合约开放（欧式/亚式指定它同样被拒）。
- 障碍解析式已经连续监测（Brownian bridge）蒙特卡洛在 8 分支 ×
  K<>H 两区间 × 有无回扣下交叉验证。

## 接口

- `GET /market` / `POST /market`：查看 / 更新内置行情（spot、volatility、
  risk_free_rate）。更新前先用新参数试算，不可达时整体拒绝，不留半更新状态。
- `POST /price`：`{"contract_id": ..., "model": ...?}` 单笔估值。
- `POST /price/batch`：`{"items": [{contract_id, model?}, ...]}`（≤64 项）
  批量估值，有界线程池（4 worker）并发，结果顺序与输入一致。
- 内置合约：`EU-CALL-100`、`EU-PUT-100`、`ASIAN-CALL-100`、
  `EXPIRED-CALL`（已到期，用于验证拒绝路径）、`UO-CALL-120`
  （上敲出看涨，回扣 2.0）、`DI-PUT-90`（下敲入看跌，回扣 1.0）。

## 校准会话（按客户/按标的隔离）

每个会话持有自己的一组报价（`contract_id -> 目标价`）和一份行情快照，
校准只认本会话的报价；会话之间完全隔离，校准结果互不复用。

- `POST /sessions`：`{"market": {...}, "quotes": {...}, "session_id"?}`
  创建会话（201），返回 `session_id` 与 `version`（创建为 1）。
- `GET /sessions/{id}`：查询当前快照。
- `PUT /sessions/{id}`：更新行情和/或报价。更新后 `version` +1，校准
  缓存整体重建——后续请求必然基于新内容重新校准，不会沿用更新前的
  结果。更新期间并发读取拿到的一定是完整的旧版或完整的新版（不可变
  快照 + 原子替换），绝不出现两半拼接的混合结果。
- `DELETE /sessions/{id}`：删除会话（204）；删除后再访问返回
  `session_not_found`。
- `POST /sessions/{id}/price` / `POST /sessions/{id}/price/batch`：
  会话内估值。返回中的 `session_id` 与 `session_version` 标明该价格
  出自哪个会话的哪一次更新后的内容。
- 会话内没有某合约的报价时按 `quote_not_found` 拒绝，不会拿别的会话
  或内置目标价凑数。

## 容量与并发约束（到顶按可区分原因拒绝）

| 资源 | 上限 | 超限错误 |
|---|---|---|
| 会话总数 | 32 | `capacity_exceeded`（resource=sessions） |
| 每会话报价条数 | 64 | `capacity_exceeded`（resource=quotes_per_session） |
| 每会话校准缓存 | 128 项（LRU） | 自动淘汰，有界 |
| 批量规模 | 64 项 | `capacity_exceeded`（resource=batch_size） |
| 批量并发度 | 4 线程 | 有界线程池 |

会话内批量请求复用同一快照的校准缓存与 single-flight，同参数只校准
一次；跨会话并发与串行逐条执行结果一致。

## 校准与结果复用规则

- 校准目标函数随合约类型选择：欧式/亚式用 BS 解析式（市场惯例），
  单障碍合约用 Reiner-Rubinstein 障碍解析式——障碍报价与 vanilla
  价格没有可比性，用 BS 反解会把合法报价误判为无解。障碍价格对
  波动率可能非单调，求根前先做几何扫描 bracket；目标价不可 bracket
  时按 `no_solution` 拒绝。
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
| `no_solution` | 422 | 目标价在波动率扫描区间内不可 bracket |
| `no_convergence` | 422 | 求根迭代在预算内不收敛 |
| `numerical_instability` | 422 | 中间结果 NaN/Inf/溢出或 MC 标准误过大 |
| `session_not_found` | 404 | 会话不存在或已删除 |
| `quote_not_found` | 422 | 会话内没有该合约的报价 |
| `capacity_exceeded` | 429 | 会话数/报价条数/批量规模到顶 |
| `unknown_contract` | 404 | 未知合约 id |

## 验证方法

```bash
uv run pytest -q -s --log-cli-level=INFO
```

测试覆盖（日志打印输入参数与判定依据）：

- 定价引擎：BS 教科书基准值与平价关系；亚式 MC 同 seed 逐位一致、
  价格低于欧式且标准误有界。
- 障碍合约：8 分支价格与希腊值有限性、in+out=vanilla 平价、回扣对
  in/out 加价一致、期初已触碰短路、解析式对照连续监测 MC、模型
  不匹配双向拒绝、非法障碍参数拒绝、单笔/批量/重复调用逐位一致。
- 校准：往返精度、参数变更强制重校准、32 线程 single-flight 一致、
  LRU 有界、`no_solution`/`no_convergence` 分类。
- 会话：创建/查询/更新/删除生命周期、更新后版本 +1 且强制重新校准、
  50 次更新 × 4 读线程下每条结果 (version, iv) 同源、会话间隔离、
  缺失报价拒绝、会话数/报价条数/批量规模到顶拒绝、批量复用校准、
  跨会话并发与串行一致。
- 服务层：单笔与批量逐位一致、16 线程并发与串行逐位一致、到期与
  越界合约正确归类、批量复用校准。
- 接口层：两入口结论一致、错误码映射、行情更新触发重校准、不可达
  更新原子拒绝、12 并发 HTTP 请求结果唯一、会话端点全生命周期与
  会话内单笔/批量一致。

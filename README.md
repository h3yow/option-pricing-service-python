# option-pricing-service-python

本地期权定价与风险指标服务：FastAPI 提供 HTTP 接口，内置行情参数与合约
定义，同一合约在单笔入口、批量入口、并发请求与参数变更下结论一致且可
解释。支持按客户/标的创建相互隔离的校准会话，并覆盖单障碍类欧式合约。

## 运行

```bash
uv sync
uv run fastapi run main.py        # 或 uv run uvicorn main:app
```

## 支持的合约与模型范围

| 合约类型 | 行权方式 | 可用定价模型 | 敏感度 |
|---|---|---|---|
| 欧式看涨/看跌 | `european` | `black_scholes`（默认） | 解析 delta/gamma/vega/theta/rho |
| 算术平均亚式看涨/看跌 | `asian_arithmetic` | `monte_carlo`（默认，对偶变量 4 万路径×2） | 中心有限差分 + 共同随机数 |
| 单障碍看涨/看跌（上/下 × 敲入/敲出，支持回扣） | `barrier_single` | `barrier_analytic`（默认，Reiner-Rubinstein 闭式）、`monte_carlo` | 解析式中心有限差分 / MC 差分+CRN |

- 模型适用范围双向守住：每种合约只接受上表列出的模型，任何不匹配的
  组合在出数之前以 `model_not_applicable` 拒绝（例如欧式强制
  `monte_carlo`、障碍合约强制 `black_scholes`），绝不返回一个看起来
  正常、含义其实是别的产品的数值。
- 障碍合约约定：连续监测障碍；回扣（`rebate`）在触碰发生时支付
  （解析式首达折现闭式解）；定价时刻标的价格已越过障碍的合约超出
  模型适用范围，以 `validation_error` 明确拒绝。
- 障碍解析式经确定性 MC（Broadie-Glasserman 连续性修正）逐分支交叉
  验证，且满足 敲入+敲出=vanilla（无回扣）平价关系。
- vega/rho 按 1% 变动报告，theta 按每自然日报告。
- 蒙特卡洛 seed 由 `(contract, market, calibrated_vol)` 经 SHA-256
  确定性派生，与入口和并发时序无关，同参数结果逐位可复现；标准误随
  结果在 `diagnostics.mc_std_err` 中返回，超过价格 2% 判定为数值不
  稳定并拒绝。

## 接口

- `GET /market` / `POST /market`：查看 / 更新内置行情（spot、
  volatility、risk_free_rate）。更新前先用新参数试算，不可达时整体
  拒绝，不留半更新状态。
- `POST /price`：`{"contract_id", "model"?, "session_id"?}` 单笔估值。
  给出 `session_id` 时在对应校准会话内估值（使用会话自己的报价与
  行情快照）；否则走内置行情与内置报价表。
- `POST /price/batch`：`{"items": [{contract_id, model?, session_id?},
  ...]}`（≤64 项）批量估值。逐项指定模型与会话；有界线程池
  （4 worker）并发，结果顺序与输入一致；同一会话内相同合约的校准
  自动复用，不会反复校准。
- `POST /sessions`：创建校准会话 `{"client", "underlying", "quotes":
  {contract_id: 目标市价}, "market": {spot, volatility,
  risk_free_rate}}`。创建即试算全部报价，不可达/未知合约整体拒绝。
- `GET /sessions` / `GET /sessions/{id}`：查询会话列表 / 单个会话
  （含当前 version、报价与行情快照）。
- `POST /sessions/{id}`：原子更新报价和/或行情 `{"quotes"?,
  "market"?}`。版本号 +1 并强制重新校准。
- `DELETE /sessions/{id}`：删除会话；后续引用以 `unknown_session`
  拒绝。
- 内置合约：`EU-CALL-100`、`EU-PUT-100`、`ASIAN-CALL-100`、
  `UO-CALL-105`（上看涨敲出，障碍 105，回扣 3）、`DI-PUT-95`（下看跌
  敲入，障碍 95，回扣 0）、`EXPIRED-CALL`（已到期，用于验证拒绝路径）。

## 校准会话与结果复用规则

- 隐含波动率定义：欧式/亚式合约为 Black-Scholes 隐含波动率；障碍合约
  为使其障碍解析式价格等于目标报价的波动率（对该合约本身成立）。
- 每个会话持有自己的报价表与行情快照，校准只认本会话的报价；会话中
  没有报价的合约以 `quote_missing` 拒绝。不同会话的校准结果完全隔离，
  互不复用。
- 求解器先在对数网格（64 点）上扫描符号变化再做 Brent 求根：障碍
  价格对波动率可能非单调，若存在多个解，取与会话行情快照波动率最
  接近者（确定性选择，同一内容必得同一解）。
- 会话内容（报价+行情）是不可变快照，每次更新版本号 +1。估值请求
  单次原子读取快照，因此更新进行中的并发读取要么整体属于旧内容、
  要么整体属于新内容，绝不出现两半拼起来的混合结果。返回的
  `session_id` 与 `session_version` 标明结果出自哪个会话、哪次更新
  后的内容。
- 更新先构造候选状态并试算全部报价，全部可解才一次性切换；任何一步
  失败都保持旧状态不变（原子拒绝）。
- 缓存键为 `(版本, 合约, 行情)` 整体：更新即失效，绝不沿用旧结果；
  并发同一键采用 single-flight，仅一个线程求解，其余等待共享结果。

## 容量与并发约束（到顶按可区分原因拒绝，不无限增长）

| 资源 | 上限 | 拒绝原因 |
|---|---|---|
| 校准会话数量 | 64 | `capacity_exceeded`（resource=`sessions`，HTTP 429） |
| 每会话校准缓存 | 128 项（LRU） | 容量有界，自动淘汰最久未用 |
| 批量规模 | 64 项 | `capacity_exceeded`（resource=`batch_size`） |
| 批量线程池 | 4 worker | 有界并发 |
| 全局并发估值 | 32 | `capacity_exceeded`（resource=`concurrency`，HTTP 429） |

## 错误分类（不透出底层数值库异常，不静默返回缺失值）

| code | HTTP | 含义 |
|---|---|---|
| `validation_error` | 422 | 参数越界（负行权价、波动率/利率超界、已触碰障碍等） |
| `contract_expired` | 410 | 合约已到期（剩余期限非正） |
| `model_not_applicable` | 400 | 模型不适用于该合约 |
| `no_solution` | 422 | 目标价超出模型可达区间，隐含波动率不存在 |
| `no_convergence` | 422 | 求根迭代在预算内不收敛 |
| `numerical_instability` | 422 | 中间结果 NaN/Inf/溢出或 MC 标准误过大 |
| `unknown_contract` | 404 | 未知合约 id |
| `unknown_session` | 404 | 会话不存在或已删除 |
| `quote_missing` | 422 | 会话中没有该合约的报价 |
| `capacity_exceeded` | 429 | 容量到顶（detail.resource 区分 sessions/batch_size/concurrency） |

## 验证方法

```bash
uv run pytest -q -s --log-cli-level=INFO
```

测试覆盖（日志打印输入参数与判定依据）：

- 定价引擎：BS 教科书基准值与平价关系；亚式 MC 同 seed 逐位一致、
  价格低于欧式且标准误有界。
- 障碍合约：8 分支（看涨/看跌 × 上/下 × 敲入/敲出）解析式与 MC 交叉
  验证、敲入+敲出=vanilla 平价、退化分支（K 越过障碍）为零、回扣
  下界、希腊值有限性、已触碰障碍拒绝、模型守卫双向拒绝、单笔与批量
  逐位一致。
- 校准：往返精度、参数变更强制重校准、32 线程 single-flight 一致、
  LRU 有界、`no_solution`/`no_convergence` 分类。
- 会话：创建/查询/更新/删除生命周期、更新后强制重校准（报价与行情）、
  不可达更新原子拒绝、缺报价拒绝、会话间隔离、更新进行中 8 线程并发
  读取只见整体旧/整体新、跨会话并发与串行逐位一致、批量复用同会话
  校准、会话数/缓存/批量规模/并发槽位到顶拒绝。
- 服务层：单笔与批量逐位一致、16 线程并发与串行逐位一致、到期与
  越界合约正确归类。
- 接口层：两入口结论一致、错误码映射、行情更新触发重校准、不可达
  更新原子拒绝、会话 HTTP 生命周期与错误码、跨会话 12 并发请求结果
  唯一。

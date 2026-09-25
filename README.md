# 本地期权定价与风险指标服务

一个通过 HTTP 接口调用的本地期权定价服务，使用内置行情参数与合约定义完成
欧式期权与单障碍（路径相关）期权的估值和主要希腊值计算，并提供基于行情报价的
**波动率校准会话**。服务使用 FastAPI 实现，定价核仅依赖 Python 标准库，
不依赖任何第三方数值库。

## 1. 支持的合约与模型范围

| 合约类型 | 取值 (`style`) | 支持的模型 (`model`) | 定价方法 |
| --- | --- | --- | --- |
| 欧式期权 | `european` | `black_scholes` | Black–Scholes–Merton 解析公式（含连续股息率 `q`） |
| 单障碍期权 | `barrier` | `reiner_rubinstein` | Reiner–Rubinstein 解析公式（连续监测） |

- 看涨/看跌：`option_type ∈ {call, put}`。
- 障碍类型（`barrier_type`）：
  `up_and_in` / `up_and_out` / `down_and_in` / `down_and_out`，共 4 种，
  与 call/put、以及障碍相对行权价的高低共同构成 16 个解析分支。
- 障碍期权支持现金回扣 `rebate`：
  - **敲出型**：在首次触碰障碍时支付（按触碰时刻贴现）；
  - **敲入型**：若到期前始终未触碰，在到期日支付。
- 不支持的组合会被明确拒绝：例如用 `reiner_rubinstein` 给欧式合约定价、
  用 `black_scholes` 给障碍合约定价，或在估值日障碍已被击穿（解析公式假设
  障碍在初始时处于"存活"状态）。

### 风险指标（Greeks）

| 指标 | 定义 |
| --- | --- |
| `delta` | ∂V/∂S |
| `gamma` | ∂²V/∂S² |
| `vega`  | ∂V/∂σ（按 1 个波动率单位） |
| `theta` | ∂V/∂t（日历时间，年化；等于 −∂V/∂T） |
| `rho`   | ∂V/∂r（按 1 个利率单位） |

- 欧式期权：全部使用解析导数。
- 障碍期权：价格用解析公式，Greeks 用同一解析价格的中心有限差分
  （到期附近 `theta` 自动切换为单侧差分）。因为差分直接作用在最终对外使用的
  定价函数上，价格与敏感度始终自洽。

### 计息与日期约定

- 到期期限：ACT/365（固定），`tenor_years = (expiry − as_of)/365`。
- 默认估值日 `as_of = 2026-09-25`（为使本地结果可复现而固定），
  也可在每个定价条目中显式传入 `as_of`。
- 参数合法区间（超出即拒绝）：
  现货/行权价 `(0, 1e9]`，无风险利率与股息率 `[-0.99, 5]`，
  波动率 `(0, 5]`，期限 `(0, 50]` 年，回扣 `≥ 0`。

## 2. 接口

服务随 `uvicorn`/FastAPI 启动，入口应用为 `main:app`（由 `app.api:create_app`
构造）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | `/health` | 健康检查、默认估值日、支持的合约/模型枚举 |
| GET  | `/v1/catalog` | 内置行情参数与示例合约定义 |
| POST | `/v1/sessions` | 用行情 + 报价创建校准会话（创建时即校准） |
| GET  | `/v1/sessions/{id}` | 查询会话版本与已校准波动率 |
| PATCH| `/v1/sessions/{id}/market` | 修改现货/利率/股息，**旧校准立即失效** |
| PUT  | `/v1/sessions/{id}/quotes` | 替换报价，**旧校准立即失效** |
| DELETE | `/v1/sessions/{id}` | 关闭会话 |
| POST | `/v1/price` | 单笔定价（`items` 恰好 1 条） |
| POST | `/v1/price/batch` | 批量定价（逐条返回成功/失败，按下标对齐） |

定价条目示例（无会话、显式给定波动率）：

```json
{
  "items": [
    {
      "contract": {
        "contract_id": "demo-uoc-120",
        "style": "barrier",
        "option_type": "call",
        "strike": 100.0,
        "expiry": "2027-09-25",
        "barrier": 120.0,
        "barrier_type": "up_and_out",
        "rebate": 2.0
      },
      "model": "reiner_rubinstein",
      "sigma_override": 0.25
    }
  ]
}
```

使用校准时令 `session_id` 指向已有会话、且**不**传 `sigma_override`，
估值结果中的 `calibrated=true` 且 `sigma_used` 为会话校准出的波动率。

### 单笔与批量一致性

`/v1/price` 与 `/v1/price/batch` 最终都调用同一个内部函数
`PricingService._price_item`，不存在两套定价路径。因此同一合约/行情/模型：

- 单笔与批量的价格、Greeks 逐字段完全相等（测试中以冻结数据结构相等性断言）；
- 串行与并发批量的结果也完全一致。

## 3. 校准与结果复用规则

### 校准方法

- 会话持有一组行情参数（现货 `S`、无风险利率 `r`、股息率 `q`）和一组欧式
  期权报价（行权价、到期、call/put、权利金）。
- 先对每条报价求**单条隐含波动率**（Black–Scholes 反解，Brent 求根）；
  若某条权利金落在无套利区间之外（≤ 内在价值，或 ≥ 波动率趋于无穷时的
  渐近上限），该报价不存在隐含波动率，直接报 `CALIBRATION_NO_SOLUTION`。
- 再以**等权重最小二乘**拟合单一平坦波动率 σ（黄金分割一维极小化），
  使模型价与报价的平方误差最小。
- 若各报价隐含波动率彼此差异过大（极差/均值 > 50%，说明存在明显
  微笑/偏斜，单一平坦波动率不具代表性），报
  `CALIBRATION_DID_NOT_CONVERGE`，不会静默返回一个未加说明的近似值。

### 失效与重新校准（关键规则）

- 会话带单调递增的 `version`。**任何**对现货、利率、股息或报价的修改都会
  令版本号 +1 并清空已缓存的校准结果；下次定价必须基于新参数重新校准，
  绝不沿用旧 σ。
- 典型情形：现货从 100 改为 110 后，若旧报价在新现货下已低于内在价值，
  重新校准会以 `CALIBRATION_NO_SOLUTION` 明确拒绝，而不是给出旧行情下的数字。

### 并发安全与资源有界

- **Single-flight（单飞）**：同一版本上并发的校准请求只有一个真正执行，
  其余请求阻塞等待并复用同一个不可变的 `CalibrationResult` 对象
  （对象身份相同，而不仅是数值相同）。
- **会话隔离**：不同会话各自持有独立的锁、版本与结果，互无污染。
- **批量复用**：一次批量定价中若引用同一会话，校准只发生一次，所有条目复用。
- **资源有界**：所有会话的校准与批量计算共享一个固定大小的线程池
  （默认 8 worker）；会话数量有上限（默认 256），达到上限后创建新会话会被
  拒绝，避免无界增长。

## 4. 数值失败的可解释处理

底层绝不向调用方透出数值库异常（本服务定价核本身不使用第三方数值库），
也不会静默返回 `NaN`/缺失值。所有失败都映射为带稳定、可区分 `error_code`
的错误响应（HTTP 422，会话不存在为 404，未预期内部错误为 500）：

| error_code | 触发条件 |
| --- | --- |
| `INVALID_PARAMETER` | 行情/波动率/期限等越界或非有限值；缺少必要波动率；会话数达上限 |
| `INVALID_CONTRACT_SPEC` | 合约结构非法（障碍合约缺障碍字段、欧式合约误带障碍字段等） |
| `CONTRACT_EXPIRED` | 估值日 ≥ 到期日 |
| `MODEL_NOT_APPLICABLE` | 模型与合约类型不匹配，或估值日障碍已被击穿 |
| `NO_QUOTES` | 创建/更新会话时报价为空 |
| `CALIBRATION_NO_SOLUTION` | 报价不存在隐含波动率（套利区间外）或求根区间无符号变化 |
| `CALIBRATION_DID_NOT_CONVERGE` | 优化不收敛，或报价隐含波动率离散度过大、无单一平坦解 |
| `NUMERICAL_INSTABILITY` | 求根/求值过程出现非有限值，或障碍有限差分 Greeks 不稳定 |
| `SESSION_NOT_FOUND` | 引用了不存在的会话（HTTP 404） |
| `EMPTY_BATCH` | 批量请求不含任何条目 |
| `INTERNAL_ERROR` | 任何未预期异常的兜底（不回传底层异常类型与堆栈细节） |

## 5. 验证方法

使用 [uv](https://docs.astral.sh/uv/) 管理环境：

```bash
uv sync
uv run pytest -q
```

测试套件（约 70+ 用例）覆盖：

- **数值核**：正态函数、Brent 求根的无解/不收敛/非有限三类失败；
  Black–Scholes 价格（与教材数值一致）、put-call 平价、Greeks 的中心有限
  差分校验、隐含波动率往返与无解分类。
- **障碍公式**：全部 16 个 call/put × in/out × 障碍相对行权价位置分支，
  以及敲出/敲入两类回扣，用**布朗桥修正的蒙特卡洛**独立交叉验证
  （`tests/test_barrier_mc.py`，numpy 仅为开发期依赖）；同时有
  in-out 平价、退化零收益、远障碍退化为普通期权等不依赖随机数的解析校验。
- **校准**：已知 σ 的还原、改现货/利率/报价后的版本失效与重校、
  单飞复用（24 并发得到同一结果对象）、多会话并发隔离、会话容量上限、
  无解/不收敛/空报价的错误归类。
- **服务与接口**：单笔与批量逐字段相等、10 组 × 8 线程并发批量与串行结果
  完全一致、批量对到期/错误模型/越界波动率的正确归类、会话全生命周期、
  未知会话 404、空批量拒绝等。
- 测试日志（caplog）会打印关键输入参数与判定依据（校准 σ 与残差、
  平价各分量、Greeks 有限差分输入等），便于复核。

启动服务（可选）：

```bash
uv run uvicorn main:app --reload
# 然后: curl http://127.0.0.1:8000/health
```

## 6. 目录结构

```
app/
  enums.py          # 合约/障碍/模型枚举
  errors.py         # 可区分的错误码与 PricingError
  models.py         # 领域值对象 + API 请求/响应模型
  validation.py     # 参数、合约、模型适用性与到期校验
  numerics.py       # 正态 CDF/PDF、Brent 求根（失败可区分）
  pricing/
    black_scholes.py# 欧式价格/Greeks/隐含波动率
    barrier.py      # Reiner-Rubinstein 障碍价格（16 分支 + 回扣）
  calibration.py    # 校准会话：失效、单飞、有界管理
  service.py        # 定价编排：校验→分派→价格/Greeks；单笔=批量
  api.py            # FastAPI 路由与异常翻译
tests/              # 单元 + 集成 + 并发 + 蒙特卡洛交叉验证
main.py             # ASGI 入口
```

# AstraTTS DirectML GPU 加速改造计划（KV 常驻显存）

> 目标：Windows + DirectML 后端下，让主要模型计算留在 GPU，KV/中间大张量常驻显存，
> 只把少量控制张量（token、停止标志）与最终音频回传 CPU。
>
> 基于 commit `7b55843`（v1.2.2）实测验证后制定。以下机制均已在
> AMD RX 5500 XT + Windows + ORT 1.23.0 DirectML 上跑通（含真实模型）。

## 一、已验证的事实（本机实测，非推测）

| 验证项 | 结果 |
|---|---|
| 4 个真实模型（encoder/first_stage/stage_decoder/vits）DML 加载 | 全部成功，约 3–5 s |
| KV 输出绑定到显存 | `present_k/v` 输出 `memInfo='DML'`，确认常驻显存 |
| 解码循环（设备张量回喂） | 10–62 步循环数值精确，无 CPU 中转 |
| 动态增长形状（每步 +1 行） | 支持，`[62,4]`、`[81,4]` 等递增形状正确 |
| 混合绑定：KV→显存，y/y_emb/stop→CPU | 支持（`BindOutputToDevice` 逐输出指定） |
| 跨会话设备张量交接（first_stage→stage） | 支持，但**必须显式同步**；不同步实测第 33 轮出现脏读 |
| CPU 标量读回（stop bool） | 支持，与设备张量同轮共存 |
| 设备常驻 vs CPU 往返（小图） | 0.04 ms vs 0.26 ms/次（微型图，仅说明机制开销方向） |

关键约束（官方 + 实测）：
1. **同一 DML Session 不允许并发 `Run`**；不同 Session 可并发。
2. 跨会话消费设备张量前需 `SynchronizeBoundOutputs()`（写方）或
   `SynchronizeBoundInputs()`（读方），否则可能读到旧值。
3. 同一会话内设备张量自回喂（KV 循环）天然有序，无需每步同步。
4. C# ORT 1.23 完整暴露所需 API：`CreateIoBinding/RunWithBinding/GetOutputValues/
   BindOutputToDevice/OrtMemoryInfo("DML", DeviceAllocator, 0, Default)`。

## 二、改造范围

### 阶段 1：基础设施（必做）

1. **配置项**（`AstraTTS.Core/Config/TTSConfig.cs`）
   - 已有 `UseDirectML`（默认 false，保持不变）。
   - 新增 `UseDeviceKvCache`（默认 true）：仅在 `UseDirectML=true` 时生效，
     启用设备常驻 KV 路径；失败自动回退经典路径。
   - 新增 `DirectMLDeviceId`（默认 0）：多显卡选择。
2. **会话串行化**（`AstraTTS.Core/Core/TtsEnginePool.cs`）
   - `UseDirectML && ReuseMemory && PoolCapacity>1` 时禁止共享 Session
     （DML 同会话不可并发），改为独立加载并输出提示日志。
3. **BERT 并发修复**（`AstraTTS.Core/Frontend/BERT/RobertaFeatureExtractor.cs`）
   - 新增 `SerializeInference` 开关；DML 下 V1 多句流式预处理会并发调用同一
     BERT Session，必须加锁串行化。
4. **设备选择**：`GetSessionOptions` 使用 `config.DirectMLDeviceId`。
5. **WebUI / 配置模板**：暴露 `UseDirectML`、`UseDeviceKvCache` 开关。

### 阶段 2：KV 常驻显存解码（核心）

新增 `AstraTTS.Core/Core/DmlKvDecodeLoop.cs`：
- `TryFirstStage(x, prompts, ...)`：first_stage_decoder 输出 KV 绑到 DML 显存，
  y/y_emb/stop 绑到 CPU；跨会话前 `SynchronizeBoundOutputs()`。
- `TryStep(...)`：stage_decoder 每步输入（iy/iy_emb/48 个 KV）中，
  KV 为设备张量，iy/iy_emb 为小 CPU 张量；新 KV 继续留在显存。
- 生命周期：每次 Run 的输出集合持有设备张量，旋转时先取新值再释放旧值。
- 失败即异常，由调用方捕获并回退经典路径（本进程内禁用设备路径）。

接入点（`InferenceEngineV1.cs`）：
- `RunT2S`（非流式）：设备分支 + 原有重试/阈值/EOS 后处理语义保持一致。
- `RunT2SStreamingTokens`（Web 流式）：设备分支 + 原有分块回调语义保持一致。
- 其余模型（encoder/vocoder/BERT）继续用普通 `Run`（音视频边界本就要回 CPU）。

### 不在本次范围

- V2 引擎的设备 KV（实验性引擎，后续按同一模式扩展）。
- 自定义采样子图 / `RandomNormalLike` 图改写（需重新导出模型，风险高）。
- 固定长度 KV 预分配（当前动态增长已可用，收益待实测）。

## 三、验收

1. `dotnet build AstraTTS.sln` 通过（Windows）。
2. `DmlKvDecodeLoop` 用真实模型做冒烟：能出 token，且 KV `memInfo='DML'`。
3. 对比测试（同一文本、同一音色）：
   - CPU 基线 vs `UseDirectML:true, UseDeviceKvCache:true`
   - 记录：热身后总耗时、RTF、首包延迟、显存占用、音频是否正常。
4. 回退演练：人为让设备路径抛错，确认自动回退经典路径且输出正常。

## 四、风险与对策

| 风险 | 对策 |
|---|---|
| 不同 GPU/驱动行为差异 | 设备路径全程 try/catch，异常回退并写日志 |
| 跨会话脏读 | 边界处显式同步；真实模型已验证 |
| 双显卡选错设备 | `DirectMLDeviceId` 可配 |
| 会话共享并发 | 池复用模式下禁共享 + BERT 串行化 |
| 首次编译/加载慢 | 仅影响首次；后续复用 |
| **某模型 DML 原生崩溃** | 已实测 `prompt_encoder` 初始化即访问违例（无法捕获），固定 CPU |

## 五、实测数据（AMD RX 5500 XT，Windows，ORT 1.23.0）

### 5.1 端到端性能

测试文本约 3 秒音频。**测量时 GPU 正被 Minecraft（javaw.exe）占用，
3D 引擎峰值 54%**（`Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage'` 实测）。

| 路径 | ms/token | 说明 |
|---|---:|---|
| A. 纯 CPU | 66.6 | 不占用 GPU，几乎不受游戏影响 |
| B. DirectML + 设备 KV | 95.6 | 与游戏争抢 GPU |
| C. DirectML + 经典 KV | 135.6 | 与游戏争抢 GPU |

**两个可确认的结论：**

1. **设备 KV 优化有效**：同为 DML 后端、完全相同的模型，B 比 C 每 token 快约
   **30%**，验证了"消除每 token 48 个 KV 张量主机往返"的设计目标。
2. **这组 DML vs CPU 的对比不作数**：GPU 被游戏占用时，CPU 路径完全不受影响，
   而 DML 路径要与之争抢算力和显存带宽。必须在空闲 GPU 上复测。

### 5.2 功能正确性

- CPU 与 DML+设备KV 两条路径都能产出音频，RMS 接近（0.172 vs 0.171），
  无静音、无崩溃。
- 流式路径（Web 默认）同样跑通，且日志确认走设备 KV：
  `[DirectML] 设备端 KV 解码已启用`、`设备 KV 已初始化：24 层`。
  CPU 流式 4 块 / 3.60s / RMS 0.148；DML 设备KV 流式 4 块 / 3.52s / RMS 0.159。
- 回退路径验证：`UseDeviceKvCache: false` 时 DML 经典路径正常出音频，
  设备路径异常回退不会破坏可用性。

> 注：复测期间 GPU 占用进一步升至 82%（Minecraft 持续运行），
> 因此**所有 DML 绝对性能数字均不可作为结论**，仅用于验证功能正确性
> 与设备 KV 相对经典 KV 的改进幅度。

### 5.3 复测方法（建议在空闲 GPU 上重做）

1. 关闭游戏等占用 GPU 的程序，确认 3D 引擎占用接近 0：
   ```powershell
   Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage'
   ```
2. `config.yaml` 开启：
   ```yaml
   UseDirectML: true
   UseDeviceKvCache: true
   DirectMLDeviceId: 0
   ```
3. 关注 Web 响应头 `X-Synthesis-Time`、`X-RTF`，或 CLI 输出。
4. 在 `UseDeviceKvCache: true/false` 各测一轮，量化设备 KV 收益。

## 六、性能优化实施记录（第二轮）

依据分阶段剖析（`perfprof`）对代码做了以下优化，并按数据决定后续方向。

### 6.1 分阶段剖析结果（CPU，可信）

| 阶段 | 耗时 |
|---|---:|
| T2S Encoder | 1.6 ms（可忽略） |
| First Stage Decoder | ~70 ms（每句一次） |
| 解码循环 · Clone 版（非流式原实现） | 20.6 ms/token |
| 解码循环 · 无 Clone 版（流式实现） | 14.4 ms/token |
| 声码器 | 80 token → 1265 ms；120 token → 1882 ms |
| BERT | 每句 ~280–390 ms（首句固定成本） |

**关键发现**：解码循环去掉每 token 的 50 次张量 Clone 可省约 **30%**；
声码器随 token 线性增长且流式会对累计 token 重算（总计算量约为最终长度的 2–3 倍）；
BERT 是每句的固定大成本；Encoder 可忽略。

### 6.2 已实施优化

| 编号 | 内容 | 证据/位置 |
|---|---|---|
| O2 | 预生成 24 层 K/V 名称数组，消除每 token 字符串插值 | `InferenceEngineV1.cs`（`PresentKNames` 等） |
| O3 | 非流式解码循环改为零 Clone + 张量缓冲直达读取，生命周期显式持有 | `InferenceEngineV1.cs` `RunT2S` |
| O4 | 设备 KV 循环"仅首步同步"（同 Session 内队列保序；跨 Session 交接仍同步） | `DmlKvDecodeLoop.RunStep` |
| O5 | 设备 KV 循环输出名→索引映射缓存（first stage 与 stage decoder 两套独立映射） | `DmlKvDecodeLoop.BuildFirstStageIndexes/BuildStageIndexes` |
| O6 | iy 增长缓冲复用（避免每步整段 `ToArray()` 的 O(L²) 拷贝） | `DmlKvDecodeLoop.RunStep`（`_yBuffer`） |
| O7 | 声码器重触发因子可配置（`VocoderRetriggerFactor`，默认 1.5） | `TTSConfig.cs`、`TtsEngineV1.cs`、`ITtsEngine.cs` |

### 6.3 优化后实测（GPU 占用 16.7%，AMD RX 5500 XT）

| 路径 | ms/token | RTF |
|---|---:|---:|
| A. 纯 CPU | **40.4** | 1.01 |
| B. DirectML + 设备 KV | 92.7 | 2.32 |
| C. DirectML + 经典 KV | 104.8 | 2.62 |

优化前后对比：CPU 66.6 → **40.4 ms/token**（`Clone` 消除 + 名称数组）；
DML 设备 KV 95.6 → 92.7；DML 经典 KV 135.6 → 104.8。

### 6.4 结论（重要）

在 **AMD RX 5500 XT** 上、GPU 基本空闲时，**纯 CPU 路径仍显著快于 DirectML**
（40.4 vs 92.7 ms/token）。原因可解释：

1. 模型已做 INT8 动态量化（MatMul/MatMulInteger），CPU 上执行高效；
2. 解码器含大量动态 shape 派生（72/98 个 Slice 依赖 Shape/CumSum），
   DirectML 每步需在 CPU 侧解析形状并做小规模同步；
3. K/V 张量本身很小（`[L,1,512]`，fp32 每层约 2KB×L），即使消除主机往返，
   GPU 收益也有限，而每次 Run 的固定调度开销（50 个输入的绑定）不可忽略。

### 6.5 长文本策略实测（CPU，19 秒音频）

| 路径 | 总耗时 |
|---|---:|
| 非流式 | **25.2 s** |
| 流式（factor=1.5） | 32.4 s |

流式比非流式慢约 **28%**，与"声码器对累计 token 反复全量重算（≈2–3×L）"
的推导一致。流式的价值在于首包延迟，而非总耗时。

### 6.6 后续建议（按性价比排序）

1. **默认使用 CPU 后端**：`UseDirectML: false` 是当前硬件下的更优默认（无需改动）。
2. **长文本走非流式**：长文本用非流式可省约 28%；短句保留流式以维持低首包延迟。
   可按文本预估时长或 token 数自动选择。
3. **BERT 缓存**：同音色重复请求可缓存按句 BERT 特征（键：文本+音色），
   省掉每句 ~300 ms。
4. **INT8 之于 vits**：vits 是唯一未量化的大模型（249MB），量化后
   CPU 声码器耗时可望明显下降；需做音频质量对比。
5. **模型级（长期）**：固定形状 KV + attention mask + idx 输入的重导出
   （对标 V2 `gpt_step` 的设计），是让 DirectML 发挥的前提；
   需要导出侧工作，属于独立项目。
6. **自动后端选择**：若未来支持更多硬件，可在启动时做微基准自动选 CPU/DML。

## 七、"纯 GPU 推理"可行性评估（实测节点归属）

用 ORT profiling 统计了每个模型实际执行的节点归属与耗时分布
（`dmlepprobe` 工具输出）：

| 模型 | GPU 节点占比 | CPU 节点占比 | **GPU 耗时占比** | **CPU 耗时占比** | CPU 绝对耗时 |
|---|---:|---:|---:|---:|---:|
| t2s_encoder | 73% | 27% | 99.8% | 0.2% | 1.2 ms |
| first_stage_decoder | 98.7% | 1.3% | 99.9% | 0.1% | 1.5 ms |
| stage_decoder（单步） | 99.5% | 0.5% | 100.0% | 0.0% | 0.1 ms |
| vits（声码器） | 59% | 41% | 98.6% | 1.4% | 13.2 ms |
| roberta | 83.7% | 16.3% | 80.5% | **19.5%** | **292.7 ms** |

### 7.1 关键结论：节点数占比严重误导

按**节点数**看，vits 有 41% 的节点在 CPU、roberta 有 16%，
但按**耗时**看，除 roberta 外 CPU 占比都不到 1.5%。
那些 CPU 节点是 ORT 的主动优化——把形状计算（`Shape`/`ConstantOfShape`/`CumSum`）
和小张量搬运放在 CPU，避免 GPU↔CPU 往返的调度开销更大（源码依据：
`fallback_cpu_capability.cc` 的 `GetCpuPreferredNodes`，日志原文
"Force fallback to CPU execution ... because the CPU execution path is deemed faster"）。

**所以主链路（编码器/两个解码器/声码器）实际上已经是 98.6%–100% 按耗时在 GPU 上执行。**
每步仅 0.1 ms 的 CPU 开销（四个形状/随机节点）。

### 7.2 真正的例外：roberta 的 293 ms

roberta 的 CPU 耗时几乎全部来自**一个 `Loop` 节点（278.6 ms）**，
其余所有 CPU 算子加起来仅 ~15 ms。这个 `Loop` 是子图控制流，
ORT 对其使用 CPU 控制流内核。BERT 每条中文分句都要跑一次，
**这是当前唯一值得且能够优化的"非 GPU 计算"**。

### 7.3 能否做到"完全纯 GPU"

| 目标 | 可行性 | 说明 |
|---|---|---|
| 主链路（编码器+解码器+声码器）按耗时 ~100% GPU | **已实现** | 实测 98.6%–100% |
| 零 CPU 节点（连形状计算也在 GPU） | 不可行且无意义 | ORT 主动选择 CPU 更快的路径；`disable_cpu_ep_fallback=1` 会导致会话创建直接失败（实测全部模型） |
| `prompt_encoder` 上 GPU | **不可行** | 会话创建即原生崩溃（访问违例，无法捕获） |
| `RandomNormalLike`（采样）上 GPU | 需改图 | DML 无该算子注册；可在图外用 CPU 生成噪声作为输入 |
| roberta 的 `Loop` 上 GPU | 需重写子图 | 控制流目前在 CPU；可用等价的无 Loop 结构重导出 |

**因此："纯 GPU 推理"在当前模型结构下不可达，但也没有必要**——
主链路的计算早已在 GPU 上，剩下的 CPU 时间不影响整体吞吐。

### 7.4 若要坚持推进（按收益排序）

1. **roberta 去 Loop**（收益最大，~290 ms/句）：把 BERT 的对齐循环
   （`SplitToSequence` + `Loop`）改写为等价的向量化算子（如 `Gather` + `Reshape`），
   或在导出侧去掉循环。需要模型转换工作，且须做输出对齐逐位校验。
2. **BERT 结果缓存**（无需改模型）：同音色重复文本可直接复用特征，
   省掉整个 roberta 前向（含那 293 ms）。
3. **采样噪声外置**：把 `RandomNormalLike` 替换为图输入，在 CPU 侧生成噪声后
   传入；可消除解码器每步唯一无法上 GPU 的算子（但每步仅 ~0.1 ms 收益）。
4. **声码器动态 shape 固化**：vits 的 41% CPU 节点主要是形状推导；
   固定输入长度后这些可被常量折叠，进一步降低每步调度开销。

> 注：这些改动的性能收益都在 **CPU 路径上更明显**（因为当前 CPU 后端更快），
> 除非未来 DirectML 在动态 shape 上有实质改进，否则优先级应低于
> 6.6 节的 CPU 优化项。

**设备 KV 优化本身是有效的**（同后端下比经典 KV 快约 11–30%），
但在当前模型与低端 GPU 组合下不足以反超 CPU。

## 八、混合模式：仅声码器使用 DirectML（CPU+GPU 协同）

### 8.1 动机

第七章的节点归属剖析显示：解码器（stage_decoder 每步仅 0.1 ms CPU）
的 GPU 化无收益，但**声码器 (vits) 的 GPU 耗时占比 98.6%、CPU 仅 1.4%**，
是整条链路中 GPU 最擅长的部分（大规模并行卷积，无动态形状循环依赖）。
分阶段剖析（`perfprof`）也证实：声码器 DML 耗时约为 CPU 的 **1/2.8**
（80 token：CPU 1233 ms vs DML 429–440 ms），而解码器 DML 反而慢 4–8 倍。

因此新增 `UseVocoderDirectML` 开关：**只把声码器放 GPU，其余模型全部保持 CPU**。

### 8.2 实现

| 文件 | 改动 |
|---|---|
| `TTSConfig.cs` | 新增 `UseVocoderDirectML`（默认 false；`UseDirectML=true` 时忽略） |
| `InferenceEngineV1.cs` | `GetSessionOptions` 增加 `forceDml` 参数；新增 `LoadVocoderWithFallback`（DML 加载失败自动回退 CPU） |
| `TtsEnginePool.cs` | 混合模式（V1）下同样禁止池内共享 Session（DML 会话禁止跨实例并发 Run） |
| `index.html` / `config.js` | WebUI 控制中心新增勾选项 |
| `config.template.yaml` | 模板新增键与注释 |

### 8.3 实测（AMD RX 5500 XT，GPU 空闲，同一二进制、隔离端口背靠背）

测试文本约 6.3–6.8 秒音频；每个配置预热 1 次后取 4–5 次测量：

| 路径 | 平均 RTF | 说明 |
|---|---:|---|
| CPU（基线） | **0.92 – 0.95** | 含第二轮 CPU 优化后的最新代码 |
| 混合模式 | **0.67 – 0.67** | 仅声码器 DML，其余 CPU |

**非流式整体提速约 28–29%**；流式（Web 默认）总耗时从 10.4 s 降到 4.9–5.0 s（约 2×），
首包延迟基本持平（1.56 s vs 1.60 s）。音频质量：RMS 0.1705 vs CPU 0.1700，
峰值一致，听感无差异。

**GPU 实证**：混合模式服务进程持有约 239 MB 显存（`GPU Process Memory` 计数器），
确认声码器实际在 GPU 上执行；日志输出
`[CPU+GPU 混合] 声码器已使用 DirectML（其余模型保持在 CPU）`。

### 8.4 结论

- 混合模式是当前硬件下的**推荐配置**：避开了解码器的 DML 劣势（动态 shape + 小算子调度），
  利用了声码器的 GPU 优势（大并行卷积）。
- 与全 `UseDirectML` 模式相比，混合模式不触发解码器动态 shape 的 CPU↔GPU 往返，
  因此不会出现第七章"全 DML 反而更慢"的现象。
- 若未来在更强 GPU 上解码器 DML 有实质改进，仍可直接切换到全 `UseDirectML`。

## 九、第三轮优化：BERT 缓存 + roberta 去 Loop（已实施）

按第七节结论推进的两项优化，第三项（声码器固定形状）与第四项（噪声外置）
经评估后跳过，证据见 9.3/9.4。

### 9.1 BERT 特征缓存（已实施）

**实现**：`BertFeatureCache.cs`（FIFO，容量 256，线程安全）+ 接入
`RobertaFeatureExtractor.Extract`（新增 `UseCache` 开关，构造参数控制）。
键 = 文本 + word2ph（防止同文本异对齐误命中）；输出数组只读共享
（两处调用方均 `Buffer.BlockCopy` 后即弃，已核实）。
计算在锁外进行，不阻塞流式多句并行预取。

**实测**（同一引擎实例连续合成，CPU，短文本 "今天天气不错，我们出去走走吧。"）：

| 配置 | 第 1/2/3 次 |
|---|---|
| 无缓存 | 2781 / 3156 / 3089 ms |
| 有缓存 | 2265 / 2598 / 2532 ms |

**约 18% 提升**；DebugMode 日志确认命中：
`[BERT Cache] 命中: '今天天气不错' (rows=12), 累计 命中=3 未命中=4`。
参考音色文本（"良宵方始"等）在首次加载时即被缓存，后续切换音色也能受益。

**配置**：`UseBertCache`（默认 true）；WebUI 已暴露开关。

### 9.2 roberta 去 Loop（已实施，并存可切换）

**问题**：roberta 图中 `/Loop` 节点是 `torch.repeat_interleave(hidden, repeats)` 的
逐 token 展开实现，DirectML 无 Loop 内核 → 强制 CPU 执行 + 设备↔主机搬运。
profile 显示 DML 后端下该节点单次占 **278.6 ms**（CPU 后端则无此问题，因为本来就在 CPU）。

**实现**：图手术脚本 `tools/converter/roberta_noloop.py` 把 Loop 替换为等价向量化子图：

```
cum  = CumSum(repeats_norm, axis=0)        # [M] 前缀和
total= ReduceSum(repeats_norm, axis=0)     # [1] Σr
ridx = CumSum(ConstantOfShape(total, 1)) - 1
mask = LessOrEqual(cum[1,M], ridx[total,1])# [total, M] 广播
idx  = ReduceSum(Cast(mask, int64), axes=[1])
phone_features = Gather(hidden_content, idx)
```

所需算子（CumSum/LessOrEqual/ReduceSum/Gather/Cast/ConstantOfShape）
**均有 DirectML 内核**（Loop 注册数为 0）。

**正确性验证**（`_tooling/robertacmp`，CPU，9 用例）：
常规/含 0 重复/全零/全 1/单元素广播/长短文本 — **全部与原型逐位一致**。
（注：过程序迭代中出现过两个 bug——用 `Shape` 求 total 应为 `ReduceSum`；
`keepdims=1` 导致输出多一维——均被对比工具捕获并修正。）

**性能实测**（同一输入 ids=82, Σr=160，隔离完整前向）：

| 后端 | 原模型 | 去 Loop | 提升 |
|---|---:|---:|---:|
| CPU | 210.1 ms | 209.0 ms | 0.5% |
| **DML** | **186.9 ms** | **34.1 ms** | **81.7%** |

**结论**：收益集中在 DirectML 后端（Loop 正是其瓶颈）；CPU 后端几乎无差别
（Loop 在 CPU 上本来就快）。这精确印证第七节的 profile 分析。

**接线**：`UseBertNoLoopModel`（默认 **false**，因需先验证）；
`EffectiveBertModelPath` 在开启且文件存在时用 `roberta_noloop.onnx`，否则自动回退原模型。
两个模型并存（均 599MB，磁盘共 ~1.2GB，E: 剩余 451GB 充足）。

**启用建议**：使用 DirectML 后端时开启；纯 CPU 后端可不开（无收益）。
`config.yaml` 设 `UseBertNoLoopModel: true` 或 WebUI 勾选。

### 9.3 声码器固定形状（评估后跳过）

**否定证据**（结构分析，详见第七节）：vits 的 `encoder_ssl`/`encoder2` 是
非因果全连接注意力（13 个 `Softmax(axis=-1)` 节点覆盖 2×pred 时间轴），
且图中 mask 链为死代码（`Equal(填充1, 0)` 恒 false，Where 永远取 logits 分支）。
在 `pred_semantic` 末尾 padding 会通过注意力分母**改变所有前缀 token 的输出**。

**收益上限**：vits 的 CPU 耗时仅 13.2 ms（1.4%），即使全部消除也微不足道。

**若要实现**：需真 mask 手术（改写 mask 链为有效屏蔽）+ 新增 valid_len 输入 +
重验证卷积边界与相对位置编码，属高风险改动。

### 9.4 采样噪声外置（评估后跳过）

`RandomNormalLike` 在 DML 无注册。但实测每步 CPU 耗时仅 **43 µs**
（stage_decoder）/ **322 µs**（vits，pred=40），合计占比 <0.02%。
需修改两个模型 + C# 每步喂噪声，收益与风险不成比例。

### 9.5 当前推荐的配置组合

| 场景 | 推荐配置 |
|---|---|
| Web 服务（长驻进程） | `UseBertCache: true` + 视后端选 `UseBertNoLoopModel` |
| DirectML 后端 | `UseBertNoLoopModel: true`（81.7% 提升） |
| 纯 CPU 后端 | `UseBertCache: true`，可不启用去 Loop 模型 |
| 长文本 | 优先非流式（见 6.5：省约 28%） |
| 声码器 GPU（本机最优） | `UseVocoderGgml: true`（见 GGML_VULKAN_MIGRATION.md；ORT 1.30 下替代 DML 混合模式） |

> 注：以上数字均为 AMD RX 5500 XT + GPU 低占用条件下的实测；
> 其他硬件请用 `_tooling` 下的工具复测。

## 十、参考（固定到验证时点）

- ORT DirectML EP：同会话并发限制、memory pattern、sequential 模式。
- ORT C#：`OrtIoBinding`、`RunWithBinding`、`OrtMemoryInfo("DML", ...)`。
- 项目内：`config.template.yaml`（根级 PascalCase 键）、`InferenceEngineV1.cs`
  （两处解码循环）、`DmlKvDecodeLoop.cs`（设备 KV 实现）。
- ggml (Vulkan/Metal) 声码器迁移：见 `docs/GGML_VULKAN_MIGRATION.md`（第五轮，ORT 1.30）。

# 情感控制（无参考音频的情感引导）

AstraTTS 支持在**不提供情感参考音频**的情况下，引导合成语音的情感表达。
方法是从公开情感语音数据集提取"情感方向向量" τ，在推理时注入说话人嵌入空间。

> Status: 实验性功能，默认关闭（`UseEmotionControl: false`）。仅在 V1 引擎（默认）生效。

## 原理

链路（与实际推理完全一致）：

```
wav(16k) ──speaker_encoder.onnx──▶ sv_emb [20480]
                                      │
                    ┌─────────────────┴──────────────────┐
                    │  x_new = x_base + α·||x_base||·τ̂   │  ← 情感注入
                    └─────────────────┬──────────────────┘
wav(32k) ────────────────────────────▶ prompt_encoder.onnx ──▶ ge [1024], ge_advanced [512]
                                                                     │
                                                        vits.onnx (风格/音色分支)
```

- `speaker_encoder.onnx` 输出 **20480 维**说话人嵌入（`sv_emb`），是 GPT-SoVITS v4 中
  送入 `prompt_encoder` 的真实说话人表示（不是 256 维，也不是 CNHuBERT 的 768 维帧级特征）。
- `t2s_encoder.onnx`（GPT 分支）**不含说话人输入**（只接收 `ssl_content [1,768,T]` 等），
  因此情感注入只需经 `sv_emb → prompt_encoder` 一条路径；不入 GPT。
- 注入数学：`x_new = x_base + α · ||x_base|| · τ̂`，其中 `τ̂` 为单位化的情感方向向量。

### 为什么注入点选 sv_emb

`prompt_encoder` 是唯一同时消费"说话人身份"与"参考音频韵律"的模块，其输出 `ge` /
`ge_advanced` 是 `vits.onnx` 的音色-风格条件。改写 `sv_emb` 即改写风格条件，
与音频内容（`ref_audio`）解耦，因此可以"用中性参考音频 + 情感方向"来引导情感。

## τ 的提取（配对差公式）

多说话人数据集必须用**同一说话人内配对差**，否则 τ 会混入说话人身份差异：

```
τ_e = mean_{s,i} ( x_e^{s,i} − x_n^{s} )
```

即先算同一说话人 s 的情感均值与中性均值之差，再跨说话人平均。

数据来源：**CSEMOTIONS**（`AIDC-AI/CSEMOTIONS`，现别名 `ATH-MaaS/CSEMOTIONS`）
- 10 位专业配音演员（5 男 5 女），7 种情感，48kHz 录音棚质量
- 本仓库提取结果：**1380 条样本 / 10 位说话人**（每说话人每情感 20 条，全覆盖）

提取命令（需要 GPU；DirectML 可用则自动使用，不占 CPU）：

```bash
# 1) 下载数据集分片（8 个 parquet，约 3GB）
#    https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS/tree/main/data

# 2) 提取 τ
python tools/emotion/extract_tau.py \
    --parquet-dir <分片目录> \
    --resources <AstraTTS/resources> \
    --out <AstraTTS/resources/emotion_vectors> \
    --per-speaker-emotion 20 --device dml --threads 2
```

产物：

| 文件 | 说明 |
|---|---|
| `tau_<emotion>.f32` | 单位化后的 τ（little-endian float32，20480 维），C# 直接读取 |
| `manifest.json` | 维度、一致性与样本统计 |
| `diagnostics.json` | 每种情感的逐说话人明细 |
| `tau_vectors.npz` | 原始（未归一化）τ，供分析 |

### 质量指标（本仓库提取结果）

| 情感 | \|τ\| | 跨说话人一致性 | 备注 |
|---|---|---|---|
| happy | 23.4 | 0.47 | 高唤醒簇 |
| angry | 32.7 | 0.52 | 高唤醒簇 |
| sad | 24.1 | 0.52 | 低唤醒簇 |
| surprise | 29.0 | 0.52 | 高唤醒簇 |
| playfulness | 22.1 | 0.48 | 高唤醒簇 |
| fearful | 21.7 | 0.50 | 低唤醒簇 |

一致性 = `||mean(Δ)|| / mean(||Δ||)`，0.5 左右属正常（情感表达在说话人间差异较大）。

**τ 之间的余弦相似度**呈两簇结构：

```
              happy  angry   sad   surprise playful fearful
happy        +1.000 +0.736 +0.159  +0.881  +0.820  +0.351
angry        +0.736 +1.000 +0.169  +0.821  +0.762  +0.344
sad          +0.159 +0.169 +1.000  +0.070  +0.133  +0.874
surprise     +0.881 +0.821 +0.070  +1.000  +0.831  +0.264
playfulness  +0.820 +0.762 +0.133  +0.831  +1.000  +0.246
fearful      +0.351 +0.344 +0.874  +0.264  +0.246  +1.000
```

高唤醒（happy/angry/surprise/playfulness）之间 0.71–0.86，低唤醒（sad/fearful）0.82，
跨簇仅 0.02–0.37。这是情感嵌入空间的常见现象——共享"唤醒度/激活"分量。
若需要更"单纯"的方向（如只保留效价），可对 τ 做两簇内的正交化后再用。

## 使用

### 配置（`config.yaml`）

```yaml
UseEmotionControl: true
EmotionVectorPath: emotion_vectors   # 相对 ResourcesDir
EmotionAlpha: 0.3                    # 默认 α
```

### WebUI

合成实验室 →「情感引导」下拉框选择情感、「情感强度 α」滑块（0–1.0）。
选择"中性（不注入）"即恢复原始行为。

### HTTP API

```bash
curl -X POST http://localhost:5000/api/tts/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气真好","emotionId":"happy","emotionAlpha":0.3}' \
  -o happy.wav
```

### CLI

```bash
# 一次性
astra-cli -e happy -a 0.3 "今天天气真好"

# 交互式
/emotion happy 0.3
/emotion neutral      # 关闭
```

## α 选择指南（实测）

单次合成的客观指标（chenxing 音色、同文本、32kHz 输出；F0 为自相关法中位数）：

| 配置 | 音色相似度 (vs 中性) | F0 (Hz) | 时长 |
|---|---|---|---|
| neutral | 1.000 | 227 | 3.48s |
| happy α=0.1 | 0.948 | 230 | 3.44s |
| happy α=0.3 | 0.915 | 251 | 3.76s |
| happy α=0.5 | 0.842 | 269 | 3.52s |
| angry α=0.1 | 0.925 | 229 | 4.16s |
| angry α=0.3 | 0.893 | 266 | 3.52s |
| angry α=0.5 | 0.786 | 271 | 4.68s |

结论与建议：

- **F0 随 α 单调上升**（227 → 271 Hz），证明韵律确实被情感方向引导。
- **音色相似度随 α 平滑下降**：α=0.1 ≈ 0.93–0.95；α=0.3 ≈ 0.89–0.92；α=0.5 ≈ 0.79–0.84。
- 推荐区间 **0.2–0.4**：可感知的情感色彩与足够的音色保持。
  追求强烈表达可用 0.5，但需接受音色偏移。
- 注意：T2S 解码器内含 `RandomNormalLike` 随机采样，**音频层不可逐位复现**；
  同一设置的两次合成相似度约 0.95–0.96（采样噪声基线）。
  因此音色相似度应看"相对该基线的下降量"，而不是绝对值。

## 验证

```bash
# 单元测试（τ 加载/归一化/α=0 位一致/维度不匹配）
dotnet test AstraTTS.Tests --filter EmotionDirection

# 端到端：生成 α 网格音频
dotnet run --project tools/emotion_eval -c Release -- \
  --config config.yaml --out _emotion_eval \
  --emotions neutral,happy,angry,sad,surprise,fearful,playfulness \
  --alphas 0.1,0.2,0.3,0.5 --repeats 2

# 客观指标分析
python tools/emotion/analyze_eval.py --eval-dir _emotion_eval --resources resources
```

验证项目与结论：

1. **α=0 / neutral 不触碰注入**：`EmotionDirectionController.Apply` 在 α=0 时返回逐位相同的拷贝
   （单元测试断言精确相等），`TtsEngineV1` 在 α≤0 时不做任何嵌入改写。
   在 `sv_emb → ge` 层面：α=0 时 `max|Δge| = 0.000e+00`（逐位一致，见 `tools/emotion/verify_injection.py`）。
2. **随机 τ 有效**：随机方向的 τ 同样显著改变 `ge`（量级与真实情感相当），
   证明注入通路确实生效而非无效扰动。
3. **τ 维度校验**：τ 与 sv_emb 维度不一致时打日志并忽略，不影响正常合成。
4. **文件缺失回退**：τ 目录/文件缺失时 `Loaded=false`，情感功能静默禁用，合成照常工作。

## 已知限制

- 情感 τ 仅对 **V1 引擎**生效（V2 的说话人通路是 `sv.onnx` ERes2NetV2 + `sv_emb.npy`，机制不同）。
- τ 从中文情感语音（CSEMOTIONS）提取，**跨语言/跨音色迁移效果未验证**；
  非中文文本或差异很大的音色建议重新提取或调整 α。
- 高唤醒情感之间的方向相关性较高（0.71–0.86），细微区分（如 happy vs surprise）可能不锐利。
- 数据集为 48kHz 但说话人编码器按 16kHz 输入工作（与推理链路一致），
  提取脚本内部做了重采样与幅度归一化。

## 文件清单

```
AstraTTS.Core/Core/EmotionDirectionController.cs   # τ 加载 + 注入计算
AstraTTS.Core/Core/TtsEngineV1.cs                  # 注入接入（PrepareReference）
AstraTTS.Core/Core/ITtsEngine.cs                   # TtsOptions.EmotionId / EmotionAlpha
AstraTTS.Core/Config/TTSConfig.cs                  # UseEmotionControl / EmotionVectorPath / EmotionAlpha
AstraTTS.Web/Controllers/TtsController.cs          # API 透传
AstraTTS.Web/Models/TtsRequest.cs                  # 请求模型
AstraTTS.Web/wwwroot/index.html + js/playground.js # WebUI 情感选择 + α 滑块
AstraTTS.CLI/Program.cs                            # -e/--emotion, -a/--emotion-alpha, /emotion
tools/emotion/extract_tau.py                       # 数据集 → τ 提取（GPU）
tools/emotion/verify_injection.py                  # ge 层验证（α=0 位一致、敏感度）
tools/emotion/analyze_eval.py                      # 音频客观指标（相似度/F0/能量）
tools/emotion_eval/                                # C# 端到端评估工具
AstraTTS.Tests/EmotionDirectionTests.cs            # 单元测试
resources/emotion_vectors/                         # τ 数据（*.f32 需自行生成或从发布物获取）
```

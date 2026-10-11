# ggml (Vulkan/Metal) 声码器迁移报告

> 评估对象：[onnxruntime-ggml](https://github.com/thewh1teagle/onnxruntime-ggml) v0.2.0
> ——ONNX Runtime 的插件式执行提供器（Windows/Linux=Vulkan，macOS=Metal）。
> 本仓库已集成"仅声码器使用 ggml"的混合模式（`UseVocoderGgml`，默认关）。
>
> 测试环境：Windows 11 + AMD RX 5500 XT + Ryzen 5 5500；ORT 1.30.0。
> 结论先行：**声码器迁移成功（内核 5×，流式整链路约 2×）；声学模型不建议迁移（实测变慢）。**

## 一、关键事实

| 项 | 结论 |
|---|---|
| 形态 | ORT 的插件 EP（非 fork）；Windows 有独立 `onnxruntime_ggml.dll`（约 56 MB，不进 git，获取方式见 `third_party/README.md`） |
| 前置要求 | ORT **1.29+** 插件 API；本仓库因此升级至 1.30.0 |
| 认领策略 | **整图认领或不认领**：图中含不支持算子即整图回落 CPU；`partial=1` 实测有 bug（认领被 ORT 回滚），不可用 |
| 官方包与 DML | ORT 1.29+ 官方包不再内置 DirectML（DML 包停更于 1.24.4）——**ggml 与 DirectML 二选一**；`UseVocoderDirectML` 在本构建下会静默回退 |
| 并发 | ggml 会话允许并发 Run（与 DML 的单会话限制不同） |
| 权重 | 编译期上传 GPU 常驻（vits 约 265 MB）；`gpu_min_weight_mb` 默认 256 会把小模型踢回 ggml-CPU 内核，声码器需显式设 0 |

## 二、逐模型实测

| 模型 | 认领 | Vulkan 耗时 vs CPU | 结论 |
|---|---|---|---|
| **vits（声码器）** | **1967/1967** | **5.0×**（C#：1784→356 ms） | ✅ 迁移 |
| t2s_encoder | 差 1 个 `ArgMax` 被拒 | 手术后 44 ms vs 1.8 ms（-24×） | ❌ 不迁 |
| first_stage_decoder | 差 1 个 `ArgMax` 被拒 | 手术后 82 ms vs 56 ms | ❌ 不迁 |
| t2s_stage_decoder | 差 2 个 `ArgMax` 被拒 | 手术后 33 ms vs 13 ms/步 | ❌ 不迁 |

声学模型被拒的唯一原因是 1~2 个 `ArgMax`。已实现等价的 `TopK(k=1)+Squeeze`
手术（token 匹配率与同后端重复运行一致，模型自带 `RandomNormalLike` 随机性），
**但迁后变慢，故未采纳**——逐 token 解码每步几十个小算子的调度开销远超收益，
与此前 DirectML 的结论一致。

## 三、vits 迁移的正确性与性能

- 正确性：跨后端频谱包络相关 **0.998**（高于 ggml 自身两次运行的 0.995）；RMS 一致。
- 内核级：5.0×。整链路（叠加 CPU 端解码优化与 BERT 优化后）：
  非流式 RTF ≈ 0.47–0.55；流式总耗时相比纯 CPU 约 2×（流式反复重算声码器，收益放大）。
- 进程显存持有约 740 MB（含 265 MB 权重常驻）。

## 四、集成方式（本仓库）

- `AstraTTS.Core/Core/GgmlExecutionProvider.cs`：EP 注册与 SessionOptions 封装，
  库路径可用环境变量 `ASTRA_GGML_LIBRARY` 覆盖；失败静默回退 CPU。
- `InferenceEngineV1.LoadVocoderWithFallback`：ggml → DirectML(若有) → CPU 三级回退。
- 配置：`UseVocoderGgml`（默认 false）；WebUI 控制中心可勾选。
- 构建：`third_party/onnxruntime_ggml.dll` 存在时自动复制到输出（缺失不影响构建，仅该功能回退）。

## 五、已知局限

1. 声学模型受"1 个 ArgMax 卡整图"限制，且即使在 GPU 上也无收益；如需全 GPU 需模型导出侧重做（固定形状等）。
2. ggml 默认 f16 权重存储（可用 `weights=f32`）；本机 vits 输出频谱与 CPU 一致。
3. 未跨平台验证（macOS Metal / Linux Vulkan 需对应平台库）。

## 六、复现要点

```bash
# 1. 下载 ggml 库到 third_party/（见 README），或设 ASTRA_GGML_LIBRARY
# 2. 配置
UseVocoderGgml: true
# 3. 访问响应头 X-RTF 评估（<1 快于实时）
```

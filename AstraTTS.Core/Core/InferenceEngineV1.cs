using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;
using System.Runtime.InteropServices;

namespace AstraTTS.Core.Core
{
    public class InferenceEngineV1 : IDisposable
    {
        /// <summary>
        /// 调试模式开关。启用后会输出详细的中间结果用于对比 Python 实现。
        /// </summary>
        public static bool DebugMode { get; set; } = false;

        private static void DebugLog(string message)
        {
            if (DebugMode) Console.WriteLine(message);
        }

        private InferenceSession? _t2sEncoder;
        private InferenceSession? _firstStageDecoder;
        private InferenceSession? _stageDecoder;
        private InferenceSession? _vocoder;

        // V2ProPlus 组件
        private InferenceSession? _promptEncoder;
        private InferenceSession? _hubert;
        private InferenceSession? _svModel; // Speaker Verification

        private SessionOptions GetSessionOptions(bool useDirectML)
        {
            var sessionOptions = new SessionOptions();

            // ============================================================
            // CPU 性能优化配置 (针对 Xeon E5 2696V3: 18核36线程)
            // ============================================================

            // 1. 线程配置
            // IntraOpNumThreads: 单个算子内部的并行度
            //   - 设为物理核心数，避免超线程竞争
            //   - 对于大矩阵运算（MatMul）效果显著
            sessionOptions.IntraOpNumThreads = 6;  // 物理核心数

            // InterOpNumThreads: 算子之间的并行度
            //   - TTS 是链式结构，设为 1 避免调度开销
            //   - 如果模型有多个独立分支可以设更大
            sessionOptions.InterOpNumThreads = 1;

            // 2. 执行模式
            //   - ORT_SEQUENTIAL: 顺序执行算子（推荐用于链式模型）
            //   - ORT_PARALLEL: 并行执行算子（适合有多分支的模型）
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

            // 3. 图优化级别
            //   - ORT_ENABLE_ALL: 启用所有优化（常量折叠、算子融合等）
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            // 4. 内存优化
            //   - EnableCpuMemArena: 使用内存池减少分配开销
            //   - EnableMemoryPattern: 基于执行模式优化内存布局
            sessionOptions.EnableCpuMemArena = true;
            sessionOptions.EnableMemoryPattern = true;

            // 5. 减少 CPU 空转等待
            sessionOptions.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");

            // 6. DirectML 配置 (如果启用)
            if (useDirectML)
            {
                // DirectML 不支持 MemoryPattern
                sessionOptions.EnableMemoryPattern = false;
                sessionOptions.AppendExecutionProvider_DML(0);
            }

            return sessionOptions;
        }

        public void LoadModels(string modelDir, string? hubertPath = null, string? svPath = null, bool useDirectML = false)
        {
            var opt = GetSessionOptions(useDirectML);

            // Parallel loading of all models (FP32 - INT8 ConvInteger 不支持)
            Parallel.Invoke(
                () => _t2sEncoder = new InferenceSession(Path.Combine(modelDir, "t2s_encoder.onnx"), opt),
                () => _firstStageDecoder = new InferenceSession(Path.Combine(modelDir, "t2s_first_stage_decoder.onnx"), opt),
                () => _stageDecoder = new InferenceSession(Path.Combine(modelDir, "t2s_stage_decoder.onnx"), opt),
                () => _vocoder = new InferenceSession(Path.Combine(modelDir, "vits.onnx"), opt),
                () =>
                {
                    string promptEncoderPath = Path.Combine(modelDir, "prompt_encoder.onnx");
                    if (File.Exists(promptEncoderPath))
                    {
                        _promptEncoder = new InferenceSession(promptEncoderPath, opt);
                    }
                },
                () =>
                {
                    if (!string.IsNullOrEmpty(hubertPath) && File.Exists(hubertPath))
                    {
                        _hubert = new InferenceSession(hubertPath, opt);
                    }
                },
                () =>
                {
                    if (!string.IsNullOrEmpty(svPath) && File.Exists(svPath))
                    {
                        _svModel = new InferenceSession(svPath, opt);
                    }
                }
            );
        }

        public float[] GetHubertContent(float[] audio16k)
        {
            if (_hubert == null) throw new InvalidOperationException("Hubert model not loaded.");

            // 检测输入名称：base 版是 "source"，full 版是 "input_values"
            string inputName = _hubert.InputMetadata.Keys.Contains("input_values") ? "input_values" : "source";
            var inputTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            using var results = _hubert.Run(inputs);
            var tensor = results.First().AsTensor<float>(); // [1, T, 768] 或 [1, 768, T]

            int dim1 = tensor.Dimensions[1];
            int dim2 = tensor.Dimensions[2];

            // 重要：T2S 期望 [1, 768, T]
            // 如果输出已经是 [1, 768, T]，则直接 ToArray
            if (dim1 == 768)
            {
                var arr = tensor.ToArray();
                Console.WriteLine($"[Hubert] Output shape: [1, {dim1}, {dim2}] (direct)");
                if (arr.Length > 5)
                    Console.WriteLine($"[Hubert] First 5: {string.Join(", ", arr.Take(5))}");
                return arr;
            }

            // 如果输出是 [1, T, 768]，需要手动转置成 [1, 768, T]
            // 以避免特征向量交织导致噪声
            int T = dim1;
            int C = dim2; // 768
            float[] transposed = new float[T * C];

            for (int t = 0; t < T; t++)
            {
                for (int c = 0; c < C; c++)
                {
                    transposed[c * T + t] = tensor[0, t, c];
                }
            }
            float sum = 0;
            float sumSq = 0;
            foreach (var f in transposed)
            {
                sum += f;
                sumSq += f * f;
            }
            float mean = sum / transposed.Length;
            float std = (float)Math.Sqrt(sumSq / transposed.Length - mean * mean);

            Console.WriteLine($"[Hubert] Output mean={mean}, std={std}");
            Console.WriteLine($"[Hubert] Output shape: [1, {C}, {T}]");
            if (transposed.Length > 5)
                Console.WriteLine($"[Hubert] First 5: {string.Join(", ", transposed.Take(5))}");

            return transposed;
        }

        public float[] GetSpeakerEmbedding(float[] audio16k)
        {
            if (_svModel == null) throw new InvalidOperationException("SV model not loaded.");

            // 根据元数据，输入名称是 "waveform"
            var inputTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("waveform", inputTensor) };

            using var results = _svModel.Run(inputs);
            return results.First().AsTensor<float>().ToArray();
        }

        public (float[] ge, float[] ge_advanced) GetPromptEmbedding(float[] audio32k, float[] svEmb)
        {
            if (_promptEncoder == null) throw new InvalidOperationException("Prompt Encoder not loaded.");

            // 根据元数据，输入名称是 "ref_audio" 和 "sv_emb"
            var audioTensor = new DenseTensor<float>(audio32k, new[] { 1, audio32k.Length });
            var svTensor = new DenseTensor<float>(svEmb, new[] { 1, svEmb.Length });

            Console.WriteLine($"[PromptEncoder] svEmb length: {svEmb.Length}");
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("ref_audio", audioTensor),
                NamedOnnxValue.CreateFromTensor("sv_emb", svTensor)
            };

            using var results = _promptEncoder.Run(inputs);
            // Output [0]: ge, [1]: ge_advanced
            var ge = results[0].AsTensor<float>().ToArray();
            var geAdvanced = results[1].AsTensor<float>().ToArray();
            Console.WriteLine($"[PromptEncoder] ge length: {ge.Length}, geAdvanced length: {geAdvanced.Length}");
            return (ge, geAdvanced);
        }

        public long[] RunT2S(long[] textSeq, float[,] textBert, long[] refSeq, float[,] refBert, float[] sslContent)
        {
            if (_t2sEncoder == null || _firstStageDecoder == null || _stageDecoder == null)
                throw new InvalidOperationException("T2S models not loaded.");

            if (sslContent == null || sslContent.Length == 0)
                throw new ArgumentException("sslContent 不能为控制或长度为 0。这通常是因为参考音频加载失败或特征提取异常，请检查 config.json 中的音频路径。");

            if (sslContent.Length % 768 != 0)
                throw new ArgumentException($"sslContent 长度 {sslContent.Length} 不合法，必须是 768 的倍数。");

            float[] refBertFlat = Flatten(refBert);
            float[] textBertFlat = Flatten(textBert);
            try
            {
                // 1. Encoder 推理
                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("ref_seq", new DenseTensor<long>(refSeq, new[] { 1, refSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("ref_bert", new DenseTensor<float>(refBertFlat.AsMemory(0, refBert.Length), new[] { (int)refBert.GetLength(0), (int)refBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("text_bert", new DenseTensor<float>(textBertFlat.AsMemory(0, textBert.Length), new[] { (int)textBert.GetLength(0), (int)textBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("ssl_content", new DenseTensor<float>(sslContent, new[] { 1, 768, sslContent.Length / 768 }))
                };

                // 输入详细日志 (仅调试模式)
                if (DebugMode)
                {
                    Console.WriteLine($"[T2S Input] ref_seq shape: [1, {refSeq.Length}]");
                    Console.WriteLine($"[T2S Input] text_seq shape: [1, {textSeq.Length}]");
                    Console.WriteLine($"[T2S Input] ref_bert shape: [{refBert.GetLength(0)}, {refBert.GetLength(1)}], first 3: {refBertFlat[0]}, {refBertFlat[1]}, {refBertFlat[2]}");
                    Console.WriteLine($"[T2S Input] text_bert shape: [{textBert.GetLength(0)}, {textBert.GetLength(1)}], first 3: {textBertFlat[0]}, {textBertFlat[1]}, {textBertFlat[2]}");
                    Console.WriteLine($"[T2S Input] ssl_content shape: [1, 768, {sslContent.Length / 768}], mean: {sslContent.Average()}, first 3: {string.Join(", ", sslContent.Take(3))}");
                }

                using var encoderResults = _t2sEncoder.Run(encoderInputs);

                if (encoderResults.Count < 2) throw new Exception("Expected at least 2 outputs from T2S Encoder");

                var val0 = encoderResults[0].Value;
                var val1 = encoderResults[1].Value;

                Tensor<float> x;
                if (val0 is Tensor<float> tf) x = tf.Clone();
                else throw new Exception($"Output 'x' is not Tensor<float>. Actual: {val0?.GetType().FullName}");

                Tensor<long> prompts;
                if (val1 is Tensor<long> pl) prompts = pl.Clone();
                else throw new Exception($"Output 'prompts' is not Tensor<long>. Actual: {val1?.GetType().FullName}");

                if (DebugMode)
                {
                    var xArr = x.ToArray();
                    DebugLog($"[T2S Encoder] x shape: [{x.Dimensions[0]}, {x.Dimensions[1]}, {x.Dimensions[2]}]");
                    DebugLog($"[T2S Encoder] x mean: {xArr.Average()}");
                }

                // 3. 全局重试循环 (包含 First Stage 和 Stage Decoder)
                // 原因：First Stage Decoder 输出也是非确定性的（包含随机采样）， sometimes producing "bad" state that causes Stage Decoder to fail immediately.
                // 因此必须重试整个解码过程。

                // 动态阈值：使用 2x 乘数兼容中英文
                int minExpectedTokens = Math.Max(8, textSeq.Length * 2);
                if (DebugMode) DebugLog($"[T2S] Min expected tokens: {minExpectedTokens} (textSeq.Length={textSeq.Length})");

                List<long> generatedTokens = new List<long>();
                int maxRetries = 10;
                int retryCount = 0;
                bool success = false;

                while (retryCount < maxRetries)
                {
                    generatedTokens.Clear();

                    // --- A. 运行 First Stage Decoder ---
                    // 每次重试都重新运行，以获取新的随机采样状态
                    var firstStageInputs = new List<NamedOnnxValue>
                 {
                     NamedOnnxValue.CreateFromTensor("x", x),
                     NamedOnnxValue.CreateFromTensor("prompts", prompts)
                 };

                    using var firstStageResults = _firstStageDecoder.Run(firstStageInputs);

                    var yTensor = firstStageResults.First(r => r.Name == "y").AsTensor<long>().Clone();
                    var yEmbTensor = firstStageResults.First(r => r.Name == "y_emb").AsTensor<float>().Clone();

                    if (DebugMode)
                    {
                        var yArr = yTensor.ToArray();
                        DebugLog($"[FirstStage] Attempt {retryCount} y mean: {yArr.Average()}");

                        // 检查 First Stage 结果中是否已经包含 EOS
                        if (yArr.Any(t => t >= 1024))
                        {
                            DebugLog($"[T2S Warning] First Stage output already contains EOS. Retry {retryCount} likely to fail.");
                        }
                    }

                    // Capture First Stage Token (The first semantic token)
                    generatedTokens.Add(yTensor.ToArray().Last());

                    // 按顺序构建 KV Cache
                    var kvCache = new List<Tensor<float>>();
                    for (int layer = 0; layer < 24; layer++)
                    {
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>().Clone());
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>().Clone());
                    }

                    // --- B. 运行 Stage Decoder Loop ---
                    var currentYTensor = yTensor.Clone();
                    var currentYEmbTensor = yEmbTensor.Clone();
                    var currentKvCache = kvCache; // 已经在本次循环中新建，直接使用
                    int numLayers = currentKvCache.Count / 2;

                    for (int i = 0; i < 500; i++)
                    {
                        var stageInputs = new List<NamedOnnxValue>();
                        stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                        stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));

                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", currentKvCache[layer * 2]));
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", currentKvCache[layer * 2 + 1]));
                        }

                        using var stageResults = _stageDecoder.Run(stageInputs);
                        currentYTensor = stageResults.First(r => r.Name == "y").AsTensor<long>().Clone();
                        currentYEmbTensor = stageResults.First(r => r.Name == "y_emb").AsTensor<float>().Clone();
                        var stopCondition = stageResults.First(r => r.Name == "stop_condition_tensor").AsTensor<bool>().First();

                        currentKvCache.Clear();
                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            currentKvCache.Add(stageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>().Clone());
                            currentKvCache.Add(stageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>().Clone());
                        }

                        var lastToken = currentYTensor.ToArray().Last();
                        generatedTokens.Add(lastToken);

                        if (stopCondition) break;
                    }

                    // --- C. 验证结果 ---
                    // 使用动态阈值：如果生成的 token 数量达到预期，则判定为成功
                    if (generatedTokens.Count >= minExpectedTokens)
                    {
                        success = true;
                        break;
                    }

                    retryCount++;
                    if (DebugMode) DebugLog($"[T2S] Insufficient tokens ({generatedTokens.Count} < {minExpectedTokens}). Retrying {retryCount}/{maxRetries}...");

                }

                if (!success)
                {
                    throw new Exception($"T2S 推理失败：在 {maxRetries} 次重试后仍未能生成有效的语义序列。请检查模型状态。");
                }

                DebugLog($"[T2S] Total tokens in list: {generatedTokens.Count}");

                // 剔除 EOS
                var eosIndex = generatedTokens.FindIndex(t => t >= 1024);
                long[] result;
                if (eosIndex >= 0)
                {
                    DebugLog($"[T2S] EOS found at index {eosIndex}, trimming...");
                    result = generatedTokens.Take(eosIndex).ToArray();
                }
                else
                {
                    result = generatedTokens.ToArray();
                }

                DebugLog($"[T2S] Final semantic tokens count: {result.Length}");
                if (DebugMode && result.Length > 5)
                    DebugLog($"[T2S] First 5 tokens: {string.Join(", ", result.Take(5))}");

                // 匹配 Python: y[0, -1] = 0
                if (result.Length > 0)
                {
                    result[result.Length - 1] = 0;
                    DebugLog("[T2S] Set last token to 0 (matching Python)");
                }

                return result;
            }
            finally
            {
                ReturnFlat(refBertFlat);
                ReturnFlat(textBertFlat);
            }
        }

        /// <summary>
        /// 流式 T2S 推理：在解码过程中每生成 chunkSize 个 token 就调用一次 vocoder 并输出音频块
        /// </summary>
        /// <param name="textSeq">文本序列</param>
        /// <param name="textBert">文本 BERT 特征</param>
        /// <param name="refSeq">参考序列</param>
        /// <param name="refBert">参考 BERT 特征</param>
        /// <param name="sslContent">SSL 内容</param>
        /// <param name="refAudio32k">参考音频 (32kHz)</param>
        /// <param name="ge">可选: Prompt Encoder ge 输出</param>
        /// <param name="geAdvanced">可选: Prompt Encoder ge_advanced 输出</param>
        /// <param name="chunkSize">每多少个 token 触发一次音频回调</param>
        /// <param name="onAudioChunk">音频块回调 (采样率 32kHz)</param>
        public void RunT2SStreaming(
            long[] textSeq, float[,] textBert,
            long[] refSeq, float[,] refBert, float[] sslContent,
            float[] refAudio32k, float[]? ge, float[]? geAdvanced,
            int chunkSize,
            Action<float[], bool> onAudioChunk,
            float speed = 1.0f)  // 添加 speed 参数
        {
            if (_t2sEncoder == null || _firstStageDecoder == null || _stageDecoder == null || _vocoder == null)
                throw new InvalidOperationException("Models not loaded.");

            // === 1. T2S Encoder ===
            float[] refBertFlat = Flatten(refBert);
            float[] textBertFlat = Flatten(textBert);
            try
            {
                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("ref_seq", new DenseTensor<long>(refSeq, new[] { 1, refSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("ref_bert", new DenseTensor<float>(refBertFlat.AsMemory(0, refBert.Length), new[] { refBert.GetLength(0), refBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("text_bert", new DenseTensor<float>(textBertFlat.AsMemory(0, textBert.Length), new[] { textBert.GetLength(0), textBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("ssl_content", new DenseTensor<float>(sslContent, new[] { 1, 768, sslContent.Length / 768 }))
                };

                using var encoderResults = _t2sEncoder.Run(encoderInputs);
                var x = encoderResults.First(r => r.Name == "x").AsTensor<float>().Clone();
                var prompts = encoderResults.First(r => r.Name == "prompts").AsTensor<long>().Clone();

                // === 2. T2S Decoder Loop with Streaming and Retry ===
                // 动态阈值：使用 2x 乘数兼容中英文
                int minExpectedTokens = Math.Max(8, textSeq.Length * 2);
                int maxRetries = 10;
                int retryCount = 0;
                bool success = false;

                while (retryCount < maxRetries)
                {
                    List<long> generatedTokens = new List<long>();
                    int lastVocoderPosition = 0;

                    // First Stage Decoder
                    var firstStageInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("x", x),
                    NamedOnnxValue.CreateFromTensor("prompts", prompts)
                };
                    using var firstStageResults = _firstStageDecoder.Run(firstStageInputs);

                    var yTensor = firstStageResults.First(r => r.Name == "y").AsTensor<long>().Clone();
                    var yEmbTensor = firstStageResults.First(r => r.Name == "y_emb").AsTensor<float>().Clone();

                    // 检查 First Stage 是否直接输出 EOS
                    var firstToken = yTensor.ToArray().Last();
                    if (firstToken >= 1024)
                    {
                        retryCount++;
                        if (DebugMode) DebugLog($"[Streaming] First Stage output EOS directly, retrying {retryCount}/{maxRetries}...");
                        continue;
                    }

                    generatedTokens.Add(firstToken);

                    // KV Cache
                    var kvCache = new List<Tensor<float>>();
                    for (int layer = 0; layer < 24; layer++)
                    {
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>().Clone());
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>().Clone());
                    }

                    // Stage Decoder Loop
                    var currentYTensor = yTensor.Clone();
                    var currentYEmbTensor = yEmbTensor.Clone();
                    int numLayers = kvCache.Count / 2;

                    for (int i = 0; i < 500; i++)
                    {
                        var stageInputs = new List<NamedOnnxValue>();
                        stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                        stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));

                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", kvCache[layer * 2]));
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", kvCache[layer * 2 + 1]));
                        }

                        using var stageResults = _stageDecoder.Run(stageInputs);
                        currentYTensor = stageResults.First(r => r.Name == "y").AsTensor<long>().Clone();
                        currentYEmbTensor = stageResults.First(r => r.Name == "y_emb").AsTensor<float>().Clone();
                        var stopCondition = stageResults.First(r => r.Name == "stop_condition_tensor").AsTensor<bool>().First();

                        kvCache.Clear();
                        for (int layer = 0; layer < numLayers; layer++)
                        {
                            kvCache.Add(stageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>().Clone());
                            kvCache.Add(stageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>().Clone());
                        }

                        var lastToken = currentYTensor.ToArray().Last();
                        generatedTokens.Add(lastToken);

                        // === 流式输出检查 ===
                        int currentTokenCount = generatedTokens.Count;
                        bool shouldEmitChunk = (currentTokenCount - lastVocoderPosition) >= chunkSize;
                        bool isEOS = stopCondition || lastToken >= 1024;

                        if (shouldEmitChunk || isEOS)
                        {
                            // 准备 semantic tokens (剔除 EOS)
                            int tokenLen = 0;
                            while (tokenLen < generatedTokens.Count && generatedTokens[tokenLen] < 1024) tokenLen++;

                            if (tokenLen > 0)
                            {
                                long[] semanticTokens = new long[tokenLen];
                                for (int k = 0; k < tokenLen; k++) semanticTokens[k] = generatedTokens[k];

                                // 只在最终块时将最后一个 token 设为 0 (匹配 Python)
                                if (isEOS)
                                {
                                    semanticTokens[semanticTokens.Length - 1] = 0;
                                }

                                // 调用 Vocoder，传入 speed 参数
                                var audio = RunVocoder(textSeq, semanticTokens, refAudio32k, out int audioLen, ge, geAdvanced, speed);
                                try
                                {
                                    // 提取当前段音频并复制（RunT2SStreaming 非零 GC 路径保持旧行为但修复长度）
                                    float[] chunk = new float[audioLen];
                                    Array.Copy(audio, 0, chunk, 0, audioLen);
                                    onAudioChunk(chunk, isEOS);
                                }
                                finally
                                {
                                    ReturnAudioBuffer(audio);
                                }
                                lastVocoderPosition = currentTokenCount;
                            }
                        }

                        if (isEOS) break;
                    }

                    // 验证结果：使用动态阈值
                    if (generatedTokens.Count >= minExpectedTokens)
                    {
                        success = true;
                        break;
                    }

                    retryCount++;
                    if (DebugMode) DebugLog($"[Streaming] Insufficient tokens ({generatedTokens.Count} < {minExpectedTokens}), retrying {retryCount}/{maxRetries}...");
                }

                if (!success)
                {
                    throw new Exception($"流式 T2S 推理失败：在 {maxRetries} 次重试后仍未能生成有效的语义序列。");
                }
            }
            finally
            {
                ReturnFlat(refBertFlat);
                ReturnFlat(textBertFlat);
            }
        }

        /// <summary>
        /// 异步流式 T2S：输出 token chunks 而不是 audio，vocoder 由调用方在独立线程处理
        /// </summary>
        public void RunT2SStreamingTokens(
            long[] textSeq, float[,] textBert,
            long[] refSeq, float[,] refBert, float[] sslContent,
            int chunkSize,
            Action<long[], bool> onTokenChunk,  // (tokensSoFar, isFinal)
            Action? onRetry = null)  // 重试时调用，用于清空队列
        {
            if (_t2sEncoder == null || _firstStageDecoder == null || _stageDecoder == null)
                throw new InvalidOperationException("T2S models not loaded.");

            float[] refBertFlat = Flatten(refBert);
            float[] textBertFlat = Flatten(textBert);
            try
            {
                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("ref_seq", new DenseTensor<long>(refSeq, new[] { 1, refSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("ref_bert", new DenseTensor<float>(refBertFlat.AsMemory(0, refBert.Length), new[] { (int)refBert.GetLength(0), (int)refBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("text_bert", new DenseTensor<float>(textBertFlat.AsMemory(0, textBert.Length), new[] { (int)textBert.GetLength(0), (int)textBert.GetLength(1) })),
                    NamedOnnxValue.CreateFromTensor("ssl_content", new DenseTensor<float>(sslContent, new[] { 1, 768, sslContent.Length / 768 }))
                };

                using var encoderResults = _t2sEncoder.Run(encoderInputs);
                var x = encoderResults.First(r => r.Name == "x").AsTensor<float>().Clone();
                var prompts = encoderResults.First(r => r.Name == "prompts").AsTensor<long>().Clone();

                // 动态阈值：使用 2x 乘数兼容中英文（流式模式）
                int minExpectedTokens = Math.Max(8, textSeq.Length * 2);
                int maxRetries = 10;
                int retryCount = 0;
                bool success = false;

                while (retryCount < maxRetries)
                {
                    // 重试时通知调用方清空之前失败的 tokens
                    if (retryCount > 0)
                    {
                        onRetry?.Invoke();
                        if (DebugMode) DebugLog($"[T2S Streaming] Retry {retryCount}/{maxRetries}, clearing previous tokens...");
                    }

                    List<long> generatedTokens = new List<long>();
                    int lastEmitPosition = 0;


                    // 延迟释放容器：持有 KV Cache 数据的引用
                    IDisposable? activeResults = null;

                    // First Stage Setup
                    var firstStageInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("x", x),
                    NamedOnnxValue.CreateFromTensor("prompts", prompts)
                };

                    // 手动管理 firstStageResults 生命周期
                    var firstStageResults = _firstStageDecoder.Run(firstStageInputs);
                    activeResults = firstStageResults; // 持有它

                    var yTensor = firstStageResults.First(r => r.Name == "y").AsTensor<long>();
                    // 注意：这里不 Clone，依赖 activeResults 存活
                    var yEmbTensor = firstStageResults.First(r => r.Name == "y_emb").AsTensor<float>();

                    // Safe access because activeResults is alive
                    var firstToken = ((DenseTensor<long>)yTensor).Buffer.Span[((DenseTensor<long>)yTensor).Buffer.Length - 1];
                    if (firstToken >= 1024)
                    {
                        retryCount++;
                        activeResults.Dispose(); // Retry case: clean up
                        continue;
                    }

                    generatedTokens.Add(firstToken);

                    // Initialize KV Cache (No Clone, references memory in firstStageResults)
                    var kvCache = new List<Tensor<float>>(48);
                    for (int layer = 0; layer < 24; layer++)
                    {
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>());
                        kvCache.Add(firstStageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>());
                    }

                    var currentYTensor = yTensor;
                    var currentYEmbTensor = yEmbTensor;
                    int numLayers = kvCache.Count / 2;

                    // 复用 List 避免每次分配
                    var stageInputs = new List<NamedOnnxValue>(100);

                    try
                    {
                        for (int i = 0; i < 500; i++)
                        {
                            stageInputs.Clear();
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy", currentYTensor));
                            stageInputs.Add(NamedOnnxValue.CreateFromTensor("iy_emb", currentYEmbTensor));

                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_k_layer_{layer}", kvCache[layer * 2]));
                                stageInputs.Add(NamedOnnxValue.CreateFromTensor($"past_v_layer_{layer}", kvCache[layer * 2 + 1]));
                            }

                            // Run Inference
                            var stageResults = _stageDecoder.Run(stageInputs);

                            // Critical: Dispose the OLD results now that we have NEW results
                            // The old kvCache/yTensor inputs are no longer needed
                            activeResults?.Dispose();
                            activeResults = stageResults; // Update active holder

                            // Update references (No Clone)
                            currentYTensor = stageResults.First(r => r.Name == "y").AsTensor<long>();
                            currentYEmbTensor = stageResults.First(r => r.Name == "y_emb").AsTensor<float>();

                            var stopCondition = stageResults.First(r => r.Name == "stop_condition_tensor").AsTensor<bool>().GetValue(0);

                            // Update KV Cache pointers for next iteration
                            kvCache.Clear();
                            for (int layer = 0; layer < numLayers; layer++)
                            {
                                kvCache.Add(stageResults.First(r => r.Name == $"present_k_layer_{layer}").AsTensor<float>());
                                kvCache.Add(stageResults.First(r => r.Name == $"present_v_layer_{layer}").AsTensor<float>());
                            }

                            // Safe access
                            var denseY = (DenseTensor<long>)currentYTensor;
                            var lastToken = denseY.Buffer.Span[denseY.Buffer.Length - 1];
                            generatedTokens.Add(lastToken);

                            int currentCount = generatedTokens.Count;
                            bool shouldEmit = (currentCount - lastEmitPosition) >= chunkSize;
                            bool isEOS = stopCondition || lastToken >= 1024;

                            if (shouldEmit || isEOS)
                            {
                                // 优化：避免 TakeWhile().ToArray()，直接从 List 构造
                                int tokenLen = 0;
                                while (tokenLen < generatedTokens.Count && generatedTokens[tokenLen] < 1024) tokenLen++;

                                if (tokenLen > 0)
                                {
                                    long[] tokens = new long[tokenLen];
                                    for (int k = 0; k < tokenLen; k++) tokens[k] = generatedTokens[k];

                                    if (isEOS) tokens[tokens.Length - 1] = 0;
                                    onTokenChunk(tokens, isEOS);  // 只输出 tokens，不调用 vocoder
                                    lastEmitPosition = currentCount;
                                }
                            }

                            if (isEOS) break;
                        }
                    }
                    finally
                    {
                        activeResults?.Dispose();
                    }

                    if (generatedTokens.Count > 0 && generatedTokens.Last() < 1024)
                    {
                        // 检查是否是因为超过最大长度而中断，而非自然 EOS
                        if (generatedTokens.Count < minExpectedTokens) // 这里的 minExpectedTokens 是我们在外面定义的动态阈值
                        {
                            if (DebugMode) DebugLog($"[T2S Streaming] Incomplete generation ({generatedTokens.Count} < {minExpectedTokens}), retrying...");
                            retryCount++;
                            continue;
                        }
                    }

                    success = true;
                    break;
                }


                if (!success) throw new Exception($"流式 T2S 推理失败：{maxRetries} 次重试后仍未达到期望 tokens 数量 ({minExpectedTokens})");
            }
            finally
            {
                ReturnFlat(refBertFlat);
                ReturnFlat(textBertFlat);
            }
        }


        public float[] RunVocoder(long[] textSeq, long[] predSemantic, float[] refAudio32k, out int audioLength, float[]? ge = null, float[]? geAdvanced = null, float speed = 1.0f)
        {
            if (_vocoder == null) throw new InvalidOperationException("Vocoder not loaded.");
            audioLength = 0;

            long[] stretchedSemantic = predSemantic;
            // 语速控制：语义 Token 插值
            if (Math.Abs(speed - 1.0f) > 0.01f)
            {
                DebugLog($"[Vocoder] 执行语义 Token 插值变速: {speed}x");
                stretchedSemantic = StretchSemanticTokens(predSemantic, speed);
            }

            try
            {
                var vocoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("text_seq", new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                    NamedOnnxValue.CreateFromTensor("pred_semantic", new DenseTensor<long>(stretchedSemantic, new[] { 1, 1, stretchedSemantic.Length }))
                };

                if (_promptEncoder != null && ge != null && geAdvanced != null)
                {
                    vocoderInputs.Add(NamedOnnxValue.CreateFromTensor("ge", new DenseTensor<float>(ge, new[] { 1, ge.Length, 1 })));
                    vocoderInputs.Add(NamedOnnxValue.CreateFromTensor("ge_advanced", new DenseTensor<float>(geAdvanced, new[] { 1, geAdvanced.Length, 1 })));
                }
                else
                {
                    vocoderInputs.Add(NamedOnnxValue.CreateFromTensor("ref_audio", new DenseTensor<float>(refAudio32k, new[] { 1, refAudio32k.Length })));
                }

                using var results = _vocoder.Run(vocoderInputs);
                var audioTensor = (DenseTensor<float>)results.First().AsTensor<float>();

                audioLength = audioTensor.Buffer.Length;
                // 优化：从张量 Buffer 直接复制到 rented 数组，避免 ToArray() 产生的大对象分配
                float[] audio = ArrayPool<float>.Shared.Rent(audioLength);
                audioTensor.Buffer.Span.CopyTo(audio);
                return audio;
            }
            finally
            {
                if (stretchedSemantic != predSemantic)
                {
                    ArrayPool<long>.Shared.Return(stretchedSemantic);
                }
            }
        }

        private float[] Flatten(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            int len = rows * cols;
            float[] flat = ArrayPool<float>.Shared.Rent(len);
            Buffer.BlockCopy(array, 0, flat, 0, len * sizeof(float));
            return flat;
        }

        private void ReturnFlat(float[] flat)
        {
            if (flat != null) ArrayPool<float>.Shared.Return(flat);
        }

        /// <summary>
        /// 语义 Token 插值 (Nearest Neighbor)
        /// </summary>
        private long[] StretchSemanticTokens(long[] tokens, float speed)
        {
            if (tokens == null || tokens.Length == 0) return Array.Empty<long>();

            int newLen = (int)Math.Round(tokens.Length / speed);
            if (newLen < 1) newLen = 1;

            long[] result = ArrayPool<long>.Shared.Rent(newLen);
            for (int i = 0; i < newLen; i++)
            {
                int oldIdx = (int)Math.Floor(i * speed);
                if (oldIdx >= tokens.Length) oldIdx = tokens.Length - 1;
                result[i] = tokens[oldIdx];
            }
            return result;
        }

        public void ReturnAudioBuffer(float[] buffer)
        {
            if (buffer != null) ArrayPool<float>.Shared.Return(buffer);
        }

        public void Dispose()
        {
            _t2sEncoder?.Dispose();
            _firstStageDecoder?.Dispose();
            _stageDecoder?.Dispose();
            _vocoder?.Dispose();
            _promptEncoder?.Dispose();
            _hubert?.Dispose();
            _svModel?.Dispose();
        }
    }
}

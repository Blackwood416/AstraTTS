using System.Buffers;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// 新版推理引擎 (基于 GPT-SoVITS_minimal_inference 架构)
    /// 支持:
    /// - 新的模型结构 (gpt_encoder + gpt_step + sovits)
    /// - KV-Cache 预分配
    /// - 语速调节
    /// - 流式推理
    /// </summary>
    public class InferenceEngineV2 : IDisposable
    {
        // ============================================================
        // 调试控制
        // ============================================================
        public static bool DebugMode { get; set; } = false;

        private static void DebugLog(string message)
        {
            if (DebugMode) Console.WriteLine($"[V2] {message}");
        }

        // ============================================================
        // ONNX Sessions
        // ============================================================
        private InferenceSession? _sslSession;       // ssl.onnx (HuBERT)
        private InferenceSession? _bertSession;      // bert.onnx
        private InferenceSession? _vqEncoderSession; // vq_encoder.onnx
        private InferenceSession? _gptEncoderSession;// gpt_encoder.onnx
        private InferenceSession? _gptStepSession;   // gpt_step.onnx
        private InferenceSession? _sovitsSession;    // sovits.onnx
        private InferenceSession? _svSession;        // sv.onnx (ERes2NetV2, V2ProPlus)

        // ============================================================
        // 模型配置
        // ============================================================
        private Config.V2ModelConfig? _modelConfig;
        private int _maxKvLen = 2000; // 默认值改为 2000，与 minimal_inference 一致
        private int[] _kvCacheShape = { 24, 1, 2000, 512 }; // 从模型动态获取
        private string _modelVersion = "v2";
        private bool _isProVersion = false;

        // ============================================================
        // 随机采样器
        // ============================================================
        private readonly Random _random = new Random();

        // ============================================================
        // 属性
        // ============================================================
        public int SamplingRate => _modelConfig?.Data?.SamplingRate ?? 32000;
        public string ModelVersion => _modelVersion;
        public bool IsLoaded => _sslSession != null && _sovitsSession != null;

        // ============================================================
        // Session 配置
        // ============================================================
        private SessionOptions CreateSessionOptions(bool useDirectML)
        {
            var options = new SessionOptions();

            // 线程配置
            options.IntraOpNumThreads = Environment.ProcessorCount / 2;
            options.InterOpNumThreads = 1;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

            // 禁用图优化以避免 Shape Inference 错误
            // 问题: 某些模型在优化时遇到 Cannot parse data from external tensors 错误
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;

            // 内存优化
            options.EnableCpuMemArena = true;

            if (useDirectML)
            {
                options.EnableMemoryPattern = false;
                options.AppendExecutionProvider_DML(0);
                Console.WriteLine("[V2] DirectML 加速已启用");
            }
            else
            {
                options.EnableMemoryPattern = true;
            }

            return options;
        }

        // ============================================================
        // 模型加载
        // ============================================================
        public void LoadModels(string modelDir, bool useDirectML = false)
        {
            var sw = Stopwatch.StartNew();
            Console.WriteLine($"[V2] 加载模型目录: {modelDir}");

            // 加载配置
            var configPath = Path.Combine(modelDir, "config.json");
            _modelConfig = Config.V2ModelConfig.Load(configPath);
            _maxKvLen = _modelConfig.ExportOptions?.MaxLen ?? 1000;
            _modelVersion = _modelConfig.Model?.Version ?? "v2";
            _isProVersion = _modelVersion.Contains("Pro", StringComparison.OrdinalIgnoreCase);

            Console.WriteLine($"[V2] 模型版本: {_modelVersion}, KV-Cache 最大长度: {_maxKvLen}");

            var options = CreateSessionOptions(useDirectML);

            // 并行加载所有模型
            var tasks = new List<Action>
            {
                () => _sslSession = new InferenceSession(Path.Combine(modelDir, "ssl.onnx"), options),
                () => _bertSession = new InferenceSession(Path.Combine(modelDir, "bert.onnx"), options),
                () => _vqEncoderSession = new InferenceSession(Path.Combine(modelDir, "vq_encoder.onnx"), options),
                () => _gptEncoderSession = new InferenceSession(Path.Combine(modelDir, "gpt_encoder.onnx"), options),
                () => _gptStepSession = new InferenceSession(Path.Combine(modelDir, "gpt_step.onnx"), options),
                () => _sovitsSession = new InferenceSession(Path.Combine(modelDir, "sovits.onnx"), options),
            };

            Parallel.Invoke(tasks.ToArray());

            // 加载 SV 模型 (V2ProPlus 需要)
            if (_isProVersion)
            {
                var svPath = Path.Combine(modelDir, "sv.onnx");
                if (File.Exists(svPath))
                {
                    _svSession = new InferenceSession(svPath, options);
                    Console.WriteLine("[V2] SV 模型 (ERes2NetV2) 已加载");
                }
                else
                {
                    Console.WriteLine("[V2] ⚠️ 未找到 sv.onnx，V2ProPlus 颠音功能可能受限");
                }
            }

            // 从 gpt_step 模型获取实际的 KV Cache 形状
            try
            {
                var kCacheInput = _gptStepSession!.InputMetadata["k_cache"];
                var dims = kCacheInput.Dimensions;
                if (dims.Length == 4)
                {
                    // 处理动态维度：ONNX 用 -1 表示动态维度，需要替换为实际值
                    int layers = dims[0] > 0 ? dims[0] : 24;      // 层数
                    int batchSize = dims[1] > 0 ? dims[1] : 1;    // 批次大小，通常为 1
                    int maxLen = dims[2] > 0 ? dims[2] : 2000;    // 最大序列长度
                    int headDim = dims[3] > 0 ? dims[3] : 512;    // 注意力头维度

                    _kvCacheShape = new[] { layers, batchSize, maxLen, headDim };
                    _maxKvLen = maxLen;
                    Console.WriteLine($"[V2] KV Cache 形状: [{layers}, {batchSize}, {maxLen}, {headDim}] (原始: [{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}])");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[V2] ⚠️ 无法从模型读取 KV Cache 形状，使用默认值: {ex.Message}");
            }

            sw.Stop();
            Console.WriteLine($"[V2] 模型加载完成，耗时: {sw.ElapsedMilliseconds}ms");
        }

        // ============================================================
        // SSL 特征提取 (HuBERT)
        // ============================================================
        public float[] ExtractSSL(float[] audio16k)
        {
            if (_sslSession == null) throw new InvalidOperationException("SSL 模型未加载");

            var inputTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("audio", inputTensor)
            };

            using var results = _sslSession.Run(inputs);
            var output = results.First().AsTensor<float>();

            // 输出形状: [1, 768, T]
            DebugLog($"SSL 输出形状: [1, {output.Dimensions[1]}, {output.Dimensions[2]}]");

            return output.ToArray();
        }

        // ============================================================
        // VQ 编码
        // ============================================================
        public long[] EncodeVQ(float[] sslContent)
        {
            if (_vqEncoderSession == null) throw new InvalidOperationException("VQ Encoder 未加载");

            int T = sslContent.Length / 768;
            var inputTensor = new DenseTensor<float>(sslContent, new[] { 1, 768, T });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("ssl_content", inputTensor)
            };

            using var results = _vqEncoderSession.Run(inputs);
            var codes = results.First().AsTensor<long>();

            DebugLog($"VQ 编码输出: {codes.Dimensions[2]} tokens");

            // 返回 [T] 形状的 codes
            return codes.ToArray();
        }

        // ============================================================
        // BERT 特征提取
        // ============================================================
        public float[] ExtractBERT(long[] inputIds, long[] attentionMask, long[] tokenTypeIds)
        {
            if (_bertSession == null) throw new InvalidOperationException("BERT 模型未加载");

            int seqLen = inputIds.Length;
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, new[] { 1, seqLen })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, new[] { 1, seqLen })),
                NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(tokenTypeIds, new[] { 1, seqLen }))
            };

            using var results = _bertSession.Run(inputs);
            var hiddenStates = results.First().AsTensor<float>();

            DebugLog($"BERT 输出形状: [{hiddenStates.Dimensions[0]}, {hiddenStates.Dimensions[1]}, {hiddenStates.Dimensions[2]}]");

            return hiddenStates.ToArray();
        }

        // ============================================================
        // 计算 FBank 特征 (Kaldi 风格 80-bin)
        // ============================================================
        /// <summary>
        /// 计算 Kaldi 风格的 FBank 特征，用于 SV 模型
        /// </summary>
        /// <param name="audio16k">16kHz 单声道音频</param>
        /// <returns>FBank 特征 [T, 80]</returns>
        public float[,] ComputeFBank(float[] audio16k)
        {
            // FBank 参数 (Kaldi 默认值)
            const int sampleRate = 16000;
            const int numMelBins = 80;
            const float frameLengthMs = 25.0f;
            const float frameShiftMs = 10.0f;
            const int nFft = 512;

            int frameLength = (int)(sampleRate * frameLengthMs / 1000);  // 400
            int frameShift = (int)(sampleRate * frameShiftMs / 1000);    // 160
            int numFrames = (audio16k.Length - frameLength) / frameShift + 1;

            if (numFrames <= 0)
            {
                return new float[1, numMelBins];  // 音频太短
            }

            var fbank = new float[numFrames, numMelBins];

            // 简化实现：使用基本的 FFT 和 Mel 滤波器
            // 实际应用中应使用与 Python Kaldi 一致的实现
            var melFilters = CreateMelFilterBank(nFft / 2 + 1, numMelBins, sampleRate, 20, sampleRate / 2);
            var window = CreateHannWindow(frameLength);

            for (int frame = 0; frame < numFrames; frame++)
            {
                int offset = frame * frameShift;

                // 加窗
                var windowed = new double[nFft];
                for (int i = 0; i < frameLength && offset + i < audio16k.Length; i++)
                {
                    windowed[i] = audio16k[offset + i] * window[i];
                }

                // FFT (简化版本 - 使用 MathNet 或其他库效果更好)
                var spectrum = ComputePowerSpectrum(windowed);

                // Mel 滤波器组
                for (int mel = 0; mel < numMelBins; mel++)
                {
                    double sum = 0;
                    for (int k = 0; k < spectrum.Length; k++)
                    {
                        sum += spectrum[k] * melFilters[mel, k];
                    }
                    fbank[frame, mel] = (float)Math.Log(Math.Max(sum, 1e-10));
                }
            }

            DebugLog($"FBank 特征: [{numFrames}, {numMelBins}]");
            return fbank;
        }

        // 创建 Hann 窗函数
        private static double[] CreateHannWindow(int length)
        {
            var window = new double[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (length - 1)));
            }
            return window;
        }

        // 创建 Mel 滤波器组
        private static double[,] CreateMelFilterBank(int numFftBins, int numMelBins, int sampleRate, double lowFreq, double highFreq)
        {
            var filters = new double[numMelBins, numFftBins];

            // Hz to Mel
            Func<double, double> hzToMel = hz => 1127.0 * Math.Log(1 + hz / 700.0);
            Func<double, double> melToHz = mel => 700.0 * (Math.Exp(mel / 1127.0) - 1);

            double melLow = hzToMel(lowFreq);
            double melHigh = hzToMel(highFreq);
            double melStep = (melHigh - melLow) / (numMelBins + 1);

            var melPoints = new double[numMelBins + 2];
            for (int i = 0; i < melPoints.Length; i++)
            {
                melPoints[i] = melLow + i * melStep;
            }

            var hzPoints = melPoints.Select(m => melToHz(m)).ToArray();
            var binPoints = hzPoints.Select(hz => (int)Math.Floor((numFftBins - 1) * 2 * hz / sampleRate)).ToArray();

            for (int m = 0; m < numMelBins; m++)
            {
                int left = binPoints[m];
                int center = binPoints[m + 1];
                int right = binPoints[m + 2];

                for (int k = left; k <= center && k < numFftBins; k++)
                {
                    if (center != left)
                        filters[m, k] = (double)(k - left) / (center - left);
                }
                for (int k = center; k <= right && k < numFftBins; k++)
                {
                    if (right != center)
                        filters[m, k] = (double)(right - k) / (right - center);
                }
            }

            return filters;
        }

        // 计算功率谱 (简化版 FFT)
        private static double[] ComputePowerSpectrum(double[] signal)
        {
            int n = signal.Length;
            int halfN = n / 2 + 1;
            var spectrum = new double[halfN];

            // 简化 DFT (实际应用应使用 FFT 库)
            for (int k = 0; k < halfN; k++)
            {
                double realSum = 0, imagSum = 0;
                for (int t = 0; t < n; t++)
                {
                    double angle = -2 * Math.PI * k * t / n;
                    realSum += signal[t] * Math.Cos(angle);
                    imagSum += signal[t] * Math.Sin(angle);
                }
                spectrum[k] = (realSum * realSum + imagSum * imagSum) / n;
            }

            return spectrum;
        }

        // ============================================================
        // 说话人嵌入提取 (SV / ERes2NetV2)
        // ============================================================
        /// <summary>
        /// 从 16kHz 音频提取说话人嵌入 (20480 维)
        /// 仅适用于 V2ProPlus 模型
        /// </summary>
        public float[]? ExtractSpeakerEmbedding(float[] audio16k)
        {
            if (_svSession == null)
            {
                DebugLog("SV 模型未加载，无法提取说话人嵌入");
                return null;
            }

            // 1. 计算 FBank 特征
            var fbank = ComputeFBank(audio16k);
            int numFrames = fbank.GetLength(0);
            int numMelBins = fbank.GetLength(1);

            // 2. 运行 SV 模型
            var fbankTensor = new DenseTensor<float>(Flatten(fbank), new[] { 1, numFrames, numMelBins });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("fbank_feat", fbankTensor)
            };

            using var results = _svSession.Run(inputs);
            var svEmb = results.First().AsTensor<float>().ToArray();

            DebugLog($"说话人嵌入提取完成: {svEmb.Length} 维");

            // 3. 处理维度 (确保 20480 维)
            const int expectedDim = 20480;
            if (svEmb.Length != expectedDim)
            {
                var padded = new float[expectedDim];
                Array.Copy(svEmb, padded, Math.Min(svEmb.Length, expectedDim));
                return padded;
            }

            return svEmb;
        }

        // ============================================================
        // Top-K 采样
        // ============================================================
        private long SampleTopK(float[] topkValues, long[] topkIndices, float temperature = 1.0f)
        {
            int k = topkValues.Length;

            // 应用温度
            if (Math.Abs(temperature - 1.0f) > 0.001f)
            {
                for (int i = 0; i < k; i++)
                {
                    topkValues[i] /= temperature;
                }
            }

            // Softmax
            float maxVal = topkValues.Max();
            float sum = 0;
            for (int i = 0; i < k; i++)
            {
                topkValues[i] = MathF.Exp(topkValues[i] - maxVal);
                sum += topkValues[i];
            }
            for (int i = 0; i < k; i++)
            {
                topkValues[i] /= sum;
            }

            // 采样
            float r = (float)_random.NextDouble();
            float cumulative = 0;
            for (int i = 0; i < k; i++)
            {
                cumulative += topkValues[i];
                if (r <= cumulative)
                {
                    return topkIndices[i];
                }
            }

            return topkIndices[k - 1];
        }

        // ============================================================
        // 流式推理 (生成器模式)
        // ============================================================
        /// <summary>
        /// 流式推理：生成音频块
        /// </summary>
        /// <param name="phonemeIds">音素 ID 序列</param>
        /// <param name="bertFeature">BERT 特征 [1024, T]</param>
        /// <param name="promptCodes">参考音频的 VQ codes</param>
        /// <param name="referSpec">参考音频的频谱</param>
        /// <param name="svEmb">说话人嵌入 (V2ProPlus 需要)</param>
        /// <param name="topK">Top-K 采样</param>
        /// <param name="temperature">采样温度</param>
        /// <param name="noiseScale">噪声系数</param>
        /// <param name="speed">语速 (0.5-2.0)</param>
        /// <param name="chunkSize">每多少 token 输出一次</param>
        /// <param name="onAudioChunk">音频块回调</param>
        public void InferStream(
            long[] refPhonemeIds,
            float[,] refBertFeature,
            long[] phonemeIds,
            float[,] bertFeature,
            long[] promptCodes,
            float[,] referSpec,
            float[]? svEmb,
            int topK = 15,
            float temperature = 1.0f,
            float noiseScale = 0.35f,
            float speed = 1.0f,
            int chunkSize = 24,
            Action<float[], bool>? onAudioChunk = null)
        {
            if (_gptEncoderSession == null || _gptStepSession == null || _sovitsSession == null)
                throw new InvalidOperationException("模型未加载");

            var sw = Stopwatch.StartNew();

            // ========== 1. GPT Encoder ==========
            DebugLog("运行 GPT Encoder...");

            // 拼接参考和目标音素 (Python: phones1 + phones2)
            var allPhonemeIds = refPhonemeIds.Concat(phonemeIds).ToArray();

            // 拼接参考和目标 BERT 特征 (Python: concatenate([bert1, bert2], axis=1))
            int bertDim = bertFeature.GetLength(0);  // 1024
            int refBertLen = refBertFeature.GetLength(1);
            int targetBertLen = bertFeature.GetLength(1);
            int totalBertLen = refBertLen + targetBertLen;

            var allBertFeature = new float[bertDim, totalBertLen];
            // 复制参考 BERT
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < refBertLen; t++)
                    allBertFeature[d, t] = refBertFeature[d, t];
            // 复制目标 BERT
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < targetBertLen; t++)
                    allBertFeature[d, refBertLen + t] = bertFeature[d, t];

            DebugLog($"音素拼接: ref={refPhonemeIds.Length} + target={phonemeIds.Length} = {allPhonemeIds.Length}");
            DebugLog($"BERT 拼接: ref={refBertLen} + target={targetBertLen} = {totalBertLen}");

            int textLen = allPhonemeIds.Length;
            int promptLen = promptCodes.Length;

            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("phoneme_ids", new DenseTensor<long>(allPhonemeIds, new[] { 1, textLen })),
                NamedOnnxValue.CreateFromTensor("prompts", new DenseTensor<long>(promptCodes, new[] { 1, promptLen })),
                NamedOnnxValue.CreateFromTensor("bert_feature", new DenseTensor<float>(Flatten(allBertFeature), new[] { 1, bertDim, totalBertLen }))
            };

            float[] topkValues, kCache, vCache;
            long[] topkIndices, xLen, yLen;

            using (var encoderResults = _gptEncoderSession.Run(encoderInputs))
            {
                topkValues = encoderResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                topkIndices = encoderResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();
                kCache = encoderResults.First(r => r.Name == "k_cache").AsTensor<float>().ToArray();
                vCache = encoderResults.First(r => r.Name == "v_cache").AsTensor<float>().ToArray();
                xLen = encoderResults.First(r => r.Name == "x_len").AsTensor<long>().ToArray();
                yLen = encoderResults.First(r => r.Name == "y_len").AsTensor<long>().ToArray();
            }

            DebugLog($"GPT Encoder 完成，耗时: {sw.ElapsedMilliseconds}ms");

            // ========== 2. GPT Step Loop ==========
            var generatedTokens = new List<long>();
            long currentToken = SampleTopK(topkValues, topkIndices, temperature);
            generatedTokens.Add(currentToken);

            int lastEmitPosition = 0;
            const int maxSteps = 1500;
            const long EOS_TOKEN = 1024;

            // 获取 KV Cache 维度信息
            var kCacheTensor = new DenseTensor<float>(kCache, GetKvCacheShape());
            var vCacheTensor = new DenseTensor<float>(vCache, GetKvCacheShape());

            int lastAudioLength = 0;
            for (int step = 0; step < maxSteps; step++)
            {
                var stepInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("samples", new DenseTensor<long>(new[] { currentToken }, new[] { 1, 1 })),
                    NamedOnnxValue.CreateFromTensor("k_cache", kCacheTensor),
                    NamedOnnxValue.CreateFromTensor("v_cache", vCacheTensor),
                    NamedOnnxValue.CreateFromTensor("x_len", new DenseTensor<long>(xLen, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("y_len", new DenseTensor<long>(yLen, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("idx", new DenseTensor<long>(new[] { (long)step }, new[] { 1 }))
                };

                using var stepResults = _gptStepSession.Run(stepInputs);

                topkValues = stepResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                topkIndices = stepResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();

                // 更新 KV Cache
                var newKCache = stepResults.First(r => r.Name == "k_cache_new").AsTensor<float>();
                var newVCache = stepResults.First(r => r.Name == "v_cache_new").AsTensor<float>();
                kCacheTensor = newKCache.Clone() as DenseTensor<float> ?? throw new Exception("KV Cache 克隆失败");
                vCacheTensor = newVCache.Clone() as DenseTensor<float> ?? throw new Exception("KV Cache 克隆失败");

                currentToken = SampleTopK(topkValues, topkIndices, temperature);
                generatedTokens.Add(currentToken);

                // 检查是否应该输出音频块
                bool isEOS = currentToken >= EOS_TOKEN;
                bool shouldEmit = (generatedTokens.Count - lastEmitPosition) >= chunkSize;

                if ((shouldEmit || isEOS) && onAudioChunk != null)
                {
                    var tokensForVocoder = generatedTokens.Where(t => t < EOS_TOKEN).ToArray();
                    if (tokensForVocoder.Length > 0)
                    {
                        // 包含语速控制的 SoVITS
                        var fullAudio = RunSoVITS(tokensForVocoder, phonemeIds, referSpec, svEmb, noiseScale, speed);

                        // 关键：计算增量音频
                        int newSamplesCount = fullAudio.Length - lastAudioLength;
                        if (newSamplesCount > 0)
                        {
                            float[] chunk = new float[newSamplesCount];
                            Array.Copy(fullAudio, lastAudioLength, chunk, 0, newSamplesCount);
                            onAudioChunk(chunk, isEOS);
                            lastAudioLength = fullAudio.Length;
                        }
                        else if (isEOS)
                        {
                            onAudioChunk(Array.Empty<float>(), true);
                        }

                        lastEmitPosition = generatedTokens.Count;
                    }
                }

                if (isEOS) break;
            }

            sw.Stop();
            DebugLog($"推理完成，总耗时: {sw.ElapsedMilliseconds}ms，生成 {generatedTokens.Count} tokens");
        }

        /// <summary>
        /// 非流式推理：一次性返回所有 semantic tokens
        /// </summary>
        /// <param name="refPhonemeIds">参考文本音素 ID</param>
        /// <param name="refBertFeature">参考文本 BERT 特征 [1024, T_ref]</param>
        /// <param name="phonemeIds">目标文本音素 ID</param>
        /// <param name="bertFeature">目标文本 BERT 特征 [1024, T_target]</param>
        /// <param name="promptCodes">参考音频的 VQ codes</param>
        /// <param name="topK">Top-K 采样</param>
        /// <param name="temperature">采样温度</param>
        public long[] InferTokens(
            long[] refPhonemeIds,
            float[,] refBertFeature,
            long[] phonemeIds,
            float[,] bertFeature,
            long[] promptCodes,
            int topK = 15,
            float temperature = 1.0f)
        {
            if (_gptEncoderSession == null || _gptStepSession == null)
                throw new InvalidOperationException("模型未加载");

            // 拼接参考和目标音素 (Python: phones1 + phones2)
            var allPhonemeIds = refPhonemeIds.Concat(phonemeIds).ToArray();

            // 拼接参考和目标 BERT 特征 (Python: concatenate([bert1, bert2], axis=1))
            int bertDim = bertFeature.GetLength(0);  // 1024
            int refBertLen = refBertFeature.GetLength(1);
            int targetBertLen = bertFeature.GetLength(1);
            int totalBertLen = refBertLen + targetBertLen;

            var allBertFeature = new float[bertDim, totalBertLen];
            // 复制参考 BERT
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < refBertLen; t++)
                    allBertFeature[d, t] = refBertFeature[d, t];
            // 复制目标 BERT
            for (int d = 0; d < bertDim; d++)
                for (int t = 0; t < targetBertLen; t++)
                    allBertFeature[d, refBertLen + t] = bertFeature[d, t];

            DebugLog($"音素拼接: ref={refPhonemeIds.Length} + target={phonemeIds.Length} = {allPhonemeIds.Length}");
            DebugLog($"BERT 拼接: ref={refBertLen} + target={targetBertLen} = {totalBertLen}");

            int textLen = allPhonemeIds.Length;
            int promptLen = promptCodes.Length;

            // GPT Encoder
            var encoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("phoneme_ids", new DenseTensor<long>(allPhonemeIds, new[] { 1, textLen })),
                NamedOnnxValue.CreateFromTensor("prompts", new DenseTensor<long>(promptCodes, new[] { 1, promptLen })),
                NamedOnnxValue.CreateFromTensor("bert_feature", new DenseTensor<float>(Flatten(allBertFeature), new[] { 1, bertDim, totalBertLen }))
            };

            float[] topkValues, kCache, vCache;
            long[] topkIndices, xLen, yLen;

            using (var encoderResults = _gptEncoderSession.Run(encoderInputs))
            {
                topkValues = encoderResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                topkIndices = encoderResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();
                kCache = encoderResults.First(r => r.Name == "k_cache").AsTensor<float>().ToArray();
                vCache = encoderResults.First(r => r.Name == "v_cache").AsTensor<float>().ToArray();
                xLen = encoderResults.First(r => r.Name == "x_len").AsTensor<long>().ToArray();
                yLen = encoderResults.First(r => r.Name == "y_len").AsTensor<long>().ToArray();
            }

            // GPT Step Loop
            var generatedTokens = new List<long>();
            long currentToken = SampleTopK(topkValues, topkIndices, temperature);
            generatedTokens.Add(currentToken);

            DebugLog($"第一个采样 token: {currentToken}, TopK 前5: [{string.Join(", ", topkIndices.Take(5))}]");

            var kCacheTensor = new DenseTensor<float>(kCache, GetKvCacheShape());
            var vCacheTensor = new DenseTensor<float>(vCache, GetKvCacheShape());

            const int maxSteps = 1500;
            const long EOS_TOKEN = 1024;

            for (int step = 0; step < maxSteps; step++)
            {
                var stepInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("samples", new DenseTensor<long>(new[] { currentToken }, new[] { 1, 1 })),
                    NamedOnnxValue.CreateFromTensor("k_cache", kCacheTensor),
                    NamedOnnxValue.CreateFromTensor("v_cache", vCacheTensor),
                    NamedOnnxValue.CreateFromTensor("x_len", new DenseTensor<long>(xLen, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("y_len", new DenseTensor<long>(yLen, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("idx", new DenseTensor<long>(new[] { (long)step }, new[] { 1 }))
                };

                using var stepResults = _gptStepSession.Run(stepInputs);

                topkValues = stepResults.First(r => r.Name == "topk_values").AsTensor<float>().ToArray();
                topkIndices = stepResults.First(r => r.Name == "topk_indices").AsTensor<long>().ToArray();

                var newKCache = stepResults.First(r => r.Name == "k_cache_new").AsTensor<float>();
                var newVCache = stepResults.First(r => r.Name == "v_cache_new").AsTensor<float>();
                kCacheTensor = newKCache.Clone() as DenseTensor<float> ?? throw new Exception("Clone failed");
                vCacheTensor = newVCache.Clone() as DenseTensor<float> ?? throw new Exception("Clone failed");

                currentToken = SampleTopK(topkValues, topkIndices, temperature);

                if (step < 5 || currentToken >= EOS_TOKEN)
                {
                    DebugLog($"Step {step}: token={currentToken}, top5=[{string.Join(",", topkIndices.Take(5))}]");
                }

                if (currentToken >= EOS_TOKEN)
                {
                    DebugLog($"EOS 检测到，总共生成 {generatedTokens.Count} tokens");
                    break;
                }
                generatedTokens.Add(currentToken);
            }

            // 设置最后一个 token 为 0 (匹配 Python)
            var result = generatedTokens.ToArray();
            if (result.Length > 0)
            {
                result[result.Length - 1] = 0;
            }

            return result;
        }

        // ============================================================
        // SoVITS 音频合成
        // ============================================================
        public float[] RunSoVITS(
            long[] predSemantic,
            long[] textSeq,
            float[,] referSpec,
            float[]? svEmb,
            float noiseScale = 0.35f,
            float speed = 1.0f)
        {
            if (_sovitsSession == null) throw new InvalidOperationException("SoVITS 未加载");

            // 1. 语速控制：通过语义 Token 插值实现（相比波形 OLA 更加稳定，不影响音调）
            if (Math.Abs(speed - 1.0f) > 0.01f)
            {
                DebugLog($"执行语义 Token 插值变速: {speed}x");
                predSemantic = StretchSemanticTokens(predSemantic, speed);
            }

            int specDim = referSpec.GetLength(0);
            int specLen = referSpec.GetLength(1);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("pred_semantic",
                    new DenseTensor<long>(predSemantic, new[] { 1, 1, predSemantic.Length })),
                NamedOnnxValue.CreateFromTensor("text_seq",
                    new DenseTensor<long>(textSeq, new[] { 1, textSeq.Length })),
                NamedOnnxValue.CreateFromTensor("refer_spec",
                    new DenseTensor<float>(Flatten(referSpec), new[] { 1, specDim, specLen })),
                NamedOnnxValue.CreateFromTensor("noise_scale",
                    new DenseTensor<float>(new[] { noiseScale }, new[] { 1 }))
            };

            // speed: 只要模型有这个输入就传 1.0，防止模型内部尝试进行不完整的变速
            if (_sovitsSession.InputMetadata.ContainsKey("speed"))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor("speed",
                    new DenseTensor<float>(new[] { 1.0f }, new[] { 1 })));
            }

            // sv_emb: 根据模型元数据判断是否需要传入
            if (_sovitsSession.InputMetadata.ContainsKey("sv_emb"))
            {
                const int svEmbDim = 20480; // V2ProPlus 的嵌入维度
                if (svEmb != null && svEmb.Length == svEmbDim)
                {
                    inputs.Add(NamedOnnxValue.CreateFromTensor("sv_emb",
                        new DenseTensor<float>(svEmb, new[] { 1, svEmb.Length })));
                }
                else
                {
                    // 如果没有提供 sv_emb，提供一个全零的占位符
                    DebugLog("⚠️ sv_emb 未提供或维度不匹配，使用全零占位符");
                    var zeroEmb = new float[svEmbDim];
                    inputs.Add(NamedOnnxValue.CreateFromTensor("sv_emb",
                        new DenseTensor<float>(zeroEmb, new[] { 1, svEmbDim })));
                }
            }

            using var results = _sovitsSession.Run(inputs);
            var audio = results.First().AsTensor<float>().ToArray();

            DebugLog($"SoVITS 输出: {audio.Length} 采样点");

            return audio;
        }

        // ============================================================
        // 辅助方法
        // ============================================================
        private int[] GetKvCacheShape()
        {
            // 使用从 gpt_step 模型动态读取的形状
            return _kvCacheShape;
        }

        private static float[] Flatten(float[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            float[] flat = new float[rows * cols];
            Buffer.BlockCopy(array, 0, flat, 0, flat.Length * sizeof(float));
            return flat;
        }

        /// <summary>
        /// 语义 Token 插值 (Nearest Neighbor)
        /// 通过在 Token 层级进行拉伸，实现不改变音调的变速
        /// </summary>
        private long[] StretchSemanticTokens(long[] tokens, float speed)
        {
            if (tokens == null || tokens.Length == 0) return tokens!;

            int newLen = (int)Math.Round(tokens.Length / speed);
            if (newLen < 1) newLen = 1;

            long[] result = new long[newLen];
            for (int i = 0; i < newLen; i++)
            {
                // 计算原索引，使用最近邻
                int oldIdx = (int)Math.Floor(i * speed);
                if (oldIdx >= tokens.Length) oldIdx = tokens.Length - 1;
                result[i] = tokens[oldIdx];
            }
            return result;
        }

        // ============================================================
        // Dispose
        // ============================================================
        public void Dispose()
        {
            _sslSession?.Dispose();
            _bertSession?.Dispose();
            _vqEncoderSession?.Dispose();
            _gptEncoderSession?.Dispose();
            _gptStepSession?.Dispose();
            _sovitsSession?.Dispose();
            _svSession?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}

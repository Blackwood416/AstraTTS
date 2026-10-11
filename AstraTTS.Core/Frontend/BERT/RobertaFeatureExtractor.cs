using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AstraTTS.Core.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AstraTTS.Core.Frontend.BERT
{
    public class RobertaFeatureExtractor : IDisposable
    {
        private readonly Tokenizers.HuggingFace.Tokenizer.Tokenizer _tokenizer;
        private readonly InferenceSession _session;
        private readonly object _inferenceLock = new object();
        private readonly BertFeatureCache? _cache;
        public bool DebugMode { get; set; } = false;

        /// <summary>
        /// 是否串行化推理。DirectML 要求同一 Session 不得并发 Run；
        /// V1 流式模式会为多句并行预取 BERT 特征，因此启用 DML 时必须串行。
        /// </summary>
        public bool SerializeInference { get; set; } = false;

        /// <summary>
        /// 是否启用特征缓存。开启后相同文本+对齐的重复调用直接返回缓存结果，
        /// 省去一次 BERT 前向（实测约 280–390 ms）。
        /// 缓存的数组为只读共享：调用方不得修改返回的 float[,]。
        /// </summary>
        public bool UseCache { get; set; } = true;

        public RobertaFeatureExtractor(string modelPath, string tokenizerJsonPath, SessionOptions? options = null, bool useCache = true)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException(modelPath);
            if (!File.Exists(tokenizerJsonPath)) throw new FileNotFoundException(tokenizerJsonPath);

            _tokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer.FromFile(tokenizerJsonPath);

            options ??= new SessionOptions();
            _session = new InferenceSession(modelPath, options);

            UseCache = useCache;
            _cache = useCache ? new BertFeatureCache() : null;
        }

        public float[,] Extract(string text, int[] word2ph)
        {
            // 0. 缓存查询（键含文本与对齐信息；命中直接跳过分词与推理）
            string? cacheKey = null;
            if (UseCache && _cache != null)
            {
                cacheKey = BertFeatureCache.MakeKey(text, word2ph);
                if (_cache.TryGet(cacheKey, out var cachedFeat))
                {
                    if (DebugMode)
                        Console.WriteLine($"[BERT Cache] 命中: '{text}' (rows={cachedFeat.GetLength(0)}), 累计 命中={_cache.Hits} 未命中={_cache.Misses}");
                    return cachedFeat;
                }
            }

            // 1. 分词 - 获取 Token IDs 和 Word 映射
            var encodings = _tokenizer.Encode(text, addSpecialTokens: true);
            var encoding = encodings.First();
            var ids = encoding.Ids.Select(x => (long)x).ToArray();
            var mask = ids.Select(_ => 1L).ToArray();

            // 2. 准备 Repeats 数组 (需要映射词与 Token)
            // 规则: word2ph 代表每个词对应的音素数。如果一个词被拆分为多个 Token，我们将音素数赋给第一个 Token，其余为 0。
            long[] fullRepeats = new long[ids.Length];
            int lastWordIdx = -1;

            if (encoding.Words != null && encoding.Words.Count == ids.Length)
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    uint wIdx = encoding.Words[i];
                    int wordIdx = (wIdx == uint.MaxValue) ? -1 : (int)wIdx;

                    if (wordIdx >= 0 && wordIdx < word2ph.Length)
                    {
                        if (wordIdx != lastWordIdx)
                        {
                            fullRepeats[i] = word2ph[wordIdx];
                            lastWordIdx = wordIdx;
                        }
                        else
                        {
                            fullRepeats[i] = 0; // 子词 Token 设为 0
                        }
                    }
                    else
                    {
                        fullRepeats[i] = 0; // 特殊 Token (如 [CLS], [SEP]) 设为 0
                    }
                }
            }
            else
            {
                // 兜底方案: 当无法获取 Word 映射时，尝试简单的一一对应
                // 假设结构为 [CLS] w1 w2 ... [SEP]
                for (int i = 0; i < Math.Min(ids.Length - 2, word2ph.Length); i++)
                {
                    fullRepeats[i + 1] = word2ph[i];
                }
            }

            // 3. 切片取出内容部分 (移除 [CLS] 和 [SEP] 的重复计数)
            // 该特定 ONNX 模型通过 Repeats 输入处理对齐，其内部逻辑会剥离特殊 Token。
            long[] inferenceRepeats = new long[Math.Max(0, ids.Length - 2)];
            if (ids.Length > 2)
            {
                Array.Copy(fullRepeats, 1, inferenceRepeats, 0, ids.Length - 2);
            }

            if (DebugMode)
            {
                Console.WriteLine($"[BERT Debug] Text: {text}");
                Console.WriteLine($"[BERT Debug] IDs: {string.Join(",", ids)}");
                Console.WriteLine($"[BERT Debug] Repeats (Content): {string.Join(",", inferenceRepeats)}");
            }

            // 4. ONNX 推理
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids, new[] { 1, ids.Length })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(mask, new[] { 1, mask.Length })),
                NamedOnnxValue.CreateFromTensor("repeats", new DenseTensor<long>(inferenceRepeats, new[] { inferenceRepeats.Length }))
            };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            if (SerializeInference)
            {
                // DirectML 同 Session 禁止并发 Run；多句流式预取会并发进入这里。
                lock (_inferenceLock)
                {
                    results = _session.Run(inputs);
                }
            }
            else
            {
                results = _session.Run(inputs);
            }
            using var _results = results;
            var outputValue = _results.First().Value;

            // 5. 解析输出维度并转换 (模型已内置对齐，直接读取即可)
            ReadOnlySpan<int> dims = outputValue switch
            {
                DenseTensor<Float16> f16 => f16.Dimensions,
                DenseTensor<float> f32 => f32.Dimensions,
                _ => throw new Exception($"Unsupported BERT output type: {outputValue?.GetType().FullName}")
            };

            int rows = dims[dims.Length - 2];
            int hidden = dims[dims.Length - 1];
            float[,] features = new float[rows, hidden];

            if (outputValue is DenseTensor<float> f32Tensor)
            {
                var flatData = f32Tensor.ToArray();
                Buffer.BlockCopy(flatData, 0, features, 0, flatData.Length * sizeof(float));
            }
            else if (outputValue is DenseTensor<Float16> f16Tensor)
            {
                var flatData = f16Tensor.ToArray();
                for (int i = 0; i < flatData.Length; i++)
                {
                    features[i / hidden, i % hidden] = (float)flatData[i];
                }
            }

            if (DebugMode)
            {
                float sum = 0, sumSq = 0;
                int total = features.Length;
                foreach (var f in features) { sum += f; sumSq += f * f; }
                float mean = total > 0 ? sum / total : 0;
                float std = total > 0 ? (float)Math.Sqrt(Math.Max(0, sumSq / total - mean * mean)) : 0;
                Console.WriteLine($"[BERT] Final Output Shape: ({rows}, {hidden}), mean={mean}, std={std}");
            }

            // 6. 写入缓存（幂等；高并发下同键偶发重复计算可接受）
            if (UseCache && _cache != null && cacheKey != null)
            {
                _cache.Set(cacheKey, features);
            }

            return features;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

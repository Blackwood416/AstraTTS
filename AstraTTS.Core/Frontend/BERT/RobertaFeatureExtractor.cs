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
        public bool DebugMode { get; set; } = false;

        public RobertaFeatureExtractor(string modelPath, string tokenizerJsonPath, SessionOptions? options = null)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException(modelPath);
            if (!File.Exists(tokenizerJsonPath)) throw new FileNotFoundException(tokenizerJsonPath);

            _tokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer.FromFile(tokenizerJsonPath);

            options ??= new SessionOptions();
            _session = new InferenceSession(modelPath, options);
        }

        public float[,] Extract(string text, int[] word2ph)
        {
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

            using var results = _session.Run(inputs);
            var outputValue = results.First().Value;

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

            return features;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

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

        public RobertaFeatureExtractor(string modelPath, string tokenizerJsonPath)
        {
            if (!File.Exists(modelPath)) throw new FileNotFoundException(modelPath);
            if (!File.Exists(tokenizerJsonPath)) throw new FileNotFoundException(tokenizerJsonPath);

            _tokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer.FromFile(tokenizerJsonPath);

            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _session = new InferenceSession(modelPath, options);
        }

        public float[,] Extract(string text, int[] word2ph)
        {
            // 1. 分词 - 使用默认行为 (addSpecialTokens: true)
            var encodings = _tokenizer.Encode(text, addSpecialTokens: true);
            var encoding = encodings.First();
            var ids = encoding.Ids.Select(x => (long)x).ToArray();
            var mask = ids.Select(_ => 1L).ToArray();
            var repeats = word2ph.Select(x => (long)x).ToArray();

            if (InferenceEngineV1.DebugMode)
            {
                Console.WriteLine($"[BERT Debug] Text: {text}");
                Console.WriteLine($"[BERT Debug] IDs: {string.Join(",", ids)}");
                Console.WriteLine($"[BERT Debug] Repeats: {string.Join(",", repeats)}");
                Console.WriteLine($"[BERT Debug] IDs Length: {ids.Length}, Repeats Length: {repeats.Length}");
            }

            // 2. 准备 Tensor
            var inputIdsTensor = new DenseTensor<long>(ids, new[] { 1, ids.Length });
            var attentionMaskTensor = new DenseTensor<long>(mask, new[] { 1, mask.Length });
            var repeatsTensor = new DenseTensor<long>(repeats, new[] { repeats.Length });

            // 3. 推理
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor("repeats", repeatsTensor)
            };

            using var results = _session.Run(inputs);
            var outputValue = results.First().Value;

            int seqLen, hidden;
            float[,] features;

            if (outputValue is DenseTensor<Float16> f16Tensor)
            {
                seqLen = f16Tensor.Dimensions[0];
                hidden = f16Tensor.Dimensions[1];
                features = new float[seqLen, hidden];
                // Float16 需要逐元素转换，无法用 Buffer.BlockCopy
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < hidden; j++)
                        features[i, j] = (float)f16Tensor[i, j];
            }
            else if (outputValue is DenseTensor<float> f32Tensor)
            {
                seqLen = f32Tensor.Dimensions[0];
                hidden = f32Tensor.Dimensions[1];
                features = new float[seqLen, hidden];

                // 优化: 使用 ToArray + Buffer.BlockCopy 替代嵌套循环
                var flatData = f32Tensor.ToArray();
                Buffer.BlockCopy(flatData, 0, features, 0, flatData.Length * sizeof(float));
            }
            else
            {
                throw new Exception($"Unsupported BERT output type: {outputValue?.GetType().FullName}");
            }

            // 统计信息仅在调试模式计算
            if (InferenceEngineV1.DebugMode)
            {
                float sum = 0;
                float sumSq = 0;
                int total = features.Length;
                foreach (var f in features)
                {
                    sum += f;
                    sumSq += f * f;
                }
                float mean = sum / total;
                float std = (float)Math.Sqrt(sumSq / total - mean * mean);

                Console.WriteLine($"[BERT] Output shape: ({seqLen}, {hidden}), magnitude: mean={mean}, std={std}");
                if (total > 0) Console.WriteLine($"[BERT] First value: {features[0, 0]}");
            }

            return features;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

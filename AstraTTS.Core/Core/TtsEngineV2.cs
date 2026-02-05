using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AstraTTS.Core.Config;
using AstraTTS.Core.Frontend.BERT;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.TextNorm;
using AstraTTS.Core.Utils;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// V2 推理引擎实现（基于 GPT-SoVITS Minimal Inference 架构）。
    /// </summary>
    public class TtsEngineV2 : ITtsEngine
    {
        private readonly InferenceEngineV2 _engine = new();
        private RobertaFeatureExtractor? _bert;
        private ChineseG2P? _chineseG2p;
        private EnglishG2P? _englishG2p;
        private MixedLanguageG2P? _mixedG2p;

        private long[] _refPhoneIds = Array.Empty<long>();
        private float[,] _refBert = new float[0, 0];
        private long[] _promptCodes = Array.Empty<long>();
        private float[,] _referSpec = new float[0, 0];
        private float[]? _svEmb;

        private TTSConfig? _config;

        public int SamplingRate => _engine.SamplingRate;

        public async Task LoadAsync(TTSConfig config)
        {
            _config = config;

            // 验证 V2 模型目录
            if (!Directory.Exists(config.ModelsV2Dir))
            {
                throw new DirectoryNotFoundException(
                    $"V2 模型目录不存在: '{config.ModelsV2Dir}'。" +
                    $"请确保 resources/models_v2 目录存在，或将 UseEngineV2 设置为 false 以使用 V1 引擎。");
            }

            await Task.Run(() =>
            {
                // 加载动态符号
                var modelConfigPath = Path.Combine(config.ModelsV2Dir, "config.json");
                if (File.Exists(modelConfigPath))
                {
                    var modelConfig = V2ModelConfig.Load(modelConfigPath);
                    if (modelConfig.SymbolToId != null)
                    {
                        Symbols.LoadDynamicSymbols(modelConfig.SymbolToId);
                    }
                }

                // 并行加载资源
                Parallel.Invoke(
                    () => _engine.LoadModels(config.ModelsV2Dir, config.UseDirectML),
                    () =>
                    {
                        if (File.Exists(config.BertModelPath) && File.Exists(config.TokenizerJsonPath))
                            _bert = new RobertaFeatureExtractor(config.BertModelPath, config.TokenizerJsonPath);
                    },
                    () => _chineseG2p = new ChineseG2P(config.ChineseG2PDict, config.PinyinDict, config.CustomDictFullPath),
                    () => _englishG2p = new EnglishG2P(config.CmuDict, config.NeuralG2PModel, config.CustomDictFullPath)
                );
            });

            if (_chineseG2p == null || _englishG2p == null)
                throw new Exception("G2P 核心初始化失败");

            _mixedG2p = new MixedLanguageG2P(_chineseG2p, _englishG2p);

            // 预处理参考音频
            await Task.Run(PrepareReference);
        }


        private void PrepareReference()
        {
            if (_config == null) return;

            // 优先使用 Avatar 系统
            string refPath;
            string refText;

            var avatarRef = _config.GetDefaultReferenceAudio();
            if (avatarRef.HasValue)
            {
                refPath = avatarRef.Value.audioPath;
                refText = avatarRef.Value.text;
            }
            else if (!string.IsNullOrEmpty(_config.RefAudioPath))
            {
                // 回退到旧版配置
                refPath = _config.RefAudioPath;
                refText = _config.RefText;
            }
            else
            {
                Console.WriteLine("警告: 未配置参考音频，请在 Avatars 中添加配置。");
                return;
            }

            if (!File.Exists(refPath))
            {
                Console.WriteLine($"警告: 参考音频文件不存在: {refPath}");
                return;
            }

            float[] audio16k = AudioHelper.ReadWav(refPath, 16000);
            float[] audio32k = AudioHelper.ReadWav(refPath, _engine.SamplingRate);

            // 提取特征
            var sslContent = _engine.ExtractSSL(audio16k);
            _promptCodes = _engine.EncodeVQ(sslContent);
            _referSpec = ComputeSpectrogram(audio32k, _engine.SamplingRate);

            // Pro 版本处理说话人嵌入
            if (_engine.ModelVersion.Contains("Pro", StringComparison.OrdinalIgnoreCase))
            {
                string npyPath = Path.Combine(_config.ModelsV2Dir, "sv_emb.npy");
                if (File.Exists(npyPath))
                {
                    _svEmb = LoadNpyFloat32(npyPath);
                }

                if (_svEmb == null)
                {
                    _svEmb = _engine.ExtractSpeakerEmbedding(audio16k);
                }
            }

            var res = _mixedG2p!.Process(refText);
            _refPhoneIds = Symbols.HasDynamicSymbols ? Symbols.GetIdsV2(res.Phones) : Symbols.GetIds(res.Phones);
            _refBert = ExtractBertFeature(_bert, res);
        }

        public async Task<float[]> PredictAsync(string text, TtsOptions options)
        {
            return await Task.Run(() =>
            {
                var (res, phoneIds, bertFeat) = ProcessFrontend(text);

                var semanticTokens = _engine.InferTokens(
                    _refPhoneIds, ConvertTo2D(_refBert),
                    phoneIds, ConvertTo2D(bertFeat),
                    _promptCodes,
                    topK: options.TopK,
                    temperature: options.Temperature
                );

                return _engine.RunSoVITS(
                    semanticTokens,
                    phoneIds,
                    _referSpec,
                    _svEmb,
                    options.NoiseScale,
                    options.Speed
                );
            });
        }

        public async IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions options, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var (res, phoneIds, bertFeat) = ProcessFrontend(text);

            var queue = new System.Collections.Concurrent.ConcurrentQueue<float[]>();
            bool isFinished = false;

            _ = Task.Run(() =>
            {
                try
                {
                    _engine.InferStream(
                        _refPhoneIds, ConvertTo2D(_refBert),
                        phoneIds, ConvertTo2D(bertFeat),
                        _promptCodes, _referSpec, _svEmb,
                        topK: options.TopK,
                        temperature: options.Temperature,
                        noiseScale: options.NoiseScale,
                        speed: options.Speed,
                        chunkSize: options.StreamingChunkTokens,
                        onAudioChunk: (chunk, isLast) =>
                        {
                            if (chunk.Length > 0) queue.Enqueue(chunk);
                            if (isLast) isFinished = true;
                        });
                }
                catch
                {
                    isFinished = true;
                }
            }, cancellationToken);

            while (!isFinished || !queue.IsEmpty)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (queue.TryDequeue(out var audio))
                {
                    yield return audio;
                }
                else
                {
                    await Task.Delay(10, cancellationToken);
                }
            }
        }

        private (G2PResult res, long[] phoneIds, float[,] bert) ProcessFrontend(string text)
        {
            string normalized = LanguageDetector.NormalizePunctuation(text);
            normalized = EnglishTextNormalizer.Normalize(normalized);

            var mode = LanguageDetector.DetectMode(normalized);
            G2PResult res;
            float[,] bertFeat;

            if (mode == LanguageDetector.LanguageMode.Chinese)
            {
                res = _chineseG2p!.Process(normalized);
                bertFeat = ExtractBertFeature(_bert, res);
            }
            else if (mode == LanguageDetector.LanguageMode.English)
            {
                res = _englishG2p!.Process(normalized);
                bertFeat = new float[res.PhoneIds.Length, 1024];
            }
            else
            {
                res = _mixedG2p!.Process(normalized);
                bertFeat = ExtractMixedBertFeature(_bert, res);
            }

            long[] phoneIds = Symbols.HasDynamicSymbols ? Symbols.GetIdsV2(res.Phones) : Symbols.GetIds(res.Phones);
            return (res, phoneIds, bertFeat);
        }

        #region Helpers (Migrated from Program_V2)

        private static float[,] ComputeSpectrogram(float[] audio, int sampleRate)
        {
            int fftSize = 2048;
            int hopLength = 640;
            int winLength = 2048;
            int numBins = fftSize / 2 + 1;
            int padSize = (fftSize - hopLength) / 2;
            var paddedAudio = new float[audio.Length + padSize * 2];

            for (int i = 0; i < padSize; i++)
            {
                paddedAudio[i] = audio[Math.Min(padSize - i, audio.Length - 1)];
                paddedAudio[paddedAudio.Length - 1 - i] = audio[Math.Max(0, audio.Length - padSize + i - 1)];
            }
            Array.Copy(audio, 0, paddedAudio, padSize, audio.Length);

            int numFrames = (paddedAudio.Length - fftSize) / hopLength + 1;
            if (numFrames <= 0) numFrames = 1;

            var spec = new float[numBins, numFrames];
            var window = new float[winLength];
            for (int i = 0; i < winLength; i++) window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / winLength));

            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * hopLength;
                var real = new double[fftSize];
                for (int i = 0; i < fftSize && start + i < paddedAudio.Length; i++)
                    real[i] = paddedAudio[start + i] * (i < winLength ? window[i] : 0);

                for (int k = 0; k < numBins; k++)
                {
                    double sumReal = 0, sumImag = 0;
                    for (int n = 0; n < fftSize; n++)
                    {
                        double angle = -2.0 * Math.PI * k * n / fftSize;
                        sumReal += real[n] * Math.Cos(angle);
                        sumImag += real[n] * Math.Sin(angle);
                    }
                    spec[k, frame] = (float)Math.Sqrt(sumReal * sumReal + sumImag * sumImag + 1e-8);
                }
            }
            return spec;
        }

        private static float[,] ExtractBertFeature(RobertaFeatureExtractor? bert, G2PResult res)
        {
            if (bert == null || res.Word2Ph.Length == 0) return new float[res.PhoneIds.Length, 1024];
            try
            {
                string text = res.NormalizedText;
                int[] w2p = res.Word2Ph;
                while (text.Length > 0 && ".。,，?？!！".Contains(text[^1]))
                {
                    text = text[..^1];
                    if (w2p.Length > 0) w2p = w2p[..^1];
                }
                if (text.Length > 0 && w2p.Length > 0)
                {
                    var feat = bert.Extract(text, w2p);
                    var final = new float[res.PhoneIds.Length, 1024];
                    Buffer.BlockCopy(feat, 0, final, 0, Math.Min(feat.GetLength(0), res.PhoneIds.Length) * 1024 * sizeof(float));
                    return final;
                }
            }
            catch { }
            return new float[res.PhoneIds.Length, 1024];
        }

        private static float[,] ExtractMixedBertFeature(RobertaFeatureExtractor? bert, G2PResult res)
        {
            var final = new float[res.PhoneIds.Length, 1024];
            if (bert == null || res.Segments == null) return final;
            foreach (var seg in res.Segments)
            {
                if (seg.Language == PhoneLanguage.Chinese && !string.IsNullOrWhiteSpace(seg.Text) && seg.Word2Ph.Length > 0)
                {
                    try
                    {
                        var feat = bert.Extract(seg.Text, seg.Word2Ph);
                        int rows = Math.Min(feat.GetLength(0), seg.PhoneCount);
                        rows = Math.Min(rows, res.PhoneIds.Length - seg.StartPhoneIndex);
                        if (rows > 0) Buffer.BlockCopy(feat, 0, final, seg.StartPhoneIndex * 1024 * sizeof(float), rows * 1024 * sizeof(float));
                    }
                    catch { }
                }
            }
            return final;
        }

        private static float[,] ConvertTo2D(float[,] bert)
        {
            int rows = bert.GetLength(0);
            int cols = bert.GetLength(1);
            var transposed = new float[cols, rows];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    transposed[j, i] = bert[i, j];
            return transposed;
        }

        private static float[]? LoadNpyFloat32(string path)
        {
            try
            {
                using var fs = File.OpenRead(path);
                using var br = new BinaryReader(fs);
                br.ReadBytes(6); // Magic
                byte major = br.ReadByte(); br.ReadByte(); // Version
                int headerLen = major == 1 ? br.ReadUInt16() : (int)br.ReadUInt32();
                var header = System.Text.Encoding.ASCII.GetString(br.ReadBytes(headerLen));
                int totalElements = 1;
                int shapeStart = header.IndexOf("'shape':");
                if (shapeStart >= 0)
                {
                    int pStart = header.IndexOf('(', shapeStart);
                    int pEnd = header.IndexOf(')', shapeStart);
                    if (pStart >= 0 && pEnd > pStart)
                    {
                        var dims = header.Substring(pStart + 1, pEnd - pStart - 1).Split(',', StringSplitOptions.RemoveEmptyEntries);
                        foreach (var dim in dims) if (int.TryParse(dim.Trim(), out int d)) totalElements *= d;
                    }
                }
                var data = new float[totalElements];
                for (int i = 0; i < totalElements; i++) data[i] = br.ReadSingle();
                return data;
            }
            catch { return null; }
        }

        #endregion

        public void Dispose()
        {
            _engine.Dispose();
            _bert?.Dispose();
        }
    }
}

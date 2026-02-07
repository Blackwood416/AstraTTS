using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using AstraTTS.Core.Config;
using AstraTTS.Core.Frontend.BERT;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.TextNorm;
using AstraTTS.Core.Utils;
using System.Buffers;
using System.Runtime.InteropServices;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// V1 推理引擎实现（基于 Genie-TTS 架构）。
    /// </summary>
    public class TtsEngineV1 : ITtsEngine
    {
        private readonly InferenceEngineV1 _engine = new();
        private RobertaFeatureExtractor? _bert;
        private ChineseG2P? _chineseG2p;
        private EnglishG2P? _englishG2p;
        private MixedLanguageG2P? _mixedG2p;

        private float[] _refAudio16k = Array.Empty<float>();
        private float[] _refAudio32k = Array.Empty<float>();
        private float[] _sslContent = Array.Empty<float>();
        private float[] _svEmb = Array.Empty<float>();
        private float[] _ge = Array.Empty<float>();
        private float[] _geAdvanced = Array.Empty<float>();

        private long[] _refPhoneIds = Array.Empty<long>();
        private float[,] _refBert = new float[0, 1024]; // 预设 1024 列避免 ONNX 维度异常

        private TTSConfig? _config;

        // 音色切换状态追踪
        private string? _currentAvatarId;
        private string? _currentReferenceId;
        private readonly object _referenceLock = new();

        public int SamplingRate => 32000;

        public async Task LoadAsync(TTSConfig config)
        {
            _config = config;

            // 验证 V1 模型目录
            if (!Directory.Exists(config.V1TtsDir))
            {
                throw new DirectoryNotFoundException(
                    $"V1 TTS 模型目录不存在: '{config.V1TtsDir}'。" +
                    $"请确保 resources/models_v1/tts 目录存在，或将 UseEngineV2 设置为 true 以使用 V2 引擎。");
            }

            // 异步并发加载资源
            await Task.Run(() =>
            {
                Parallel.Invoke(
                    () => _engine.LoadModels(config.V1TtsDir, config.HubertPath, config.SpeakerEncoderPath, config.UseDirectML),
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

            // 预处理参考音频 (可能包含模型推理，保持在 Task.Run 里或之后执行)
            await Task.Run(PrepareReference);
        }


        private void PrepareReference() => PrepareReference(null, null);

        private void PrepareReference(string? avatarId, string? referenceId)
        {
            if (_config == null) return;

            // 解析目标音色
            avatarId ??= _config.DefaultAvatarId;
            var avatar = _config.Avatars.Find(a => a.Id == avatarId);

            string refPath;
            string refText;

            if (avatar != null)
            {
                var reference = avatar.GetReference(referenceId);
                if (reference != null)
                {
                    refPath = reference.GetFullAudioPath(_config.AvatarsDir, avatarId);
                    refText = reference.Text;
                }
                else
                {
                    Console.WriteLine($"警告: 音色 '{avatarId}' 下找不到参考音频 '{referenceId}'。");
                    return;
                }
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
                throw new FileNotFoundException($"参考音频文件不存在: {refPath}。请检查 config.json 中的 Avatars 配置。");
            }

            _refAudio16k = AudioHelper.ReadWav(refPath, 16000);
            _refAudio32k = AudioHelper.ReadWav(refPath, 32000);

            _sslContent = _engine.GetHubertContent(_refAudio16k);
            _svEmb = _engine.GetSpeakerEmbedding(_refAudio16k);
            var (ge, geAdvanced) = _engine.GetPromptEmbedding(_refAudio32k, _svEmb);
            _ge = ge;
            _geAdvanced = geAdvanced;

            var res = _mixedG2p!.Process(refText);
            _refPhoneIds = res.PhoneIds;

            if (_bert != null)
            {
                string clean = res.NormalizedText.Replace("，", ",").Replace("。", ".");
                _refBert = _bert.Extract(clean, res.Word2Ph);
            }
            else
            {
                _refBert = new float[_refPhoneIds.Length, 1024];
            }

            if (_sslContent == null || _sslContent.Length == 0)
                throw new InvalidOperationException("无法提取参考音频的 SSL 特征。请确保资源目录中的模型文件 (hubert) 完整且音频可读。");

            if (_refPhoneIds == null || _refPhoneIds.Length == 0)
                throw new InvalidOperationException("无法处理参考音频的文本 (G2P 失败)。请检查 RefText 配置。");
        }

        public async Task<float[]> PredictAsync(string text, TtsOptions options)
        {
            // 检测音色切换
            EnsureReferenceLoaded(options.AvatarId, options.ReferenceId);

            var (res, bertFeat) = ProcessFrontendOptimal(text);
            try
            {
                return await Task.Run(() =>
                {
                    // 转换为 float[,] 以兼容旧引擎接口
                    var bert2D = new float[res.PhoneIds.Length, 1024];
                    Buffer.BlockCopy(bertFeat, 0, bert2D, 0, res.PhoneIds.Length * 1024 * sizeof(float));

                    var predSemantic = _engine.RunT2S(res.PhoneIds, bert2D, _refPhoneIds, _refBert, _sslContent);
                    var audio = _engine.RunVocoder(res.PhoneIds, predSemantic, _refAudio32k, out int audioLen, _ge, _geAdvanced, options.Speed);
                    try
                    {
                        float[] result = new float[audioLen];
                        Array.Copy(audio, 0, result, 0, audioLen);
                        return result;
                    }
                    finally
                    {
                        _engine.ReturnAudioBuffer(audio);
                    }
                });
            }
            finally
            {
                ArrayPool<float>.Shared.Return(bertFeat);
            }
        }

        /// <summary>
        /// 检查当前加载的音色是否与请求匹配，如不匹配则重新加载。
        /// </summary>
        private void EnsureReferenceLoaded(string? avatarId, string? referenceId)
        {
            if (_config == null) return;

            // 如果未指定，使用默认值
            avatarId ??= _config.DefaultAvatarId;

            // 查找 Avatar
            var avatar = _config.Avatars.Find(a => a.Id == avatarId);
            if (avatar == null)
            {
                Console.WriteLine($"警告: 找不到音色 '{avatarId}'，保持使用当前音色。");
                return;
            }

            // 查找 Reference
            var reference = avatar.GetReference(referenceId);
            string effectiveRefId = reference?.Id ?? "default";

            // 检查是否需要重新加载
            if (avatarId == _currentAvatarId && effectiveRefId == _currentReferenceId)
                return;

            lock (_referenceLock)
            {
                // 双重检查
                if (avatarId == _currentAvatarId && effectiveRefId == _currentReferenceId)
                    return;

                Console.WriteLine($"🔄 切换音色: {avatarId}/{effectiveRefId}");
                PrepareReference(avatarId, referenceId);
                _currentAvatarId = avatarId;
                _currentReferenceId = effectiveRefId;
            }
        }

        internal class SentenceContext
        {
            public string Text { get; set; } = string.Empty;
            public G2PResult? G2p { get; set; }
            public float[]? BertFeat { get; set; } // ArrayPool 租用的
            public Task? PreprocessTask { get; set; }
        }

        public async IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions options, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // 检测音色切换
            EnsureReferenceLoaded(options.AvatarId, options.ReferenceId);

            string normalized = LanguageDetector.NormalizePunctuation(text);
            normalized = EnglishTextNormalizer.Normalize(normalized);

            // 1. 句子切分
            var sentencesText = LanguageDetector.SplitSentences(normalized);
            var contexts = sentencesText.Select(s => new SentenceContext { Text = s }).ToList();

            // 2. 启动并行的前端预取 (BERT + G2P)
            foreach (var ctx in contexts)
            {
                ctx.PreprocessTask = Task.Run(() =>
                {
                    var (res, bertFeat) = ProcessFrontendOptimal(ctx.Text);
                    ctx.G2p = res;
                    ctx.BertFeat = bertFeat;
                }, cancellationToken);
            }

            // 3. 逐句驱动流水线：T2S -> Vocoder（指数间隔调用策略）
            var mainChannel = System.Threading.Channels.Channel.CreateUnbounded<float[]>();

            _ = Task.Run(async () =>
            {
                try
                {
                    foreach (var ctx in contexts)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        if (ctx.PreprocessTask != null) await ctx.PreprocessTask;
                        if (ctx.G2p == null || ctx.BertFeat == null) continue;

                        try
                        {
                            G2PResult res = ctx.G2p.Value;
                            float[] bertFeat = ctx.BertFeat!;

                            // 每句重置状态
                            int lastAudioLength = 0;
                            var tokenQueue = new System.Collections.Concurrent.ConcurrentQueue<(long[] tokens, bool isFinal)>();
                            bool t2sComplete = false;

                            // 指数间隔策略：初始阈值为 chunkSize，之后每次翻倍
                            int nextVocoderThreshold = options.StreamingChunkSize;

                            var bert2D = new float[res.PhoneIds.Length, 1024];
                            Buffer.BlockCopy(bertFeat, 0, bert2D, 0, res.PhoneIds.Length * 1024 * sizeof(float));

                            // Vocoder 线程
                            var innerVocoderTask = Task.Run(async () =>
                            {
                                try
                                {
                                    while (!t2sComplete || !tokenQueue.IsEmpty)
                                    {
                                        if (tokenQueue.TryDequeue(out var item))
                                        {
                                            var (tokens, isFinal) = item;
                                            float[] fullAudio = _engine.RunVocoder(res.PhoneIds, tokens, _refAudio32k, out int currentTotalLen, _ge, _geAdvanced, options.Speed);
                                            try
                                            {
                                                int newSamples = currentTotalLen - lastAudioLength;
                                                if (newSamples > 0)
                                                {
                                                    // 句子结束时追加静音尾部避免截断（200ms 容错缓冲）
                                                    int tailPadding = isFinal ? 6400 : 0;  // 200ms @ 32kHz
                                                    float[] chunk = new float[newSamples + tailPadding];
                                                    Array.Copy(fullAudio, lastAudioLength, chunk, 0, newSamples);
                                                    // tailPadding 部分保持为 0（静音）
                                                    lastAudioLength = currentTotalLen;
                                                    await mainChannel.Writer.WriteAsync(chunk, cancellationToken);
                                                }
                                            }
                                            finally
                                            {
                                                _engine.ReturnAudioBuffer(fullAudio);
                                            }
                                        }
                                        else
                                        {
                                            await Task.Delay(5, cancellationToken);
                                        }
                                    }
                                }
                                catch (Exception ex)
                                {
                                    mainChannel.Writer.TryComplete(ex);
                                }
                            }, cancellationToken);

                            // T2S 推理：使用指数间隔策略（1.5x 增长）
                            _engine.RunT2SStreamingTokens(
                                res.PhoneIds, bert2D,
                                _refPhoneIds, _refBert, _sslContent,
                                chunkSize: options.StreamingChunkSize,
                                onTokenChunk: (tokens, isFinal) =>
                                {
                                    // 指数间隔：只在 token 数达到阈值或 isFinal 时调用 Vocoder
                                    // 使用 1.5x 增长因子平滑卡顿
                                    if (isFinal || tokens.Length >= nextVocoderThreshold)
                                    {
                                        tokenQueue.Enqueue((tokens, isFinal));
                                        // 阈值增长 1.5x（比 2x 更平滑）
                                        nextVocoderThreshold = Math.Max((int)(nextVocoderThreshold * 1.5), tokens.Length + options.StreamingChunkSize);
                                    }
                                },
                                onRetry: () =>
                                {
                                    while (tokenQueue.TryDequeue(out _)) { }
                                    lastAudioLength = 0;
                                    nextVocoderThreshold = options.StreamingChunkSize;
                                });

                            t2sComplete = true;
                            await innerVocoderTask;
                        }
                        finally
                        {
                            if (ctx.BertFeat != null)
                            {
                                ArrayPool<float>.Shared.Return(ctx.BertFeat);
                                ctx.BertFeat = null;
                            }
                        }
                    }
                    mainChannel.Writer.Complete();
                }
                catch (Exception ex)
                {
                    mainChannel.Writer.TryComplete(ex);
                }
            }, cancellationToken);

            await foreach (var chunk in mainChannel.Reader.ReadAllAsync(cancellationToken))
            {
                yield return chunk;
            }
        }

        private (G2PResult result, float[] bert) ProcessFrontendOptimal(string text)
        {
            string normalized = LanguageDetector.NormalizePunctuation(text);
            normalized = EnglishTextNormalizer.Normalize(normalized);

            var mode = LanguageDetector.DetectMode(normalized);
            G2PResult res;
            if (mode == LanguageDetector.LanguageMode.Chinese)
                res = _chineseG2p!.Process(normalized);
            else if (mode == LanguageDetector.LanguageMode.English)
                res = _englishG2p!.Process(normalized);
            else
                res = _mixedG2p!.Process(normalized);

            var bertFeat = ExtractBertOptimal(res);
            return (res, bertFeat);
        }

        private float[] ExtractBertOptimal(G2PResult res)
        {
            var mode = LanguageDetector.DetectMode(res.NormalizedText);
            if (mode == LanguageDetector.LanguageMode.Chinese)
                return ExtractChineseBertOptimal(res);
            else if (mode == LanguageDetector.LanguageMode.English)
                return ArrayPool<float>.Shared.Rent(res.PhoneIds.Length * 1024);
            else
                return ExtractMixedBertOptimal(res);
        }

        private float[] ExtractChineseBertOptimal(G2PResult res)
        {
            float[] final = ArrayPool<float>.Shared.Rent(res.PhoneIds.Length * 1024);
            Array.Clear(final, 0, final.Length);

            if (_bert == null || res.Word2Ph.Length == 0) return final;

            string text = res.NormalizedText;
            int[] w2p = res.Word2Ph;

            while (text.Length > 0 && ".。,，?？!！".Contains(text[^1]))
            {
                text = text[..^1];
                if (w2p.Length > 0) w2p = w2p[..^1];
            }

            if (text.Length == 0 || w2p.Length == 0) return final;

            var feat = _bert.Extract(text, w2p);
            int rows = Math.Min(feat.GetLength(0), res.PhoneIds.Length);
            Buffer.BlockCopy(feat, 0, final, 0, rows * 1024 * sizeof(float));
            return final;
        }

        private float[] ExtractMixedBertOptimal(G2PResult res)
        {
            float[] final = ArrayPool<float>.Shared.Rent(res.PhoneIds.Length * 1024);
            Array.Clear(final, 0, final.Length);

            if (_bert == null || res.Segments == null) return final;

            foreach (var seg in res.Segments)
            {
                if (seg.Language == PhoneLanguage.Chinese && !string.IsNullOrWhiteSpace(seg.Text) && seg.Word2Ph.Length > 0)
                {
                    try
                    {
                        var feat = _bert.Extract(seg.Text, seg.Word2Ph);
                        int rows = Math.Min(feat.GetLength(0), seg.PhoneCount);
                        rows = Math.Min(rows, res.PhoneIds.Length - seg.StartPhoneIndex);
                        if (rows > 0)
                        {
                            int offset = seg.StartPhoneIndex * 1024 * sizeof(float);
                            Buffer.BlockCopy(feat, 0, final, offset * sizeof(float), rows * 1024 * sizeof(float));
                        }
                    }
                    catch { }
                }
            }
            return final;
        }

        private float[,] ExtractChineseBert(G2PResult res)
        {
            if (_bert == null || res.Word2Ph.Length == 0) return new float[res.PhoneIds.Length, 1024];

            string text = res.NormalizedText;
            int[] w2p = res.Word2Ph;

            while (text.Length > 0 && ".。,，?？!！".Contains(text[^1]))
            {
                text = text[..^1];
                if (w2p.Length > 0) w2p = w2p[..^1];
            }

            if (text.Length == 0 || w2p.Length == 0) return new float[res.PhoneIds.Length, 1024];

            var feat = _bert.Extract(text, w2p);
            var final = new float[res.PhoneIds.Length, 1024];
            int rows = Math.Min(feat.GetLength(0), res.PhoneIds.Length);
            Buffer.BlockCopy(feat, 0, final, 0, rows * 1024 * sizeof(float));
            return final;
        }

        private float[,] ExtractMixedBert(G2PResult res)
        {
            var final = new float[res.PhoneIds.Length, 1024];
            if (_bert == null || res.Segments == null) return final;

            foreach (var seg in res.Segments)
            {
                if (seg.Language == PhoneLanguage.Chinese && !string.IsNullOrWhiteSpace(seg.Text) && seg.Word2Ph.Length > 0)
                {
                    try
                    {
                        var feat = _bert.Extract(seg.Text, seg.Word2Ph);
                        int rows = Math.Min(feat.GetLength(0), seg.PhoneCount);
                        rows = Math.Min(rows, res.PhoneIds.Length - seg.StartPhoneIndex);
                        if (rows > 0)
                        {
                            int offset = seg.StartPhoneIndex * 1024 * sizeof(float);
                            Buffer.BlockCopy(feat, 0, final, offset, rows * 1024 * sizeof(float));
                        }
                    }
                    catch { }
                }
            }
            return final;
        }

        public void Dispose()
        {
            _engine.Dispose();
            _bert?.Dispose();
        }
    }
}

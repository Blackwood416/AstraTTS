using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using AstraTTS.Core.Config;
using AstraTTS.Core.Frontend.BERT;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.G2P.Common;
using AstraTTS.Core.Frontend.G2P.Chinese;
using AstraTTS.Core.Frontend.G2P.English;
using AstraTTS.Core.Frontend.G2P.Japanese;
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
        private JapaneseG2P? _japaneseG2p;
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
                var opt = _engine.GetSessionOptions(config);
                Parallel.Invoke(
                    () => _engine.LoadModels(config.V1TtsDir, config, config.HubertPath, config.SpeakerEncoderPath),
                    () =>
                    {
                        if (File.Exists(config.BertModelPath) && File.Exists(config.TokenizerJsonPath))
                            _bert = new RobertaFeatureExtractor(config.BertModelPath, config.TokenizerJsonPath, opt);
                    },
                    () => _chineseG2p = new ChineseG2P(config.ChineseG2PDict, config.PinyinDict, config.CustomDictFullPath ?? "", config.PolyphonicJson, config.JiebaDictDir),
                    () => _englishG2p = new EnglishG2P(config.CmuDict, config.NeuralG2PModel, config.CustomDictFullPath,
                        config.EnglishPosTaggerDir, config.EnglishWordSegmentDir, config.EnglishSpecialDict, config.G2P.PriorityMode),
                    () =>
                    {
                        // 日语 G2P (可选 - 词典目录存在时加载)
                        if (Directory.Exists(config.JapaneseDictDir))
                        {
                            _japaneseG2p = new JapaneseG2P(config.JapaneseDictDir, config.CustomDictFullPath);
                        }
                        else
                        {
                            Console.WriteLine($"[TtsEngineV1] Warning: Japanese dictionary not found at: {config.JapaneseDictDir}");
                        }
                    }
                );
            });

            if (_chineseG2p == null || _englishG2p == null)
                throw new Exception("G2P 核心初始化失败");

            _englishG2p.DebugMode = config.DebugMode;
            if (_japaneseG2p != null) _japaneseG2p.DebugMode = config.DebugMode;
            if (_bert != null) _bert.DebugMode = config.DebugMode;

            _mixedG2p = new MixedLanguageG2P(_chineseG2p, _englishG2p, _japaneseG2p, _config.G2P.Languages);
            _mixedG2p.DebugMode = config.DebugMode;

            // 预处理参考音频 (可能包含模型推理，保持在 Task.Run 里或之后执行)
            await Task.Run(PrepareReference);
        }

        /// <summary>
        /// 从另一个引擎实例共享核心资源（用于池化中的内存复用模式）
        /// </summary>
        public void ShareResourcesFrom(TtsEngineV1 other)
        {
            _engine.ShareModelsFrom(other._engine);
            _bert = other._bert;
            _chineseG2p = other._chineseG2p;
            _englishG2p = other._englishG2p;
            _japaneseG2p = other._japaneseG2p;
            _mixedG2p = other._mixedG2p;
            _config = other._config;
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
            string? refLanguage;

            if (avatar != null)
            {
                var reference = avatar.GetReference(referenceId);
                if (reference != null)
                {
                    refPath = reference.GetFullAudioPath(_config.AvatarsDir, avatarId);
                    refText = reference.Text;
                    refLanguage = reference.Language;
                }
                else
                {
                    Console.WriteLine($"警告: 音色 '{avatarId}' 下找不到参考音频 '{referenceId}'。");
                    return;
                }
            }
            else
            {
                Console.WriteLine("警告: 未配置参考音频，请在 Avatars 中添加配置。");
                return;
            }

            // 自动补全语言标签
            if (string.IsNullOrEmpty(refLanguage))
            {
                bool hasJaHints = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) || c == 0x30FC || c == 0x300C || c == 0x300D || c == 0x30FB || c == 0x3005);
                if (hasJaHints) refLanguage = "ja";
                else
                {
                    var mode = LanguageDetector.DetectMode(refText);
                    if (mode == LanguageDetector.LanguageMode.Japanese) refLanguage = "ja";
                }
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

            // 使用参考音频配置的语言进行 G2P 处理
            // 如果参考音频的语种不在全局允许列表中，创建临时 G2P 以纳入该语种
            MixedLanguageG2P refG2p = _mixedG2p!;
            if (!string.IsNullOrEmpty(refLanguage))
            {
                var refLangs = new List<string>(_config.G2P.Languages ?? new List<string> { "zh", "en" });
                var lower = refLanguage.ToLowerInvariant();
                if (!refLangs.Contains(lower) && (lower == "ja" || lower == "jp"))
                {
                    refLangs.Add("ja");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
                else if (!refLangs.Contains(lower) && lower == "en")
                {
                    refLangs.Add("en");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
            }
            else
            {
                // 参考音频未设置语种 → 自动检测：如果文本包含假名，临时允许日语
                bool hasKana = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF));
                if (hasKana)
                {
                    refLanguage = "ja"; // 自动推断并赋值，用于后续 BERT 判断
                    var refLangs = new List<string>(_config.G2P.Languages ?? new List<string> { "zh", "en" });
                    if (!refLangs.Contains("ja")) refLangs.Add("ja");
                    refG2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, refLangs);
                    refG2p.DebugMode = _config.DebugMode;
                }
            }
            var res = refG2p.Process(refText, refLanguage);
            _refPhoneIds = res.PhoneIds;

            if (_bert != null)
            {
                // 使用统一的 BERT 提取逻辑，确保参考音频与合成文本行为一致
                float[] pooledBert = ExtractBertOptimal(res);
                try
                {
                    // 将辅助 1D 数组转换为持久化的 2D 数组
                    _refBert = new float[res.PhoneIds.Length, 1024];
                    Buffer.BlockCopy(pooledBert, 0, _refBert, 0, res.PhoneIds.Length * 1024 * sizeof(float));
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(pooledBert);
                }
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

        public async Task<TtsResult> PredictAsync(string text, TtsOptions options)
        {
            // 检测音色切换
            EnsureReferenceLoaded(options.AvatarId, options.ReferenceId);

            // 1. 识别全文语言倾向，为分句处理提供全局上下文
            var overallMode = LanguageDetector.DetectMode(text);
            string? globalLangHint = (overallMode == LanguageDetector.LanguageMode.Japanese) ? "ja" : null;

            // 2. 切分句子并逐句处理
            string normalized = LanguageDetector.NormalizePunctuation(text);
            var sentences = LanguageDetector.SplitSentences(normalized);
            if (sentences.Count == 0) return new TtsResult { Audio = Array.Empty<float>() };

            var allAudio = new List<float>();
            int totalTokens = 0;

            foreach (var s in sentences)
            {
                // 3. 前端处理 (G2P + BERT) - 传入全局语言暗示
                var (res, bertFeat) = ProcessFrontendOptimal(s, options, globalLangHint);
                try
                {
                    // 转换为 float[,] 以兼容旧引擎接口
                    var bert2D = new float[res.PhoneIds.Length, 1024];
                    Buffer.BlockCopy(bertFeat, 0, bert2D, 0, res.PhoneIds.Length * 1024 * sizeof(float));

                    var predSemantic = _engine.RunT2S(res.PhoneIds, bert2D, _refPhoneIds, _refBert, _sslContent, res.LanguageTags, options);
                    var audio = _engine.RunVocoder(res.PhoneIds, predSemantic, _refAudio32k, out int audioLen, _ge, _geAdvanced, options.Speed);
                    try
                    {
                        float[] audioData = new float[audioLen];
                        Array.Copy(audio, 0, audioData, 0, audioLen);
                        allAudio.AddRange(audioData);
                        totalTokens += predSemantic.Length;
                    }
                    finally
                    {
                        _engine.ReturnAudioBuffer(audio);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(bertFeat);
                }
            }

            return new TtsResult
            {
                Audio = allAudio.ToArray(),
                TokenCount = totalTokens
            };
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
            string refText = reference?.Text ?? "";
            string? refLanguage = reference?.Language;

            // 如果未显式指定语言，尝试根据文本特征判定（解决纯汉字日语参考文本识别为中文的问题）
            if (string.IsNullOrEmpty(refLanguage))
            {
                bool hasJaHints = refText.Any(c => (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) || c == 0x30FC || c == 0x300C || c == 0x300D || c == 0x3005);
                if (hasJaHints) refLanguage = "ja";
                else
                {
                    var mode = LanguageDetector.DetectMode(refText);
                    if (mode == LanguageDetector.LanguageMode.Japanese) refLanguage = "ja";
                }
            }

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

            // 1. 句子切分
            var sentencesText = LanguageDetector.SplitSentences(normalized);
            var contexts = sentencesText.Select(s => new SentenceContext { Text = s }).ToList();

            // 2. 启动并行的前端预取 (BERT + G2P)
            var overallMode = LanguageDetector.DetectMode(normalized);
            string? globalLangHint = (overallMode == LanguageDetector.LanguageMode.Japanese) ? "ja" : null;

            foreach (var ctx in contexts)
            {
                ctx.PreprocessTask = Task.Run(() =>
                {
                    var (res, bertFeat) = ProcessFrontendOptimal(ctx.Text, options, globalLangHint);
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
                                                // 精确跳过逻辑：
                                                // 如果当前生成的总长度 currentTotalLen 还没有超过之前已发送的水位线 lastAudioLength，
                                                // 说明这是重试后正在生成的重复片段，直接跳过。
                                                int newSamplesToPush = currentTotalLen - lastAudioLength;
                                                if (newSamplesToPush > 0)
                                                {
                                                    // 句子结束时追加静音尾部避免截断（200ms 容错缓冲）
                                                    int tailPadding = isFinal ? 6400 : 0;  // 200ms @ 32kHz
                                                    float[] chunk = new float[newSamplesToPush + tailPadding];

                                                    // 从 fullAudio 的末尾截取 newSamplesToPush 个点，
                                                    // 相当于跳过了前面的音频，只取出水位线之后的新采样点。
                                                    Array.Copy(fullAudio, currentTotalLen - newSamplesToPush, chunk, 0, newSamplesToPush);

                                                    // 更新水位线
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
                                     // 核心修复：重试时不重置 lastAudioLength 水位线。
                                     // 之前发送给用户的音频已经不可撤回，新尝试产生的重复部分应在推送前被跳过。
                                     while (tokenQueue.TryDequeue(out _)) { }
                                     nextVocoderThreshold = options.StreamingChunkSize;
                                     if (_config?.DebugMode == true) Console.WriteLine("[TtsEngineV1] T2S 重试触发，水位线维持。");
                                 },
                                languageTags: res.LanguageTags,
                                options: options); // 透传选项

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

        private (G2PResult res, float[] bertFeat) ProcessFrontendOptimal(string text, TtsOptions options, string? explicitLanguage = null)
        {
            // 仅进行标点符号归一化，有助于后续语种识别
            string normalized = LanguageDetector.NormalizePunctuation(text);

            // 统一使用 MixedLanguageG2P 处理，它会自动进行语种分割和路由
            // 如果 Options 中指定了允许的语种列表，则为本次请求创建临时的 MixedLanguageG2P
            MixedLanguageG2P g2p = _mixedG2p!;
            if (options.Languages != null && options.Languages.Count > 0)
            {
                g2p = new MixedLanguageG2P(_chineseG2p!, _englishG2p!, _japaneseG2p, options.Languages);
                g2p.DebugMode = _config?.DebugMode ?? false;
            }

            var res = g2p.Process(normalized, explicitLanguage, options.G2PPriorityMode);

            var bertFeat = ExtractBertOptimal(res);
            return (res, bertFeat);
        }

        private float[] ExtractBertOptimal(G2PResult res)
        {
            // 统一调用混合语种 BERT 提取逻辑
            // 该逻辑会根据 G2P 的分段信息 (PhoneLanguage) 决定哪些段落需要提取 BERT
            // 这样能确保日语汉字不会被误判为中文进行 BERT 提取（只要 G2P 将其标为日语）
            return ExtractMixedBertOptimal(res);
        }


        private float[] ExtractMixedBertOptimal(G2PResult res)
        {
            float[] final = ArrayPool<float>.Shared.Rent(res.PhoneIds.Length * 1024);
            Array.Clear(final, 0, final.Length);

            if (_bert == null || res.Segments == null) return final;

            foreach (var seg in res.Segments)
            {
                bool isSupported = seg.Language == PhoneLanguage.Chinese;
                if (isSupported && !string.IsNullOrWhiteSpace(seg.Text) && seg.Word2Ph.Length > 0)
                {
                    try
                    {
                        // 对齐修复: BERT 分词器忽略空格，但 ChineseG2P 为空格生成了 SP 音素 (word2ph=1)
                        // 需要剔除空格对应的 word2ph 条目，并去除文本中的空格，以保证维度一致
                        string bertText = seg.Text;
                        int[] bertWord2Ph = seg.Word2Ph;

                        // 逐字符匹配: word2ph 与归一化文本的字符一一对应
                        // 空格字符对应的 word2ph 条目需要移除
                        var cleanChars = new List<char>();
                        var cleanW2P = new List<int>();
                        int charIdx = 0;
                        for (int w = 0; w < bertWord2Ph.Length && charIdx < bertText.Length; w++)
                        {
                            char c = bertText[charIdx];
                            if (char.IsWhiteSpace(c))
                            {
                                // 跳过空格及其 word2ph 条目
                                charIdx++;
                                continue;
                            }
                            cleanChars.Add(c);
                            cleanW2P.Add(bertWord2Ph[w]);
                            charIdx++;
                        }

                        bertText = new string(cleanChars.ToArray());
                        bertWord2Ph = cleanW2P.ToArray();

                        if (bertWord2Ph.Length > 0 && !string.IsNullOrEmpty(bertText))
                        {
                            var feat = _bert.Extract(bertText, bertWord2Ph);
                            int rows = Math.Min(feat.GetLength(0), seg.PhoneCount);
                            // 防御性检查
                            rows = Math.Min(rows, res.PhoneIds.Length - seg.StartPhoneIndex);

                            if (rows > 0)
                            {
                                int offset = seg.StartPhoneIndex * 1024;
                                Buffer.BlockCopy(feat, 0, final, offset * sizeof(float), rows * 1024 * sizeof(float));
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        if (_config?.DebugMode == true)
                            Console.WriteLine($"[TtsEngineV1] Mixed BERT extraction failed for segment: {ex.Message}");
                    }
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

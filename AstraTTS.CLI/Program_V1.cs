using System.Diagnostics;
using System.Text;
using System.Buffers;
using System.Runtime.InteropServices;
using AstraTTS.Core.Config;
using AstraTTS.Core.Core;
using AstraTTS.Core.Frontend.BERT;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.TextNorm;
using AstraTTS.Core.Utils;

namespace AstraTTS.Core
{
    /// <summary>
    /// 旧版程序入口 (基于 Genie-TTS 模型)
    /// 保留供参考，使用 InferenceEngine (V1)
    /// </summary>
    public class Program_V1
    {
        // 预分配静音缓冲区 (200ms @ 32kHz Mono)，避免重复分配
        private static readonly byte[] _silenceBuffer = new byte[6400 * 2];

        public static void Run(TTSConfig config)
        {
            Console.WriteLine("=== AstraTTS.Core V1 (Genie-TTS Inference) ===");

            // 3. Initialize Inference Engine
            // 3. Initialize Variables
            InferenceEngineV1.DebugMode = true;
            var stepSw = new Stopwatch();

            Console.WriteLine($"[Config] Model Dir: {config.V1TtsDir}");
            Console.WriteLine($"[Config] Ref Audio: {config.RefAudioPath}");

            try
            {
                // Initialize Inference Engine
                using var engine = new InferenceEngineV1();

                Console.WriteLine("1. 初始化引擎 & 加载模型...");
                stepSw.Restart();

                // Parallel Loading
                RobertaFeatureExtractor? bert = null;
                ChineseG2P? chineseG2p = null;
                EnglishG2P? englishG2p = null;

                Parallel.Invoke(
                    () => engine.LoadModels(config.V1TtsDir, config.HubertPath, config.SpeakerEncoderPath, config.UseDirectML),
                    () =>
                    {
                        if (File.Exists(config.BertModelPath) && File.Exists(config.TokenizerJsonPath))
                            bert = new RobertaFeatureExtractor(config.BertModelPath, config.TokenizerJsonPath);
                    },
                    () => chineseG2p = new ChineseG2P(config.ChineseG2PDict, config.PinyinDict),
                    () => englishG2p = new EnglishG2P(config.CmuDict, config.NeuralG2PModel)
                );

                // Create MixedLanguageG2P router
                if (chineseG2p == null) throw new Exception("ChineseG2P init failed");
                if (englishG2p == null) throw new Exception("EnglishG2P init failed");
                var g2p = new MixedLanguageG2P(chineseG2p, englishG2p);

                stepSw.Stop();
                Console.WriteLine($"   ✅ 模型加载完成! 耗时: {stepSw.ElapsedMilliseconds} ms");

                if (bert == null) Console.WriteLine("⚠️ BERT 模型未找到，效果可能受限.");

                // 2. 预处理参考音频 (只做一次)
                Console.WriteLine("2. 预处理参考音频...");
                stepSw.Restart();
                float[] audio16k, audio32k;
                string refAudioPath = config.RefAudioPath;
                string refAudio16kPath = Path.Combine(Path.GetDirectoryName(refAudioPath)!, Path.GetFileNameWithoutExtension(refAudioPath) + "_16k.wav");
                string refAudio32kPath = Path.Combine(Path.GetDirectoryName(refAudioPath)!, Path.GetFileNameWithoutExtension(refAudioPath) + "_32k.wav");

                if (File.Exists(refAudio16kPath) && File.Exists(refAudio32kPath))
                {
                    audio16k = AudioHelper.ReadWav(refAudio16kPath, 16000);
                    audio32k = AudioHelper.ReadWav(refAudio32kPath, 32000);
                }
                else
                {
                    audio16k = AudioHelper.ReadWav(refAudioPath, 16000);
                    audio32k = AudioHelper.ReadWav(refAudioPath, 32000);
                }

                // Pre-calculate Reference Embeddings
                var sslContent = engine.GetHubertContent(audio16k);
                var svEmb = engine.GetSpeakerEmbedding(audio16k);
                var (ge, geAdvanced) = engine.GetPromptEmbedding(audio32k, svEmb);

                // Pre-process Reference Text
                //string refText = "在此之前，请您务必继续享受旅居拉古那的时光。"; 
                string refText = "良宵方始，不必心急。";
                var refRes = g2p.Process(refText);
                float[,] refBert;
                if (bert != null)
                {
                    string refClean = refRes.NormalizedText.Replace("，", ",").Replace("。", ".");
                    refBert = bert.Extract(refClean, refRes.Word2Ph);
                }
                else
                {
                    refBert = new float[refRes.PhoneIds.Length, 1024];
                }
                stepSw.Stop();
                Console.WriteLine($"   ✅ 参考音频预处理完成! 耗时: {stepSw.ElapsedMilliseconds} ms");
                Console.WriteLine($"   当前参考文本: {refText}");

                // 3. 交互循环
                Console.WriteLine("\n🚀 引擎已就绪! 请输入文本开始合成 (输入 'q' 或 'exit' 退出)");
                Console.WriteLine("---------------------------------------------------------------");

                while (true)
                {
                    Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write("Input > ");
                    Console.ResetColor();

                    string? input = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(input)) continue;
                    if (input.Trim().ToLower() == "q" || input.Trim().ToLower() == "exit") break;

                    try
                    {
                        var loopSw = Stopwatch.StartNew();

                        // === 三模式 G2P 处理 ===
                        // 0. 标点规范化 (全角→半角)
                        string normalizedInput = LanguageDetector.NormalizePunctuation(input);

                        // 0.5 英文特殊符号规范化 (C# → C sharp, .NET → dot net 等)
                        // 必须在语言检测前执行，否则 ".NET" 会被错误拆分
                        normalizedInput = EnglishTextNormalizer.Normalize(normalizedInput);

                        // 1. 检测语言模式
                        var mode = LanguageDetector.DetectMode(normalizedInput);

                        G2PResult textRes;
                        float[,] textBert;

                        if (mode == LanguageDetector.LanguageMode.Chinese)
                        {
                            // === 中文模式 ===
                            if (InferenceEngineV1.DebugMode)
                                Console.WriteLine($"[Mode] Chinese: {normalizedInput}");

                            textRes = chineseG2p.Process(normalizedInput);

                            // 中文使用 BERT
                            if (bert != null && textRes.Word2Ph.Length > 0)
                            {
                                string bertText = textRes.NormalizedText;
                                int[] bertWord2Ph = textRes.Word2Ph;

                                // 移除句末标点，同时调整 Word2Ph
                                while (bertText.Length > 0 && ".。,，?？!！".Contains(bertText[^1]))
                                {
                                    bertText = bertText[..^1];
                                    if (bertWord2Ph.Length > 0)
                                    {
                                        bertWord2Ph = bertWord2Ph[..^1];
                                    }
                                }

                                // 确保处理后仍有内容
                                if (bertText.Length > 0 && bertWord2Ph.Length > 0)
                                {
                                    var bertFeatures = bert.Extract(bertText, bertWord2Ph);
                                    int bertRows = bertFeatures.GetLength(0);
                                    textBert = new float[textRes.PhoneIds.Length, 1024];
                                    int bytesToCopy = Math.Min(bertRows, textRes.PhoneIds.Length) * 1024 * sizeof(float);
                                    Buffer.BlockCopy(bertFeatures, 0, textBert, 0, bytesToCopy);
                                }
                                else
                                {
                                    textBert = new float[textRes.PhoneIds.Length, 1024];
                                }
                            }
                            else
                            {
                                textBert = new float[textRes.PhoneIds.Length, 1024];
                            }
                        }
                        else if (mode == LanguageDetector.LanguageMode.English)
                        {
                            // === 英文模式 ===
                            if (InferenceEngineV1.DebugMode)
                                Console.WriteLine($"[Mode] English: {normalizedInput}");

                            textRes = englishG2p.Process(normalizedInput);

                            // 英文使用零向量
                            textBert = new float[textRes.PhoneIds.Length, 1024];
                        }
                        else
                        {
                            // === 混合模式 ===
                            if (InferenceEngineV1.DebugMode)
                                Console.WriteLine($"[Mode] Mixed: {normalizedInput}");

                            textRes = g2p.Process(normalizedInput);
                            textBert = new float[textRes.PhoneIds.Length, 1024];

                            // 分段 BERT 处理
                            if (bert != null && textRes.Segments != null)
                            {
                                foreach (var segment in textRes.Segments)
                                {
                                    if (segment.Language == PhoneLanguage.Chinese &&
                                        !string.IsNullOrWhiteSpace(segment.Text) &&
                                        segment.Word2Ph.Length > 0)
                                    {
                                        try
                                        {
                                            var bertFeatures = bert.Extract(segment.Text, segment.Word2Ph);
                                            int bertRows = bertFeatures.GetLength(0);
                                            int rowsToCopy = Math.Min(bertRows, segment.PhoneCount);
                                            rowsToCopy = Math.Min(rowsToCopy, textRes.PhoneIds.Length - segment.StartPhoneIndex);

                                            if (rowsToCopy > 0)
                                            {
                                                int destOffset = segment.StartPhoneIndex * 1024 * sizeof(float);
                                                int bytesToCopy = rowsToCopy * 1024 * sizeof(float);
                                                Buffer.BlockCopy(bertFeatures, 0, textBert, destOffset, bytesToCopy);
                                            }

                                            if (InferenceEngineV1.DebugMode)
                                                Console.WriteLine($"[BERT] Chinese segment: '{segment.Text}'");
                                        }
                                        catch { /* 失败时保持零向量 */ }
                                    }
                                }
                            }
                        }

                        // B. Inference
                        bool useStreaming = config.StreamingMode;
                        if (useStreaming)
                        {
                            // === 异步流式推理：T2S 和 Vocoder 并行 ===
                            Console.WriteLine($"   🎵 异步流式推理 (WASAPI {(config.WasapiExclusiveMode ? "Exclusive" : "Shared")} + Lock-Free Buffer)...");

                            var ttsFormat = new NAudio.Wave.WaveFormat(32000, 16, 1);

                            // 使用无锁环形缓冲区替代 BufferedWaveProvider
                            var lockFreeProvider = new LockFreeWaveProvider(ttsFormat, config.LockFreeBufferSize)
                            {
                                ReadFully = true // 关键：Buffer空时填充静音，防止 WasapiOut 停止播放
                            };

                            // 音频源：根据模式决定是否重采样
                            NAudio.Wave.IWaveProvider audioSource;
                            NAudio.Wave.MediaFoundationResampler? resampler = null;
                            double drainRatio; // 用于 Smart Drain 计算

                            if (config.WasapiExclusiveMode)
                            {
                                // Exclusive Mode: 必须重采样到硬件支持的格式
                                var targetFormat = new NAudio.Wave.WaveFormat(48000, 16, 2);
                                resampler = new NAudio.Wave.MediaFoundationResampler(lockFreeProvider, targetFormat)
                                {
                                    ResamplerQuality = 1 // Linear (Fastest)
                                };
                                audioSource = resampler;
                                drainRatio = 3.0; // 32k Mono -> 48k Stereo = 3x bytes
                            }
                            else
                            {
                                // Shared Mode: 直接输出原始格式，Windows 音频引擎自动处理
                                audioSource = lockFreeProvider;
                                drainRatio = 1.0; // 无重采样，1:1 字节比
                            }

                            // 配置 Audio Backend
                            NAudio.Wave.WasapiOut waveOut;
                            if (config.WasapiExclusiveMode)
                            {
                                waveOut = new NAudio.Wave.WasapiOut(NAudio.CoreAudioApi.AudioClientShareMode.Exclusive, 50);
                            }
                            else
                            {
                                waveOut = new NAudio.Wave.WasapiOut(NAudio.CoreAudioApi.AudioClientShareMode.Shared, 50);
                            }

                            waveOut.Init(audioSource);

                            // Token 队列：T2S 生成 tokens，Vocoder 线程消费
                            var tokenQueue = new System.Collections.Concurrent.ConcurrentQueue<(long[] tokens, bool isFinal)>();
                            int chunkCount = 0;
                            int lastAudioLength = 0;
                            long totalInputBytes = 0; // 追踪总产生的音频字节数 (32kHz 16bit Mono)
                            bool t2sComplete = false;
                            bool playbackStarted = false;
                            // 增加预缓冲块数以抵抗生成抖动 (Latency vs Stability)
                            int PreBufferChunks = config.StreamingPreBufferChunks;

                            // Cross-fade 参数
                            const int crossFadeLen = 640;  // ~20ms @ 32kHz
                            float[]? prevChunkTail = null;  // 保存上一块的尾部用于 cross-fade

                            // High-Pass Filter (DC Offset Removal)
                            var hpFilter = new HighPassFilter(32000, 20); // 20Hz cutoff

                            // Vocoder 线程：从 token 队列取 tokens，调用 vocoder，添加到音频缓冲区
                            var vocoderThread = new System.Threading.Thread(() =>
                            {
                                while (!t2sComplete || !tokenQueue.IsEmpty)
                                {
                                    if (tokenQueue.TryDequeue(out var item))
                                    {
                                        var (tokens, isFinal) = item;

                                        // 调用 Vocoder (在独立线程，不阻塞 T2S)
                                        // 关键修复：传递 config.Speed 语速参数
                                        var audio = engine.RunVocoder(textRes.PhoneIds, tokens, audio32k, out int currentAudioLen, ge, geAdvanced, config.Speed);

                                        // 计算新增的音频
                                        int newSamples = currentAudioLen - lastAudioLength;
                                        if (newSamples > 0)
                                        {
                                            // Rent buffers from ArrayPool to avoid GC pressure
                                            float[] newAudioBuf = ArrayPool<float>.Shared.Rent(newSamples);
                                            byte[]? pcmBuf = null;

                                            try
                                            {
                                                // Copy data manually (avoid LINQ Skip/Take)
                                                Array.Copy(audio, lastAudioLength, newAudioBuf, 0, newSamples);

                                                // Manually run HP filter on the valid range
                                                hpFilter.Process(newAudioBuf.AsSpan(0, newSamples));

                                                // === Cross-fade 处理 ===
                                                int effectiveLen = newSamples;

                                                if (prevChunkTail != null && effectiveLen > crossFadeLen)
                                                {
                                                    for (int f = 0; f < crossFadeLen; f++)
                                                    {
                                                        float alpha = (float)f / crossFadeLen;
                                                        newAudioBuf[f] = prevChunkTail[f] * (1f - alpha) + newAudioBuf[f] * alpha;
                                                    }
                                                }

                                                // 保存当前块的尾部用于下次 cross-fade
                                                if (!isFinal && effectiveLen > crossFadeLen)
                                                {
                                                    prevChunkTail = new float[crossFadeLen];
                                                    Array.Copy(newAudioBuf, effectiveLen - crossFadeLen, prevChunkTail, 0, crossFadeLen);

                                                    effectiveLen -= crossFadeLen; // Hide tail from output
                                                }
                                                else
                                                {
                                                    // 最后一块：应用淡出
                                                    if (isFinal && effectiveLen > crossFadeLen)
                                                    {
                                                        int start = effectiveLen - crossFadeLen;
                                                        for (int f = 0; f < crossFadeLen; f++)
                                                        {
                                                            float fadeOut = 1f - (float)f / crossFadeLen;
                                                            newAudioBuf[start + f] *= fadeOut;
                                                        }
                                                    }
                                                    prevChunkTail = null;
                                                }

                                                // 转换为 16-bit PCM
                                                pcmBuf = ArrayPool<byte>.Shared.Rent(effectiveLen * 2);
                                                var pcmShorts = MemoryMarshal.Cast<byte, short>(pcmBuf.AsSpan(0, effectiveLen * 2));

                                                for (int i = 0; i < effectiveLen; i++)
                                                {
                                                    pcmShorts[i] = (short)(Math.Clamp(newAudioBuf[i], -1f, 1f) * 32767);
                                                }

                                                lockFreeProvider.AddSamples(pcmBuf, 0, effectiveLen * 2);
                                                totalInputBytes += effectiveLen * 2;

                                                if (isFinal)
                                                {
                                                    lockFreeProvider.AddSamples(_silenceBuffer, 0, _silenceBuffer.Length);
                                                    totalInputBytes += _silenceBuffer.Length;
                                                }

                                                chunkCount++;
                                                double bufSeconds = lockFreeProvider.BufferedBytes / 32000.0 / 2.0;

                                                if (!playbackStarted && (chunkCount >= PreBufferChunks || isFinal))
                                                {
                                                    waveOut.Play();
                                                    playbackStarted = true;
                                                }

                                                if (chunkCount % 5 == 0 || isFinal)
                                                {
                                                    Console.Write($"\r   🔊 Chunk {chunkCount}: +{effectiveLen / 32000.0:F2}s (buf: {bufSeconds:F1}s)  ");
                                                }
                                            }
                                            finally
                                            {
                                                ArrayPool<float>.Shared.Return(newAudioBuf);
                                                if (pcmBuf != null) ArrayPool<byte>.Shared.Return(pcmBuf);
                                            }
                                        }
                                        lastAudioLength = currentAudioLen; // 使用物长度
                                    }
                                    else
                                    {
                                        Thread.Sleep(5);
                                    }

                                }
                            });
                            vocoderThread.Priority = ThreadPriority.AboveNormal;
                            vocoderThread.Start();

                            // T2S 主线程：生成 tokens，入队
                            engine.RunT2SStreamingTokens(
                                textRes.PhoneIds, textBert,
                                refRes.PhoneIds, refBert, sslContent,
                                chunkSize: config.StreamingChunkSize,
                                onTokenChunk: (tokens, isFinal) =>
                                {
                                    tokenQueue.Enqueue((tokens.ToArray(), isFinal));
                                },
                                onRetry: () =>
                                {
                                    while (tokenQueue.TryDequeue(out _)) { }
                                    lastAudioLength = 0;
                                    chunkCount = 0;
                                    prevChunkTail = null;
                                    Console.WriteLine("\n   ⚠️ 生成不完整，正在重试...");
                                });

                            t2sComplete = true;
                            vocoderThread.Join();

                            while (lockFreeProvider.BufferedBytes > 0)
                            {
                                Thread.Sleep(100);
                            }

                            long totalValidSourceBytes = totalInputBytes + lockFreeProvider.PaddingBytes;
                            long expectedOutputBytes = (long)(totalValidSourceBytes * drainRatio);
                            long startWait = loopSw.ElapsedMilliseconds;

                            while (waveOut.GetPosition() < expectedOutputBytes)
                            {
                                Thread.Sleep(50);
                                if (loopSw.ElapsedMilliseconds - startWait > 5000) break;
                            }

                            Thread.Sleep(100);
                            waveOut.Stop();
                            waveOut.Dispose();

                            loopSw.Stop();
                            Console.WriteLine();
                            Console.WriteLine($"   ✅ 异步流式完成 | Chunks: {chunkCount} | Padding: {lockFreeProvider.PaddingBytes}B | ⏱️ Time: {loopSw.ElapsedMilliseconds}ms");
                        }
                        else
                        {
                            // === 常规推理模式 ===
                            // 补全语速参数
                            var predSemantic = engine.RunT2S(textRes.PhoneIds, textBert, refRes.PhoneIds, refBert, sslContent);
                            var audioOutFull = engine.RunVocoder(textRes.PhoneIds, predSemantic, audio32k, out int audioLenFinal, ge, geAdvanced, config.Speed);
                            float[] audioOut = new float[audioLenFinal];
                            Array.Copy(audioOutFull, 0, audioOut, 0, audioLenFinal);
                            engine.ReturnAudioBuffer(audioOutFull);

                            loopSw.Stop();

                            // C. Save
                            string fileName = $"output_{DateTime.Now:HHmmss}.wav";
                            AudioHelper.SaveWav(fileName, audioOut, 32000);

                            double rtf = (double)loopSw.ElapsedMilliseconds / ((double)audioOut.Length / 32000 * 1000);
                            Console.WriteLine($"   ✅ Saved: {fileName} | ⏱️ Time: {loopSw.ElapsedMilliseconds}ms | RTF: {rtf:F2}x");
                        }
                    }
                    catch (Exception loopEx)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"   ❌ Error: {loopEx.Message}");
                        Console.ResetColor();
                    }
                }

                bert?.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FATAL ERROR: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
    }
}

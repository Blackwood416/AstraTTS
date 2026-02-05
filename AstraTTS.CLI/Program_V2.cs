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

namespace AstraTTS.CLI
{
    public class Program_V2
    {
        // 预分配静音缓冲区 (200ms @ 32kHz Mono)
        private static readonly byte[] _silenceBuffer = new byte[6400 * 2];

        public static void Run(TTSConfig config)
        {
            Console.WriteLine("=== AstraTTS.Core V2 (GPT-SoVITS Minimal Inference) ===");
            Console.WriteLine();

            Console.WriteLine($"[Config] Model Dir V2: {config.ModelsV2Dir}");
            Console.WriteLine($"[Config] Speed: {config.Speed}, NoiseScale: {config.NoiseScale}");
            Console.WriteLine($"[Config] TopK: {config.TopK}, Temperature: {config.Temperature}");

            try
            {
                // 2. Initialize InferenceEngineV2
                InferenceEngineV2.DebugMode = true;
                var stepSw = new Stopwatch();

                Console.WriteLine("\n1. 初始化引擎 & 加载模型...");
                stepSw.Restart();

                using var engine = new InferenceEngineV2();

                // 加载 G2P
                ChineseG2P? chineseG2p = null;
                EnglishG2P? englishG2p = null;
                RobertaFeatureExtractor? bert = null;

                Parallel.Invoke(
                    () => engine.LoadModels(config.ModelsV2Dir, config.UseDirectML),
                    () => chineseG2p = new ChineseG2P(config.ChineseG2PDict, config.PinyinDict),
                    () => englishG2p = new EnglishG2P(config.CmuDict, config.NeuralG2PModel),
                    () =>
                    {
                        if (File.Exists(config.BertModelPath) && File.Exists(config.TokenizerJsonPath))
                            bert = new RobertaFeatureExtractor(config.BertModelPath, config.TokenizerJsonPath);
                    }
                );

                if (chineseG2p == null) throw new Exception("ChineseG2P init failed");
                if (englishG2p == null) throw new Exception("EnglishG2P init failed");
                var g2p = new MixedLanguageG2P(chineseG2p, englishG2p);

                // 加载动态符号表 (如果 config.json 中存在)
                var modelConfigPath = Path.Combine(config.ModelsV2Dir, "config.json");
                if (File.Exists(modelConfigPath))
                {
                    var modelConfig = V2ModelConfig.Load(modelConfigPath);
                    if (modelConfig.SymbolToId != null)
                    {
                        Symbols.LoadDynamicSymbols(modelConfig.SymbolToId);
                    }
                }

                stepSw.Stop();
                Console.WriteLine($"   ✅ 模型加载完成! 耗时: {stepSw.ElapsedMilliseconds} ms");
                Console.WriteLine($"   模型版本: {engine.ModelVersion}, 采样率: {engine.SamplingRate}Hz");

                if (bert == null) Console.WriteLine("   ⚠️ BERT 模型未找到，效果可能受限.");

                // 3. 预处理参考音频
                Console.WriteLine("\n2. 预处理参考音频...");
                stepSw.Restart();

                string refAudioPath = config.RefAudioPath;
                float[] audio16k = AudioHelper.ReadWav(refAudioPath, 16000);
                float[] audio32k = AudioHelper.ReadWav(refAudioPath, engine.SamplingRate);

                // 提取参考音频特征
                var sslContent = engine.ExtractSSL(audio16k);
                var promptCodes = engine.EncodeVQ(sslContent);

                // 提取参考频谱 (用于 SoVITS)
                var referSpec = ComputeSpectrogram(audio32k, engine.SamplingRate);

                // 提取说话人嵌入 (V2ProPlus 需要)
                float[]? svEmb = null;
                if (engine.ModelVersion.Contains("Pro", StringComparison.OrdinalIgnoreCase))
                {
                    // 优先从预计算文件加载
                    string svEmbPath = Path.Combine(config.ModelsV2Dir, "sv_emb.npy");
                    if (File.Exists(svEmbPath))
                    {
                        Console.WriteLine($"   加载预计算说话人嵌入: {svEmbPath}");
                        svEmb = LoadNpyFloat32(svEmbPath);
                        if (svEmb != null)
                        {
                            Console.WriteLine($"   ✅ 说话人嵌入: {svEmb.Length} 维 (从 .npy 加载)");
                        }
                    }

                    // 如果没有预计算文件，尝试用 SV 模型提取
                    if (svEmb == null)
                    {
                        Console.WriteLine("   提取说话人嵌入 (ERes2NetV2)...");
                        svEmb = engine.ExtractSpeakerEmbedding(audio16k);
                        if (svEmb != null)
                        {
                            Console.WriteLine($"   ✅ 说话人嵌入: {svEmb.Length} 维");
                        }
                        else
                        {
                            Console.WriteLine("   ⚠️ 说话人嵌入提取失败，音色可能受影响");
                        }
                    }
                }

                // 预处理参考文本
                string refText = config.RefText;
                var refRes = g2p.Process(refText);
                float[,] refBert = ExtractBertFeature(bert, refRes);

                // 提取参考音素 ID (用于拼接，与 Python 实现匹配)
                long[] refPhonemeIds = Symbols.HasDynamicSymbols
                    ? Symbols.GetIdsV2(refRes.Phones)
                    : Symbols.GetIds(refRes.Phones);
                Console.WriteLine($"   参考音素: {refPhonemeIds.Length} 个");

                stepSw.Stop();
                Console.WriteLine($"   ✅ 参考音频预处理完成! 耗时: {stepSw.ElapsedMilliseconds} ms");
                Console.WriteLine($"   参考文本: {refText}");
                Console.WriteLine($"   Prompt Codes: {promptCodes.Length} tokens");

                // 4. 交互循环
                Console.WriteLine("\n🚀 引擎已就绪! 请输入文本开始合成 (输入 'q' 或 'exit' 退出)");
                Console.WriteLine("---------------------------------------------------------------");

                while (true)
                {
                    Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write("Input > ");
                    Console.ResetColor();

                    string? input = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(input)) continue;
                    if (input.Trim().ToLower() is "q" or "exit") break;

                    try
                    {
                        var loopSw = Stopwatch.StartNew();

                        // 文本预处理
                        string normalizedInput = LanguageDetector.NormalizePunctuation(input);
                        normalizedInput = EnglishTextNormalizer.Normalize(normalizedInput);
                        var mode = LanguageDetector.DetectMode(normalizedInput);

                        G2PResult textRes;
                        float[,] textBert;

                        if (mode == LanguageDetector.LanguageMode.Chinese)
                        {
                            Console.WriteLine($"   [Mode] Chinese");
                            textRes = chineseG2p.Process(normalizedInput);
                            textBert = ExtractBertFeature(bert, textRes);
                        }
                        else if (mode == LanguageDetector.LanguageMode.English)
                        {
                            Console.WriteLine($"   [Mode] English");
                            textRes = englishG2p.Process(normalizedInput);
                            textBert = new float[textRes.PhoneIds.Length, 1024];
                        }
                        else
                        {
                            Console.WriteLine($"   [Mode] Mixed");
                            textRes = g2p.Process(normalizedInput);
                            textBert = ExtractMixedBertFeature(bert, textRes);
                        }

                        Console.WriteLine($"   Phones: {string.Join(" ", textRes.Phones.Take(20))}...");

                        // 获取音素 ID (使用动态符号表)
                        long[] phonemeIds = Symbols.HasDynamicSymbols
                            ? Symbols.GetIdsV2(textRes.Phones)
                            : Symbols.GetIds(textRes.Phones);

                        // 推理
                        if (config.StreamingMode)
                        {
                            // === 异步流式推理 ===
                            Console.WriteLine($"   🎵 异步流式推理 (WASAPI {(config.WasapiExclusiveMode ? "Exclusive" : "Shared")} + Lock-Free Buffer)...");

                            var ttsFormat = new NAudio.Wave.WaveFormat(engine.SamplingRate, 16, 1);
                            var lockFreeProvider = new LockFreeWaveProvider(ttsFormat, config.LockFreeBufferSize)
                            {
                                ReadFully = true
                            };

                            NAudio.Wave.IWaveProvider audioSource;
                            NAudio.Wave.MediaFoundationResampler? resampler = null;
                            double drainRatio;

                            if (config.WasapiExclusiveMode)
                            {
                                var targetFormat = new NAudio.Wave.WaveFormat(48000, 16, 2);
                                resampler = new NAudio.Wave.MediaFoundationResampler(lockFreeProvider, targetFormat) { ResamplerQuality = 1 };
                                audioSource = resampler;
                                drainRatio = (48000.0 * 2 * 2) / (engine.SamplingRate * 1 * 2);
                            }
                            else
                            {
                                audioSource = lockFreeProvider;
                                drainRatio = 1.0;
                            }

                            NAudio.Wave.WasapiOut waveOut = config.WasapiExclusiveMode
                                ? new NAudio.Wave.WasapiOut(NAudio.CoreAudioApi.AudioClientShareMode.Exclusive, 50)
                                : new NAudio.Wave.WasapiOut(NAudio.CoreAudioApi.AudioClientShareMode.Shared, 50);

                            waveOut.Init(audioSource);

                            var audioChunks = new List<float[]>();
                            int chunkCount = 0;
                            bool playbackStarted = false;
                            long totalInputBytes = 0;

                            // Cross-fade 参数
                            const int crossFadeLen = 640;
                            float[]? prevChunkTail = null;
                            var hpFilter = new HighPassFilter(engine.SamplingRate, 20);

                            engine.InferStream(
                                refPhonemeIds, ConvertTo2D(refBert),
                                phonemeIds, ConvertTo2D(textBert),
                                promptCodes, referSpec, svEmb,
                                topK: config.TopK,
                                temperature: config.Temperature,
                                noiseScale: config.NoiseScale,
                                speed: config.Speed,
                                chunkSize: config.StreamingChunkTokens,
                                onAudioChunk: (chunk, isFinal) =>
                                {
                                    if (chunk.Length == 0 && !isFinal) return;

                                    chunkCount++;
                                    audioChunks.Add(chunk);

                                    if (chunk.Length > 0)
                                    {
                                        // 滤波与淡入淡出处理
                                        float[] processed = ArrayPool<float>.Shared.Rent(chunk.Length);
                                        try
                                        {
                                            Array.Copy(chunk, processed, chunk.Length);
                                            hpFilter.Process(processed.AsSpan(0, chunk.Length));

                                            int effectiveLen = chunk.Length;
                                            if (prevChunkTail != null && effectiveLen > crossFadeLen)
                                            {
                                                for (int f = 0; f < crossFadeLen; f++)
                                                {
                                                    float alpha = (float)f / crossFadeLen;
                                                    processed[f] = prevChunkTail[f] * (1f - alpha) + processed[f] * alpha;
                                                }
                                            }

                                            if (!isFinal && effectiveLen > crossFadeLen)
                                            {
                                                prevChunkTail = new float[crossFadeLen];
                                                Array.Copy(processed, effectiveLen - crossFadeLen, prevChunkTail, 0, crossFadeLen);
                                                effectiveLen -= crossFadeLen;
                                            }
                                            else
                                            {
                                                if (isFinal && effectiveLen > crossFadeLen)
                                                {
                                                    int start = effectiveLen - crossFadeLen;
                                                    for (int f = 0; f < crossFadeLen; f++) processed[start + f] *= (1f - (float)f / crossFadeLen);
                                                }
                                                prevChunkTail = null;
                                            }

                                            // 转换为 PCM
                                            byte[] pcmBuf = ArrayPool<byte>.Shared.Rent(effectiveLen * 2);
                                            try
                                            {
                                                var pcmShorts = MemoryMarshal.Cast<byte, short>(pcmBuf.AsSpan(0, effectiveLen * 2));
                                                for (int i = 0; i < effectiveLen; i++)
                                                    pcmShorts[i] = (short)(Math.Clamp(processed[i], -1f, 1f) * 32767);

                                                lockFreeProvider.AddSamples(pcmBuf, 0, effectiveLen * 2);
                                                totalInputBytes += effectiveLen * 2;
                                            }
                                            finally { ArrayPool<byte>.Shared.Return(pcmBuf); }
                                        }
                                        finally { ArrayPool<float>.Shared.Return(processed); }
                                    }

                                    if (isFinal)
                                    {
                                        lockFreeProvider.AddSamples(_silenceBuffer, 0, _silenceBuffer.Length);
                                        totalInputBytes += _silenceBuffer.Length;
                                    }

                                    if (!playbackStarted && (chunkCount >= config.StreamingPreBufferChunks || isFinal))
                                    {
                                        waveOut.Play();
                                        playbackStarted = true;
                                    }

                                    Console.Write($"\r   🔊 Chunk {chunkCount}: {chunk.Length / (float)engine.SamplingRate:F2}s");
                                    if (isFinal) Console.WriteLine(" [FINAL]");
                                }
                            );

                            // 等待播放
                            while (lockFreeProvider.BufferedBytes > 0) Thread.Sleep(100);

                            long expectedPos = (long)((totalInputBytes + lockFreeProvider.PaddingBytes) * drainRatio);
                            long startWait = loopSw.ElapsedMilliseconds;
                            while (waveOut.GetPosition() < expectedPos && loopSw.ElapsedMilliseconds - startWait < 5000) Thread.Sleep(50);

                            Thread.Sleep(100);
                            waveOut.Stop();
                            waveOut.Dispose();

                            var fullAudio = MergeAudioChunks(audioChunks);
                            loopSw.Stop();

                            string fileName = $"output_v2_{DateTime.Now:HHmmss}.wav";
                            AudioHelper.SaveWav(fileName, fullAudio, engine.SamplingRate);
                            double rtf = (double)loopSw.ElapsedMilliseconds / (fullAudio.Length / (double)engine.SamplingRate * 1000);
                            Console.WriteLine($"   ✅ Saved: {fileName} | ⏱️ {loopSw.ElapsedMilliseconds}ms | RTF: {rtf:F2}x");
                        }
                        else
                        {
                            // === 非流式推理 ===
                            Console.WriteLine($"   🎵 非流式推理...");

                            // 生成 semantic tokens
                            var semanticTokens = engine.InferTokens(
                                refPhonemeIds,
                                ConvertTo2D(refBert),
                                phonemeIds,
                                ConvertTo2D(textBert),
                                promptCodes,
                                topK: config.TopK,
                                temperature: config.Temperature
                            );

                            Console.WriteLine($"   Generated {semanticTokens.Length} semantic tokens");

                            // 运行 SoVITS
                            var audioOut = engine.RunSoVITS(
                                semanticTokens,
                                phonemeIds,
                                referSpec,
                                svEmb,
                                config.NoiseScale,
                                config.Speed
                            );

                            loopSw.Stop();

                            // 保存
                            string fileName = $"output_v2_{DateTime.Now:HHmmss}.wav";
                            AudioHelper.SaveWav(fileName, audioOut, engine.SamplingRate);

                            double rtf = (double)loopSw.ElapsedMilliseconds / ((double)audioOut.Length / engine.SamplingRate * 1000);
                            Console.WriteLine($"   ✅ Saved: {fileName} | ⏱️ {loopSw.ElapsedMilliseconds}ms | RTF: {rtf:F2}x");
                        }
                    }
                    catch (Exception loopEx)
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.WriteLine($"   ❌ Error: {loopEx.Message}");
                        if (InferenceEngineV2.DebugMode)
                            Console.WriteLine(loopEx.StackTrace);
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

        // ============================================================
        // 辅助方法
        // ============================================================

        /// <summary>
        /// 计算频谱图 (用于 SoVITS refer_spec)
        /// 实现与 Python spectrogram_torch 一致
        /// </summary>
        private static float[,] ComputeSpectrogram(float[] audio, int sampleRate)
        {
            int fftSize = 2048;
            int hopLength = 640;
            int winLength = 2048;
            int numBins = fftSize / 2 + 1;  // 1025

            // Reflect padding (Python: F.pad(y, (int((n_fft - hop_size) / 2), ...), mode="reflect"))
            int padSize = (fftSize - hopLength) / 2;
            var paddedAudio = new float[audio.Length + padSize * 2];

            // 反射填充
            for (int i = 0; i < padSize; i++)
            {
                paddedAudio[i] = audio[Math.Min(padSize - i, audio.Length - 1)];
                paddedAudio[paddedAudio.Length - 1 - i] = audio[Math.Max(0, audio.Length - padSize + i - 1)];
            }
            Array.Copy(audio, 0, paddedAudio, padSize, audio.Length);

            int numFrames = (paddedAudio.Length - fftSize) / hopLength + 1;
            if (numFrames <= 0) numFrames = 1;

            var spec = new float[numBins, numFrames];

            // 预计算 Hann 窗
            var window = new float[winLength];
            for (int i = 0; i < winLength; i++)
            {
                window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / winLength));
            }

            // STFT 计算
            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * hopLength;

                // 应用窗函数并准备 FFT 输入
                var real = new double[fftSize];
                var imag = new double[fftSize];

                for (int i = 0; i < fftSize && start + i < paddedAudio.Length; i++)
                {
                    real[i] = paddedAudio[start + i] * (i < winLength ? window[i] : 0);
                }

                // DFT (简化版，对于实时应用可以用 FFT 库替代)
                for (int k = 0; k < numBins; k++)
                {
                    double sumReal = 0, sumImag = 0;
                    for (int n = 0; n < fftSize; n++)
                    {
                        double angle = -2.0 * Math.PI * k * n / fftSize;
                        sumReal += real[n] * Math.Cos(angle);
                        sumImag += real[n] * Math.Sin(angle);
                    }
                    // 计算幅度 (Python: torch.sqrt(spec.pow(2).sum(-1) + 1e-8))
                    spec[k, frame] = (float)Math.Sqrt(sumReal * sumReal + sumImag * sumImag + 1e-8);
                }
            }

            Console.WriteLine($"   频谱计算完成: [{numBins}, {numFrames}]");
            return spec;
        }

        /// <summary>
        /// 提取 BERT 特征
        /// </summary>
        private static float[,] ExtractBertFeature(RobertaFeatureExtractor? bert, G2PResult res)
        {
            if (bert == null || res.Word2Ph.Length == 0)
            {
                return new float[res.PhoneIds.Length, 1024];
            }

            try
            {
                string bertText = res.NormalizedText;
                int[] bertWord2Ph = res.Word2Ph;

                // 移除句末标点
                while (bertText.Length > 0 && ".。,，?？!！".Contains(bertText[^1]))
                {
                    bertText = bertText[..^1];
                    if (bertWord2Ph.Length > 0)
                        bertWord2Ph = bertWord2Ph[..^1];
                }

                if (bertText.Length > 0 && bertWord2Ph.Length > 0)
                {
                    var bertFeatures = bert.Extract(bertText, bertWord2Ph);
                    int bertRows = bertFeatures.GetLength(0);
                    var textBert = new float[res.PhoneIds.Length, 1024];
                    int bytesToCopy = Math.Min(bertRows, res.PhoneIds.Length) * 1024 * sizeof(float);
                    Buffer.BlockCopy(bertFeatures, 0, textBert, 0, bytesToCopy);
                    return textBert;
                }
            }
            catch { }

            return new float[res.PhoneIds.Length, 1024];
        }

        /// <summary>
        /// 提取混合语言 BERT 特征
        /// </summary>
        private static float[,] ExtractMixedBertFeature(RobertaFeatureExtractor? bert, G2PResult res)
        {
            var textBert = new float[res.PhoneIds.Length, 1024];

            if (bert == null || res.Segments == null)
                return textBert;

            foreach (var segment in res.Segments)
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
                        rowsToCopy = Math.Min(rowsToCopy, res.PhoneIds.Length - segment.StartPhoneIndex);

                        if (rowsToCopy > 0)
                        {
                            int destOffset = segment.StartPhoneIndex * 1024 * sizeof(float);
                            int bytesToCopy = rowsToCopy * 1024 * sizeof(float);
                            Buffer.BlockCopy(bertFeatures, 0, textBert, destOffset, bytesToCopy);
                        }
                    }
                    catch { }
                }
            }

            return textBert;
        }

        /// <summary>
        /// 将一维 BERT 特征转换为二维
        /// </summary>
        private static float[,] ConvertTo2D(float[,] bert)
        {
            // 已经是二维，直接返回
            // 这里是为了与 InferenceEngineV2 的接口匹配
            // InferenceEngineV2 期望 [1024, T] 而不是 [T, 1024]
            int rows = bert.GetLength(0);
            int cols = bert.GetLength(1);

            var transposed = new float[cols, rows];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[j, i] = bert[i, j];
                }
            }
            return transposed;
        }

        /// <summary>
        /// 合并音频块
        /// </summary>
        private static float[] MergeAudioChunks(List<float[]> chunks)
        {
            int totalLength = chunks.Sum(c => c.Length);
            var merged = new float[totalLength];
            int offset = 0;

            foreach (var chunk in chunks)
            {
                Array.Copy(chunk, 0, merged, offset, chunk.Length);
                offset += chunk.Length;
            }

            return merged;
        }

        /// <summary>
        /// 读取 NumPy .npy 文件 (仅支持 float32)
        /// </summary>
        private static float[]? LoadNpyFloat32(string path)
        {
            try
            {
                using var fs = File.OpenRead(path);
                using var br = new BinaryReader(fs);

                // 读取魔数 (0x93NUMPY)
                var magic = br.ReadBytes(6);
                if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
                    magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
                {
                    Console.WriteLine("   ⚠️ 无效的 .npy 文件格式");
                    return null;
                }

                // 版本号
                byte majorVersion = br.ReadByte();
                byte minorVersion = br.ReadByte();

                // 头部长度
                int headerLen;
                if (majorVersion == 1)
                {
                    headerLen = br.ReadUInt16();
                }
                else
                {
                    headerLen = (int)br.ReadUInt32();
                }

                // 读取头部字典
                var headerBytes = br.ReadBytes(headerLen);
                var header = System.Text.Encoding.ASCII.GetString(headerBytes);

                // 解析形状 (简化版，假设是一维或二维 float32)
                // 例如: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 20480), }
                int totalElements = 1;
                int shapeStart = header.IndexOf("'shape':");
                if (shapeStart >= 0)
                {
                    int parenStart = header.IndexOf('(', shapeStart);
                    int parenEnd = header.IndexOf(')', shapeStart);
                    if (parenStart >= 0 && parenEnd > parenStart)
                    {
                        var shapeStr = header.Substring(parenStart + 1, parenEnd - parenStart - 1);
                        var dims = shapeStr.Split(',', StringSplitOptions.RemoveEmptyEntries);
                        foreach (var dim in dims)
                        {
                            if (int.TryParse(dim.Trim(), out int d))
                                totalElements *= d;
                        }
                    }
                }

                // 读取数据
                var data = new float[totalElements];
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = br.ReadSingle();
                }

                return data;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ 读取 .npy 文件失败: {ex.Message}");
                return null;
            }
        }
    }
}

using AstraTTS.Core.Core;
using AstraTTS.Core.Config;
using AstraTTS.Core.Utils;
using System.Diagnostics;
using System.Buffers;
using System.Runtime.Versioning;
using NAudio.Wave;

namespace AstraTTS.CLI
{
    class Program
    {
        private static string? _currentAvatarId;
        private static string? _currentReferenceId;
        private static string? _outputPath;
        private static bool _streamingPlayback;
        private static WasapiLowLatencyHelper? _latencyHelper;

        [SupportedOSPlatform("windows")]
        static async Task Main(string[] args)
        {
            PrintBanner();

            // Parse arguments
            string configPath = File.Exists("config.json") ? Path.GetFullPath("config.json") : TTSConfig.DefaultConfigPath;
            List<string> remainingArgs = new List<string>();
            bool stopParsingFlags = false;

            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];

                if (stopParsingFlags)
                {
                    remainingArgs.Add(arg);
                    continue;
                }

                if (arg == "--")
                {
                    stopParsingFlags = true;
                    continue;
                }

                if (arg.StartsWith("-"))
                {
                    string flag = arg;
                    string? value = null;

                    if (arg.Contains('='))
                    {
                        var parts = arg.Split('=', 2);
                        flag = parts[0];
                        value = parts[1];
                    }

                    switch (flag.ToLower())
                    {
                        case "-c":
                        case "--config":
                            if (value == null)
                            {
                                if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) value = args[++i];
                                else { Console.WriteLine("Error: Missing value for config flag."); ShowUsage(); return; }
                            }
                            configPath = value;
                            break;

                        case "-o":
                        case "-O":
                        case "--output":
                            if (value == null)
                            {
                                if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) value = args[++i];
                                else { Console.WriteLine("Error: Missing value for output flag."); ShowUsage(); return; }
                            }
                            _outputPath = value.Trim('\"');
                            break;

                        case "-s":
                        case "--stream":
                            _streamingPlayback = true;
                            break;

                        case "-h":
                        case "--help":
                        case "/?":
                            ShowUsage();
                            return;

                        default:
                            Console.WriteLine($"Error: Unknown flag '{flag}'");
                            ShowUsage();
                            return;
                    }
                }
                else
                {
                    remainingArgs.Add(arg);
                }
            }

            Console.WriteLine($"Loading config from: {configPath}");
            var config = TTSConfig.LoadOrCreate(configPath);

            // 尝试启用低延迟模式
            try
            {
                _latencyHelper = new WasapiLowLatencyHelper();
                _latencyHelper.EnableLowLatency();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Wasapi] 启用低延迟辅助失败: {ex.Message}");
            }

            using var sdk = new AstraTtsSdk(config);

            Console.WriteLine("Initializing SDK...");
            var sw = Stopwatch.StartNew();
            await sdk.InitializeAsync();
            sw.Stop();
            Console.WriteLine($"SDK Initialized in {sw.ElapsedMilliseconds}ms.");
            Console.WriteLine($"Sampling Rate: {sdk.SamplingRate}Hz");
            Console.WriteLine($"Available Avatars: {sdk.Avatars.Count}");

            // Set default avatar
            _currentAvatarId = config.DefaultAvatarId;

            if (remainingArgs.Count > 0)
            {
                // Simple one-shot mode
                string text = string.Join(" ", remainingArgs);
                await RunOneShot(sdk, text);
            }
            else
            {
                ShowUsage();
                Console.WriteLine("\nEntering interactive mode...\n");
                await RunInteractive(sdk);
            }
        }

        static string GetDefaultOutputPath(string avatarId)
        {
            return $"{avatarId}_{DateTime.Now:yyyyMMdd_HHmmss}.wav";
        }

        static string GetEffectiveOutputPath(string? outputPath, string avatarId)
        {
            if (string.IsNullOrWhiteSpace(outputPath))
            {
                return Path.GetFullPath(GetDefaultOutputPath(avatarId));
            }

            string path = Path.GetFullPath(outputPath.Trim('\"'));

            // 如果路径是一个已存在的目录，或者是以后缀分隔符结尾（暗示是目录）
            if (Directory.Exists(path) || path.EndsWith(Path.DirectorySeparatorChar.ToString()) || path.EndsWith(Path.AltDirectorySeparatorChar.ToString()))
            {
                return Path.Combine(path, GetDefaultOutputPath(avatarId));
            }

            return path;
        }

        static void EnsureDirectoryExists(string filePath)
        {
            string? dir = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }
        }

        static async Task RunOneShot(AstraTtsSdk sdk, string text)
        {
            Console.WriteLine($"Synthesizing: {text}");

            if (_streamingPlayback)
            {
                await RunStreamingPlayback(sdk, text);
            }
            else
            {
                var sw = Stopwatch.StartNew();
                var audio = await sdk.PredictAsync(text, null, _currentAvatarId, _currentReferenceId);
                sw.Stop();

                string fileName = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
                EnsureDirectoryExists(fileName);
                AudioHelper.SaveWav(fileName, audio, sdk.SamplingRate);
                Console.WriteLine($"Saved to {fileName} (Time: {sw.ElapsedMilliseconds}ms)");
            }
        }

        static async Task RunInteractive(AstraTtsSdk sdk)
        {
            while (true)
            {
                Console.Write("Input > ");
                string? input = Console.ReadLine()?.Trim();
                if (string.IsNullOrWhiteSpace(input)) continue;

                if (input.StartsWith("/"))
                {
                    if (input.ToLower() == "/exit")
                        break;
                    await HandleCommand(sdk, input);
                    continue;
                }

                // Synthesize
                if (_streamingPlayback)
                {
                    await RunStreamingPlayback(sdk, input);
                }
                else
                {
                    var sw = Stopwatch.StartNew();
                    var audio = await sdk.PredictAsync(input, null, _currentAvatarId, _currentReferenceId);
                    sw.Stop();

                    string fileName = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
                    EnsureDirectoryExists(fileName);
                    AudioHelper.SaveWav(fileName, audio, sdk.SamplingRate);
                    Console.WriteLine($"Done in {sw.ElapsedMilliseconds}ms. Saved to {fileName}");
                }
            }
        }

        static async Task RunStreamingPlayback(AstraTtsSdk sdk, string text)
        {
            Console.WriteLine("🎵 Streaming playback...");
            var sw = Stopwatch.StartNew();

            var ttsFormat = new WaveFormat(sdk.SamplingRate, 16, 1);
            var lockFreeProvider = new LockFreeWaveProvider(ttsFormat, sdk.Config.LockFreeBufferSize)
            {
                ReadFully = true
            };

            // Setup audio output
            IWaveProvider audioSource;
            MediaFoundationResampler? resampler = null;

            if (sdk.Config.WasapiExclusiveMode)
            {
                var targetFormat = new WaveFormat(48000, 16, 2);
                resampler = new MediaFoundationResampler(lockFreeProvider, targetFormat)
                {
                    ResamplerQuality = 1
                };
                audioSource = resampler;
            }
            else
            {
                audioSource = lockFreeProvider;
            }

            using var waveOut = new WasapiOut(
                sdk.Config.WasapiExclusiveMode
                    ? NAudio.CoreAudioApi.AudioClientShareMode.Exclusive
                    : NAudio.CoreAudioApi.AudioClientShareMode.Shared,
                50);
            waveOut.Init(audioSource);

            using var pipeline = new AudioPipeline(sdk.SamplingRate, 20, 20);

            int chunkCount = 0;
            bool playbackStarted = false;
            int preBufferChunks = sdk.Config.StreamingPreBufferChunks;
            var allAudio = new List<float>();

            await foreach (var chunk in sdk.PredictStreamAsync(text, options: null, avatarId: _currentAvatarId, referenceId: _currentReferenceId))
            {
                chunkCount++;
                int samples = chunk.Length;

                // 通过流水线处理增量 (CrossFade + HPF + PCM 转换)
                var (pcmBytes, pcmLen) = pipeline.ProcessChunk(chunk, samples, false);
                try
                {
                    lockFreeProvider.AddSamples(pcmBytes, 0, pcmLen);
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(pcmBytes);
                }

                if (!playbackStarted && chunkCount >= preBufferChunks)
                {
                    waveOut.Play();
                    playbackStarted = true;
                }

                double bufSecs = lockFreeProvider.BufferedBytes / (double)(sdk.SamplingRate * 2);
                Console.Write($"\r🔊 Chunk {chunkCount}: +{samples / (double)sdk.SamplingRate:F2}s (buf: {bufSecs:F1}s)  ");

                allAudio.AddRange(chunk);
            }

            // 处理最后一块的淡出
            float[] silence = ArrayPool<float>.Shared.Rent(500); // 租用一块较大的静音块触发淡出
            Array.Clear(silence, 0, 500);
            try
            {
                var (pcmBytes, pcmLen) = pipeline.ProcessChunk(silence, 500, true);
                try { lockFreeProvider.AddSamples(pcmBytes, 0, pcmLen); }
                finally { ArrayPool<byte>.Shared.Return(pcmBytes); }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(silence);
            }
            pipeline.Reset();

            if (!playbackStarted)
            {
                waveOut.Play();
            }

            // Wait for playback to finish
            while (lockFreeProvider.BufferedBytes > 0)
            {
                await Task.Delay(100);
            }
            await Task.Delay(200);

            waveOut.Stop();
            sw.Stop();

            Console.WriteLine();
            Console.WriteLine($"✅ Streaming complete | Chunks: {chunkCount} | Time: {sw.ElapsedMilliseconds}ms");

            // 保存到文件 (流式模式也默认保存，除非用户显式关闭，目前这里保持与非流式一致的行为)
            string savePath = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
            EnsureDirectoryExists(savePath);
            AudioHelper.SaveWav(savePath, allAudio.ToArray(), sdk.SamplingRate);
            Console.WriteLine($"Saved to {savePath}");
        }

        static async Task HandleCommand(AstraTtsSdk sdk, string input)
        {
            var parts = input.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
            var rawCommand = parts[0].ToLower(); // e.g., "/out"
            var arg = parts.Length > 1 ? parts[1] : null;

            // 定义所有可用指令及其处理逻辑
            var commands = new (string Name, string Description, Func<Task> Handler)[]
            {
                ("/reload", "Reload configuration", async () => {
                    Console.WriteLine("Reloading configuration...");
                    var sw = Stopwatch.StartNew();
                    await sdk.ReloadConfigAsync();
                    sw.Stop();
                    Console.WriteLine($"Configuration reloaded in {sw.ElapsedMilliseconds}ms.");
                    Console.WriteLine($"Available Avatars: {sdk.Avatars.Count}");
                }),
                ("/avatars", "List all available avatars", async () => {
                    if (sdk.Avatars.Count == 0) Console.WriteLine("No avatars configured.");
                    else {
                        Console.WriteLine("Available Avatars:");
                        foreach (var avatar in sdk.Avatars) {
                            var marker = avatar.Id == _currentAvatarId ? " [*]" : "";
                            Console.WriteLine($"  - {avatar.Id}: {avatar.Name}{marker}");
                        }
                    }
                    await Task.CompletedTask;
                }),
                ("/avatar", "<id> - Switch to avatar", async () => {
                    if (string.IsNullOrEmpty(arg)) Console.WriteLine("Usage: /avatar <avatarId>");
                    else {
                        var avatar = sdk.GetAvatar(arg);
                        if (avatar == null) Console.WriteLine($"Avatar '{arg}' not found.");
                        else {
                            _currentAvatarId = arg;
                            _currentReferenceId = avatar.DefaultReferenceId;
                            Console.WriteLine($"Switched to avatar: {avatar.Name} (ID: {avatar.Id})");
                        }
                    }
                    await Task.CompletedTask;
                }),
                ("/refs", "List references for current avatar", async () => {
                    var avatar = sdk.GetAvatar(_currentAvatarId);
                    if (avatar == null) {
                        Console.WriteLine($"Current avatar '{_currentAvatarId}' not found.");
                    } else if (avatar.References.Count == 0) {
                        Console.WriteLine($"No references configured for avatar '{avatar.Name}'.");
                    } else {
                        Console.WriteLine($"References for '{avatar.Name}':");
                        foreach (var r in avatar.References) {
                            var marker = r.Id == _currentReferenceId ? " [*]" : "";
                            Console.WriteLine($"  - {r.Id}: {r.Name ?? r.AudioPath}{marker}");
                        }
                    }
                    await Task.CompletedTask;
                }),
                ("/ref", "<id> - Switch reference audio", async () => {
                    if (string.IsNullOrEmpty(arg)) Console.WriteLine("Usage: /ref <referenceId>");
                    else {
                        _currentReferenceId = arg;
                        Console.WriteLine($"Reference audio set to: {arg}");
                    }
                    await Task.CompletedTask;
                }),
                ("/stream", "- Toggle streaming playback", async () => {
                    _streamingPlayback = !_streamingPlayback;
                    Console.WriteLine($"Streaming playback: {(_streamingPlayback ? "ON" : "OFF")}");
                    await Task.CompletedTask;
                }),
                ("/output", "<path> - Set output file path", async () => {
                    if (string.IsNullOrEmpty(arg)) {
                        Console.WriteLine($"Current output path: {(_outputPath ?? "(Default/Not Set)")}");
                        Console.WriteLine("Usage: /output <path> | off | clear");
                    } else if (arg.ToLower() is "off" or "clear" or "none" or "-") {
                        _outputPath = null;
                        Console.WriteLine("Output path cleared. Will use default filename in current directory.");
                    } else {
                        _outputPath = arg.Trim('\"');
                        string effective = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
                        Console.WriteLine($"Output path base set to: {_outputPath}");
                        Console.WriteLine($"Example full path: {effective}");
                    }
                    await Task.CompletedTask;
                }),
                ("/help", "- Show this help", async () => {
                    Console.WriteLine("Commands:");
                    // 这里可以直接通过变量访问
                    await Task.CompletedTask; // 会在下面单独处理帮助输出显示
                })
            };

            if (rawCommand == "/help" || rawCommand == "/?")
            {
                ShowUsage();
                return;
            }

            // 优先精确匹配
            var exactMatch = commands.FirstOrDefault(c => c.Name.Equals(rawCommand, StringComparison.OrdinalIgnoreCase));
            if (exactMatch.Name != null)
            {
                await exactMatch.Handler();
                return;
            }

            // 模糊匹配 (前缀匹配)
            var matches = commands.Where(c => c.Name.StartsWith(rawCommand, StringComparison.OrdinalIgnoreCase)).ToList();

            if (matches.Count == 0)
            {
                Console.WriteLine($"Unknown command: {rawCommand}. Type /help for available commands.");
            }
            else if (matches.Count == 1)
            {
                var cmd = matches[0];
                Console.WriteLine($"[Fuzzy Match] Executing: {cmd.Name}");
                await cmd.Handler();
            }
            else
            {
                Console.WriteLine($"Ambiguous command '{rawCommand}'. Possible matches:");
                foreach (var m in matches)
                {
                    Console.WriteLine($"  {m.Name}");
                }
            }
        }
        static void PrintBanner()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("========================================");
            Console.WriteLine("       AstraTTS CLI Tool v1.0.0           ");
            Console.WriteLine("========================================");
            Console.ResetColor();
        }

        static void ShowUsage()
        {
            Console.WriteLine("\nUsage: AstraTTS.CLI [options] [text]");

            Console.WriteLine("\n[Options]");
            Console.WriteLine("  -c, --config <path>  Path to config.json (Default: config.json)");
            Console.WriteLine("  -O, --output <path>  Set output WAV file path");
            Console.WriteLine("  -s, --stream         Enable real-time streaming playback");
            Console.WriteLine("  -h, --help           Show this help information");
            Console.WriteLine("  --                   Treat all following arguments as text");

            Console.WriteLine("\n[Interactive Commands]");
            Console.WriteLine("  /avatar <id>     Switch to a different voice");
            Console.WriteLine("  /avatars         List all available voices");
            Console.WriteLine("  /ref <id>        Switch reference audio within current voice");
            Console.WriteLine("  /refs            List all reference audios for current voice");
            Console.WriteLine("  /stream          Toggle streaming playback ON/OFF");
            Console.WriteLine("  /output <path>   Change output file path");
            Console.WriteLine("  /reload          Reload configuration and models");
            Console.WriteLine("  /help            Show this command list");
            Console.WriteLine("  /exit            Quit AstraTTS CLI");
            Console.WriteLine();
        }
    }
}

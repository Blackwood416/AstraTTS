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
            Console.WriteLine("=== AstraTTS CLI Tool ===");

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
                        case "--output":
                            if (value == null)
                            {
                                if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) value = args[++i];
                                else { Console.WriteLine("Error: Missing value for output flag."); ShowUsage(); return; }
                            }
                            _outputPath = value;
                            break;

                        case "-s":
                        case "--stream":
                            _streamingPlayback = true;
                            break;

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
                Console.WriteLine("\nUsage: AstraTTS.CLI [options] [text]");
                Console.WriteLine("\nOptions:");
                Console.WriteLine("  -c, --config <path>  Config file path");
                Console.WriteLine("  -O, --output <path>  Output audio file path");
                Console.WriteLine("  -s, --stream         Enable streaming playback");
                Console.WriteLine("\nCommands:");
                Console.WriteLine("  /reload          - Reload configuration");
                Console.WriteLine("  /avatars         - List available avatars");
                Console.WriteLine("  /switch <id>     - Switch avatar");
                Console.WriteLine("  /ref <id>        - Switch reference audio");
                Console.WriteLine("  /stream          - Toggle streaming playback");
                Console.WriteLine("  /help            - Show this help");
                Console.WriteLine("  exit | q         - Quit");
                Console.WriteLine("\nEntering interactive mode...\n");
                await RunInteractive(sdk);
            }
        }

        static string GetDefaultOutputPath(string avatarId)
        {
            return $"{avatarId}_{DateTime.Now:yyyyMMdd_HHmmss}.wav";
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

                string fileName = _outputPath ?? GetDefaultOutputPath(_currentAvatarId ?? "default");
                AudioHelper.SaveWav(fileName, audio, sdk.SamplingRate);
                Console.WriteLine($"Saved to {fileName} (Time: {sw.ElapsedMilliseconds}ms)");
            }
        }

        static async Task RunInteractive(AstraTtsSdk sdk)
        {
            while (true)
            {
                Console.Write("Input > ");
                string? input = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(input)) continue;

                // Handle commands
                if (input.StartsWith("/"))
                {
                    await HandleCommand(sdk, input);
                    continue;
                }

                if (input == "exit" || input == "q") break;

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

                    string fileName = _outputPath ?? GetDefaultOutputPath(_currentAvatarId ?? "default");
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

            // Optionally save to file
            if (!string.IsNullOrEmpty(_outputPath))
            {
                AudioHelper.SaveWav(_outputPath, allAudio.ToArray(), sdk.SamplingRate);
                Console.WriteLine($"Saved to {_outputPath}");
            }
        }

        static async Task HandleCommand(AstraTtsSdk sdk, string input)
        {
            var parts = input.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
            var command = parts[0].ToLower();
            var arg = parts.Length > 1 ? parts[1] : null;

            switch (command)
            {
                case "/reload":
                    Console.WriteLine("Reloading configuration...");
                    var sw = Stopwatch.StartNew();
                    await sdk.ReloadConfigAsync();
                    sw.Stop();
                    Console.WriteLine($"Configuration reloaded in {sw.ElapsedMilliseconds}ms.");
                    Console.WriteLine($"Available Avatars: {sdk.Avatars.Count}");
                    break;

                case "/avatars":
                    if (sdk.Avatars.Count == 0)
                    {
                        Console.WriteLine("No avatars configured.");
                    }
                    else
                    {
                        Console.WriteLine("Available Avatars:");
                        foreach (var avatar in sdk.Avatars)
                        {
                            var marker = avatar.Id == _currentAvatarId ? " [*]" : "";
                            Console.WriteLine($"  - {avatar.Id}: {avatar.Name}{marker}");
                            foreach (var r in avatar.References)
                            {
                                var refMarker = r.Id == _currentReferenceId ? " [*]" : "";
                                Console.WriteLine($"      - {r.Id}: {r.Name ?? r.AudioPath}{refMarker}");
                            }
                        }
                    }
                    break;

                case "/switch":
                    if (string.IsNullOrEmpty(arg))
                    {
                        Console.WriteLine("Usage: /switch <avatarId>");
                    }
                    else
                    {
                        var avatar = sdk.GetAvatar(arg);
                        if (avatar == null)
                        {
                            Console.WriteLine($"Avatar '{arg}' not found.");
                        }
                        else
                        {
                            _currentAvatarId = arg;
                            _currentReferenceId = avatar.DefaultReferenceId;
                            Console.WriteLine($"Switched to avatar: {avatar.Name} (ID: {avatar.Id})");
                        }
                    }
                    break;

                case "/ref":
                    if (string.IsNullOrEmpty(arg))
                    {
                        Console.WriteLine("Usage: /ref <referenceId>");
                    }
                    else
                    {
                        _currentReferenceId = arg;
                        Console.WriteLine($"Reference audio set to: {arg}");
                    }
                    break;

                case "/stream":
                    _streamingPlayback = !_streamingPlayback;
                    Console.WriteLine($"Streaming playback: {(_streamingPlayback ? "ON" : "OFF")}");
                    break;

                case "/output":
                    if (string.IsNullOrEmpty(arg))
                    {
                        _outputPath = null;
                        Console.WriteLine("Output path cleared. Will use default filename.");
                    }
                    else
                    {
                        _outputPath = arg;
                        Console.WriteLine($"Output path set to: {arg}");
                    }
                    break;

                case "/help":
                    Console.WriteLine("Commands:");
                    Console.WriteLine("  /reload          - Reload configuration");
                    Console.WriteLine("  /avatars         - List available avatars");
                    Console.WriteLine("  /switch <id>     - Switch avatar");
                    Console.WriteLine("  /ref <id>        - Switch reference audio");
                    Console.WriteLine("  /stream          - Toggle streaming playback");
                    Console.WriteLine("  /output <path>   - Set output file path");
                    Console.WriteLine("  /help            - Show this help");
                    Console.WriteLine("  exit | q         - Quit");
                    break;

                default:
                    Console.WriteLine($"Unknown command: {command}. Type /help for available commands.");
                    break;
            }
        }
        static void ShowUsage()
        {
            Console.WriteLine("\nUsage: AstraTTS.CLI [options] [text]");
            Console.WriteLine("\nOptions:");
            Console.WriteLine("  -c, --config <path>  Config file path");
            Console.WriteLine("  -O, --output <path>  Output audio file path");
            Console.WriteLine("  -s, --stream         Enable streaming playback");
            Console.WriteLine("  --                   Stop parsing flags and treat remaining as text");
            Console.WriteLine("\nCommands (interactive mode):");
            Console.WriteLine("  /reload          - Reload configuration");
            Console.WriteLine("  /avatars         - List available avatars");
            Console.WriteLine("  /switch <id>     - Switch avatar");
            Console.WriteLine("  /ref <id>        - Switch reference audio");
            Console.WriteLine("  /stream          - Toggle streaming playback");
            Console.WriteLine("  /help            - Show this help");
            Console.WriteLine("  exit | q         - Quit");
        }
    }
}

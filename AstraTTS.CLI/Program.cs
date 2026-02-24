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
        private static int? _priorityMode;
        private static WasapiLowLatencyHelper? _latencyHelper;

        [SupportedOSPlatform("windows")]
        static async Task Main(string[] args)
        {
            // Set console encoding to UTF-8 to support Chinese and Japanese input/output
            Console.InputEncoding = System.Text.Encoding.UTF8;
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            PrintBanner();

            // Parse arguments
            string? configPath = null;
            bool? forceDebug = null;
            List<string> remainingArgs = new List<string>();
            bool stopParsingFlags = false;

            List<string>? langsOverride = null;
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

                        case "-p":
                        case "--priority":
                            if (value == null)
                            {
                                if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) value = args[++i];
                                else { Console.WriteLine("Error: Missing value for priority flag."); ShowUsage(); return; }
                            }
                            if (int.TryParse(value, out int p)) _priorityMode = p;
                            break;

                        case "-h":
                        case "--help":
                        case "/?":
                            ShowUsage();
                            return;

                        case "-L":
                        case "--langs":
                            if (value == null)
                            {
                                if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) value = args[++i];
                                else { Console.WriteLine("Error: Missing value for langs flag."); ShowUsage(); return; }
                            }
                            langsOverride = value.Split(',').Select(s => s.Trim()).ToList();
                            break;

                        case "--debug":
                            forceDebug = true;
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

            var config = configPath != null ? TTSConfig.LoadOrCreate(configPath) : TTSConfig.Load();
            if (forceDebug.HasValue) config.DebugMode = forceDebug.Value;
            if (langsOverride != null) config.G2P.Languages = langsOverride;
            Console.WriteLine($"Loading config from: {TTSConfig.LoadedPath}");

            // 如果命令行未强制开启 (-s)，则遵循配置文件
            if (!_streamingPlayback)
                _streamingPlayback = config.StreamingMode;

            // 尝试启用低延迟模式 (仅 Windows)
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
            {
                try
                {
                    _latencyHelper = new WasapiLowLatencyHelper();
                    _latencyHelper.EnableLowLatency();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Wasapi] 启用低延迟辅助失败: {ex.Message}");
                }
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

            // 智能判断是目录还是文件：
            // 1. 已经是一个存在的目录
            // 2. 路径以分隔符结尾 (e.g. "out/")
            // 3. 路径没有扩展名 (e.g. "results" -> 视为目录)
            bool isExplicitDir = Directory.Exists(path) ||
                                 path.EndsWith(Path.DirectorySeparatorChar.ToString()) ||
                                 path.EndsWith(Path.AltDirectorySeparatorChar.ToString());

            bool hasExtension = Path.HasExtension(path);

            if (isExplicitDir || !hasExtension)
            {
                // 视为目录，拼接带时间戳的文件名
                return Path.Combine(path, GetDefaultOutputPath(avatarId));
            }

            // 视为具体文件名
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
            try
            {
                Console.WriteLine($"Synthesizing: {text}");

                if (_streamingPlayback)
                {
                    await RunStreamingPlayback(sdk, text);
                }
                else
                {
                    var sw = Stopwatch.StartNew();
                    var options = new TtsOptions { G2PPriorityMode = _priorityMode };
                    var result = await sdk.PredictAsync(text, options, _currentAvatarId, _currentReferenceId);
                    sw.Stop();

                    string fileName = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
                    EnsureDirectoryExists(fileName);
                    AudioHelper.SaveWav(fileName, result.Audio, sdk.SamplingRate);
                    Console.WriteLine($"Saved to {fileName} (Time: {sw.ElapsedMilliseconds}ms)");
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[Error] Synthesis failed: {ex.Message}");
                Console.ResetColor();
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
                try
                {
                    if (_streamingPlayback)
                    {
                        await RunStreamingPlayback(sdk, input);
                    }
                    else
                    {
                        var sw = Stopwatch.StartNew();
                        var options = new TtsOptions { G2PPriorityMode = _priorityMode };
                        var result = await sdk.PredictAsync(input, options, _currentAvatarId, _currentReferenceId);
                        sw.Stop();

                        string fileName = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
                        EnsureDirectoryExists(fileName);
                        AudioHelper.SaveWav(fileName, result.Audio, sdk.SamplingRate);
                        Console.WriteLine($"Done in {sw.ElapsedMilliseconds}ms. Saved to {fileName}");
                    }
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\n[Error] {ex.Message}");
                    Console.ResetColor();
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
            bool isWindows = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows);
            IDisposable? waveOut = null;
            LinuxAudioPlayer? linuxPlayer = null;

            if (isWindows)
            {
                var wo = new WasapiOut(
                    sdk.Config.WasapiExclusiveMode
                        ? NAudio.CoreAudioApi.AudioClientShareMode.Exclusive
                        : NAudio.CoreAudioApi.AudioClientShareMode.Shared,
                    50);

                IWaveProvider audioSource = lockFreeProvider;
                if (sdk.Config.WasapiExclusiveMode)
                {
                    var targetFormat = new WaveFormat(48000, 16, 2);
                    var resampler = new MediaFoundationResampler(lockFreeProvider, targetFormat) { ResamplerQuality = 1 };
                    audioSource = resampler;
                }
                wo.Init(audioSource);
                waveOut = wo;
            }
            else
            {
                linuxPlayer = new LinuxAudioPlayer();
                linuxPlayer.Init(sdk.SamplingRate);
                waveOut = linuxPlayer;
            }

            using var pipeline = new AudioPipeline(sdk.SamplingRate, 20, 20);
            var options = new TtsOptions { G2PPriorityMode = _priorityMode };

            int chunkCount = 0;
            bool playbackStarted = false;
            int preBufferChunks = sdk.Config.StreamingPreBufferChunks;
            var allAudio = new List<float>();

            await foreach (var chunk in sdk.PredictStreamAsync(text, options, avatarId: _currentAvatarId, referenceId: _currentReferenceId))
            {
                chunkCount++;
                int samples = chunk.Length;

                // 通过流水线处理增量 (CrossFade + HPF + PCM 转换)
                var (pcmBytes, pcmLen) = pipeline.ProcessChunk(chunk, samples, false);
                try
                {
                    if (isWindows)
                    {
                        lockFreeProvider.AddSamples(pcmBytes, 0, pcmLen);
                    }
                    else
                    {
                        linuxPlayer?.AddSamples(pcmBytes, 0, pcmLen);
                    }
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(pcmBytes);
                }

                if (!playbackStarted && chunkCount >= preBufferChunks)
                {
                    if (isWindows) (waveOut as WasapiOut)?.Play();
                    else linuxPlayer?.Play();
                    playbackStarted = true;
                }

                double bufSecs = isWindows ? lockFreeProvider.BufferedBytes / (double)(sdk.SamplingRate * 2) : 0;
                string bufInfo = isWindows ? $"(buf: {bufSecs:F1}s)" : "";
                Console.Write($"\r🔊 Chunk {chunkCount}: +{samples / (double)sdk.SamplingRate:F2}s {bufInfo}  ");

                allAudio.AddRange(chunk);
            }

            // 处理最后一块的淡出
            float[] silence = ArrayPool<float>.Shared.Rent(500);
            Array.Clear(silence, 0, 500);
            try
            {
                var (pcmBytes, pcmLen) = pipeline.ProcessChunk(silence, 500, true);
                try
                {
                    if (isWindows) lockFreeProvider.AddSamples(pcmBytes, 0, pcmLen);
                    else linuxPlayer?.AddSamples(pcmBytes, 0, pcmLen);
                }
                finally { ArrayPool<byte>.Shared.Return(pcmBytes); }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(silence);
            }
            pipeline.Reset();

            if (!playbackStarted)
            {
                if (isWindows) (waveOut as WasapiOut)?.Play();
                else linuxPlayer?.Play();
            }

            // Wait for playback to finish
            if (isWindows)
            {
                while (lockFreeProvider.BufferedBytes > 0) await Task.Delay(100);
            }
            else
            {
                linuxPlayer?.WaitForFinish();
            }
            await Task.Delay(200);

            if (isWindows) (waveOut as WasapiOut)?.Stop();
            else linuxPlayer?.Stop();

            waveOut?.Dispose();
            sw.Stop();

            Console.WriteLine();
            Console.WriteLine($"✅ Streaming complete | Chunks: {chunkCount} | Time: {sw.ElapsedMilliseconds}ms");

            string savePath = GetEffectiveOutputPath(_outputPath, _currentAvatarId ?? "default");
            EnsureDirectoryExists(savePath);
            AudioHelper.SaveWav(savePath, allAudio.ToArray(), sdk.SamplingRate);
            Console.WriteLine($"Saved to {savePath}");
        }

        /// <summary>
        /// Linux 环境下的音频播放器，通过管道 (Pipe) 调用系统播放器
        /// </summary>
        class LinuxAudioPlayer : IDisposable
        {
            private Process? _process;
            private Stream? _stdin;
            private string _command = "aplay";
            private bool _isStarted = false;

            public void Init(int sampleRate)
            {
                // 探测播放器
                if (CanRun("pw-play")) _command = "pw-play";
                else if (CanRun("paplay")) _command = "paplay";
                else if (CanRun("aplay")) _command = "aplay";
                else if (CanRun("ffplay")) _command = "ffplay";

                var args = _command switch
                {
                    "pw-play" => $"--format=s16 --rate={sampleRate} --channels=1 -",
                    "paplay" => $"--format=s16le --rate={sampleRate} --channels=1 --raw",
                    "aplay" => $"-f S16_LE -r {sampleRate} -c 1",
                    "ffplay" => $"-f s16le -ar {sampleRate} -ac 1 -nodisp -autoexit -i pipe:0",
                    _ => ""
                };

                try
                {
                    _process = new Process
                    {
                        StartInfo = new ProcessStartInfo
                        {
                            FileName = _command,
                            Arguments = args,
                            RedirectStandardInput = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }
                    };
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[LinuxAudio] 无法初始化播放进程 ({_command}): {ex.Message}");
                }
            }

            private bool CanRun(string cmd)
            {
                try
                {
                    using var p = Process.Start(new ProcessStartInfo
                    {
                        FileName = "which",
                        Arguments = cmd,
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    });
                    p?.WaitForExit();
                    return p?.ExitCode == 0;
                }
                catch { return false; }
            }

            public void Play()
            {
                if (_isStarted) return;
                try
                {
                    _process?.Start();
                    _stdin = _process?.StandardInput.BaseStream;
                    _isStarted = true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[LinuxAudio] 启动播放失败: {ex.Message}");
                }
            }

            public void AddSamples(byte[] buffer, int offset, int count)
            {
                try
                {
                    _stdin?.Write(buffer, offset, count);
                    _stdin?.Flush();
                }
                catch { /* 进程可能已退出 */ }
            }

            public void WaitForFinish()
            {
                // 关闭 stdin 触发播放器结束
                _stdin?.Close();
                _process?.WaitForExit(2000);
            }

            public void Stop()
            {
                try
                {
                    if (_process != null && !_process.HasExited) _process.Kill();
                }
                catch { }
            }

            public void Dispose()
            {
                Stop();
                _process?.Dispose();
            }
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
                    // Sync CLI state with new config
                    _currentAvatarId = sdk.Config.DefaultAvatarId;
                    var avatar = sdk.GetAvatar(_currentAvatarId);
                    _currentReferenceId = avatar?.DefaultReferenceId;

                    Console.WriteLine($"Configuration reloaded in {sw.ElapsedMilliseconds}ms.");
                    Console.WriteLine($"Current Avatar: {_currentAvatarId} (Synced)");
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
                ("/priority", "<mode> - Set G2P priority (0=DictFirst, 1=DictOnly, 2=ModelFirst)", async () => {
                    if (string.IsNullOrEmpty(arg)) {
                        Console.WriteLine($"Current G2P Priority: {(_priorityMode?.ToString() ?? "Default (0)")}");
                        Console.WriteLine("Usage: /priority <0|1|2|default>");
                    } else if (arg.ToLower() == "default" || arg == "-1") {
                        _priorityMode = null;
                        Console.WriteLine("G2P Priority set to: Default (from config)");
                    } else if (int.TryParse(arg, out int p)) {
                        _priorityMode = p;
                        sdk.Config.G2P.PriorityMode = p; // 同步到 SDK 配置以便 Apply 时一致
                        await sdk.ApplyConfigAsync();
                        Console.WriteLine($"G2P Priority set to: {p}");
                    }
                    await Task.CompletedTask;
                }),
                ("/langs", "<l1,l2> - Set allowed languages (zh, en, ja)", async () => {
                    if (string.IsNullOrEmpty(arg)) {
                        Console.WriteLine($"Current Allowed Languages: {string.Join(", ", sdk.Config.G2P.Languages)}");
                        Console.WriteLine("Usage: /langs zh,en | zh,ja | en");
                    } else {
                        var newLangs = arg.Split(',').Select(s => s.Trim()).ToList();
                        sdk.Config.G2P.Languages = newLangs;
                        await sdk.ApplyConfigAsync(); // 重载以应用语言约束重选 MixedLanguageG2P
                        Console.WriteLine($"Allowed Languages set to: {string.Join(", ", newLangs)}");
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
            Console.WriteLine("  -c, --config <path>  Path to config.yaml (Default: config.yaml)");
            Console.WriteLine("  -O, --output <path>  Output file or directory path");
            Console.WriteLine("  -s, --stream         Enable real-time streaming playback");
            Console.WriteLine("  -p, --priority <0|1|2> G2P priority mode (0=DictFirst, 1=DictOnly, 2=ModelFirst)");
            Console.WriteLine("  -L, --langs <l1[,l2]>  Allowed languages (comma separated, e.g. zh,en)");
            Console.WriteLine("  --debug              Enable detailed debug logging");
            Console.WriteLine("  -h, --help           Show this help information");
            Console.WriteLine("  --                   Treat all following arguments as text");

            Console.WriteLine("\n[Interactive Commands]");
            Console.WriteLine("  /avatar <id>     Switch to a different voice");
            Console.WriteLine("  /avatars         List all available voices");
            Console.WriteLine("  /ref <id>        Switch reference audio within current voice");
            Console.WriteLine("  /refs            List all reference audios for current voice");
            Console.WriteLine("  /stream          Toggle streaming playback ON/OFF");
            Console.WriteLine("  /output <path>   Change output file or directory");
            Console.WriteLine("  /priority <0|1|2> Change G2P priority mode");
            Console.WriteLine("  /langs <l1[,l2]>   Change allowed languages (comma separated, e.g. zh,en)");
            Console.WriteLine("  /reload          Reload configuration and models");
            Console.WriteLine("  /help            Show this command list");
            Console.WriteLine("  /exit            Quit AstraTTS CLI");
            Console.WriteLine();
        }
    }
}

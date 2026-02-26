using Microsoft.AspNetCore.Mvc;
using AstraTTS.Core.Core;
using AstraTTS.Core.Config;
using AstraTTS.Core.Utils;
using AstraTTS.Web.Models;
using AstraTTS.Web.Services;
using System.Diagnostics;

namespace AstraTTS.Web.Controllers
{
    /// <summary>
    /// TTS 合成控制器
    /// </summary>
    [ApiController]
    [Route("api/tts")]
    public class TtsController : ControllerBase
    {
        private readonly AstraTtsSdk _sdk;
        private readonly StreamSessionManager _sessionManager;

        public TtsController(AstraTtsSdk sdk, StreamSessionManager sessionManager)
        {
            _sdk = sdk;
            _sessionManager = sessionManager;
        }

        /// <summary>
        /// 全量语音合成。返回 WAV 文件。
        /// </summary>
        /// <param name="request">合成请求</param>
        [HttpPost("predict")]
        public async Task<IActionResult> Predict([FromBody] TtsRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Text))
                return BadRequest("Text cannot be empty");

            var options = new TtsOptions
            {
                Speed = request.Speed ?? _sdk.Config.Speed,
                NoiseScale = request.NoiseScale ?? _sdk.Config.NoiseScale,
                Temperature = request.Temperature ?? _sdk.Config.Temperature,
                TopK = request.TopK ?? _sdk.Config.TopK,
                G2PPriorityMode = request.G2PPriorityMode,
                Languages = request.Languages
            };

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = await _sdk.PredictAsync(request.Text, options, request.AvatarId, request.ReferenceId);
            sw.Stop();

            var elapsedMs = sw.Elapsed.TotalMilliseconds;
            var audioDurationSec = (double)result.Audio.Length / _sdk.SamplingRate;
            var rtf = elapsedMs / 1000.0 / audioDurationSec;

            Response.Headers.Append("X-Synthesis-Time", elapsedMs.ToString("F2"));
            Response.Headers.Append("X-Audio-Duration", audioDurationSec.ToString("F3"));
            Response.Headers.Append("X-RTF", rtf.ToString("F3"));
            Response.Headers.Append("X-Token-Count", result.TokenCount.ToString());

            using var ms = new MemoryStream();
            AudioHelper.SaveWav(ms, result.Audio, _sdk.SamplingRate);

            return File(ms.ToArray(), "audio/wav", "output.wav");
        }

        /// <summary>
        /// 流式语音合成（GET 方法，支持 ffplay 直接读取）。返回音频流 (PCM Float32)。
        /// 使用示例: ffplay -f f32le -ar 44100 -ac 1 "http://localhost:5000/api/tts/predict-stream?text=你好世界"
        /// </summary>
        /// <param name="text">待合成的文本</param>
        /// <param name="avatarId">音色 ID（可选）</param>
        /// <param name="referenceId">参考音频 ID（可选）</param>
        /// <param name="speed">语速 0.5-2.0（可选）</param>
        /// <param name="noiseScale">噪声系数（可选）</param>
        /// <param name="temperature">采样温度（可选）</param>
        /// <param name="topK">Top-K 采样（可选）</param>
        /// <param name="chunkSize">流式分块大小（可选）</param>
        [HttpGet("predict-stream")]
        public async Task PredictStream(
            [FromQuery] string text,
            [FromQuery] string? avatarId = null,
            [FromQuery] string? referenceId = null,
            [FromQuery] float? speed = null,
            [FromQuery] float? noiseScale = null,
            [FromQuery] float? temperature = null,
            [FromQuery] int? topK = null,
            [FromQuery] int? chunkSize = null,
            [FromQuery] List<string>? languages = null)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                Response.StatusCode = 400;
                await Response.WriteAsync("text parameter is required");
                return;
            }

            var options = new TtsOptions
            {
                Speed = speed ?? _sdk.Config.Speed,
                NoiseScale = noiseScale ?? _sdk.Config.NoiseScale,
                Temperature = temperature ?? _sdk.Config.Temperature,
                TopK = topK ?? _sdk.Config.TopK,
                StreamingChunkSize = chunkSize ?? _sdk.Config.StreamingChunkSize,
                Languages = (languages != null && languages.Count > 0) ? languages : null
            };

            Response.ContentType = "audio/pcm";
            Response.Headers.Append("Content-Disposition", "inline; filename=\"stream.pcm\"");
            Response.Headers.Append("X-Audio-Sample-Rate", _sdk.SamplingRate.ToString());
            Response.Headers.Append("X-Audio-Channels", "1");
            Response.Headers.Append("X-Audio-Format", "f32le");

            try
            {
                await foreach (var chunk in _sdk.PredictStreamAsync(text, options, avatarId, referenceId, HttpContext.RequestAborted))
                {
                    byte[] bytes = new byte[chunk.Length * 4];
                    Buffer.BlockCopy(chunk, 0, bytes, 0, bytes.Length);
                    await Response.Body.WriteAsync(bytes, HttpContext.RequestAborted);
                    await Response.Body.FlushAsync(HttpContext.RequestAborted);
                }
            }
            catch (OperationCanceledException)
            {
                Response.StatusCode = 499;
            }
        }

        /// <summary>
        /// 创建流式合成会话。返回会话 ID 和流式播放 URL。
        /// </summary>
        /// <param name="request">合成请求</param>
        [HttpPost("stream/create")]
        public Task<IActionResult> CreateStreamSession([FromBody] TtsRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Text))
                return Task.FromResult<IActionResult>(BadRequest(new { error = "Text cannot be empty" }));

            var sessionId = _sessionManager.CreateSession(
                    request.Text,
                    request.AvatarId,
                    request.ReferenceId,
                    request.Speed ?? _sdk.Config.Speed,
                    request.NoiseScale ?? _sdk.Config.NoiseScale,
                    request.Temperature ?? _sdk.Config.Temperature,
                    request.TopK ?? _sdk.Config.TopK,
                    request.StreamingChunkSize ?? _sdk.Config.StreamingChunkSize,
                    request.G2PPriorityMode,
                    request.Languages
                );

            var streamUrl = Url.Action(nameof(GetStream), new { sessionId = sessionId });

            return Task.FromResult<IActionResult>(Ok(new
            {
                sessionId,
                streamUrl,
                contentType = "audio/pcm",
                sampleRate = _sdk.SamplingRate,
                channels = 1,
                format = "f32le"
            }));
        }


        /// <summary>
        /// 获取流式音频数据。返回 PCM Float32 音频流。
        /// </summary>
        /// <param name="sessionId">会话 ID</param>
        [HttpGet("stream/{sessionId}")]
        public async Task GetStream(string sessionId)
        {
            var session = _sessionManager.GetSession(sessionId);
            if (session == null)
            {
                Response.StatusCode = 404;
                return;
            }

            Response.ContentType = "audio/pcm";
            Response.Headers.Append("Content-Disposition", $"inline; filename=\"{sessionId}.pcm\"");
            Response.Headers.Append("X-Audio-Sample-Rate", _sdk.SamplingRate.ToString());
            Response.Headers.Append("X-Audio-Channels", "1");
            Response.Headers.Append("X-Audio-Format", "f32le");

            var options = new TtsOptions
            {
                Speed = session.Speed,
                NoiseScale = session.NoiseScale,
                Temperature = session.Temperature,
                TopK = session.TopK,
                StreamingChunkSize = session.StreamingChunkSize,
                G2PPriorityMode = session.G2PPriorityMode,
                Languages = session.Languages
            };

            try
            {
                await foreach (var chunk in _sdk.PredictStreamAsync(session.Text, options, session.AvatarId, session.ReferenceId, session.CancellationTokenSource.Token))
                {
                    byte[] bytes = new byte[chunk.Length * 4];
                    Buffer.BlockCopy(chunk, 0, bytes, 0, bytes.Length);
                    await Response.Body.WriteAsync(bytes, session.CancellationTokenSource.Token);
                    await Response.Body.FlushAsync(session.CancellationTokenSource.Token);
                }

                _sessionManager.CompleteSession(sessionId);
            }
            catch (OperationCanceledException)
            {
                Response.StatusCode = 499;
                await Response.WriteAsync("Stream cancelled by client");
            }
            finally
            {
                _sessionManager.RemoveSession(sessionId);
            }
        }

        /// <summary>
        /// 取消流式合成会话。
        /// </summary>
        /// <param name="sessionId">会话 ID</param>
        [HttpDelete("stream/{sessionId}")]
        public IActionResult CancelStreamSession(string sessionId)
        {
            var session = _sessionManager.GetSession(sessionId);
            if (session == null)
                return NotFound(new { error = "Session not found" });

            _sessionManager.CancelSession(sessionId);
            _sessionManager.RemoveSession(sessionId);

            return Ok(new { message = "Session cancelled" });
        }

        /// <summary>
        /// 获取所有活动的流式会话。
        /// </summary>
        [HttpGet("stream/sessions")]
        public IActionResult GetStreamSessions()
        {
            var sessions = _sessionManager.GetAllSessions()
                .Select(s => new
                {
                    s.SessionId,
                    s.Text,
                    s.AvatarId,
                    s.ReferenceId,
                    s.Speed,
                    s.CreatedAt,
                    s.IsCompleted,
                    Age = (DateTime.UtcNow - s.CreatedAt).TotalSeconds
                });

            return Ok(sessions);
        }


        /// <summary>
        /// 获取所有可用的音色列表。
        /// </summary>
        [HttpGet("avatars")]
        public IActionResult GetAvatars()
        {
            return Ok(_sdk.Avatars.Select(a => new { a.Id, a.Name, a.Description, ReferenceCount = a.References.Count }));
        }

        /// <summary>
        /// 获取指定音色的详细信息，包括所有参考音频。
        /// </summary>
        /// <param name="avatarId">音色 ID</param>
        [HttpGet("avatars/{avatarId}")]
        public IActionResult GetAvatar(string avatarId)
        {
            var avatar = _sdk.GetAvatar(avatarId);
            if (avatar == null)
                return NotFound(new { error = $"Avatar '{avatarId}' not found." });

            return Ok(new
            {
                avatar.Id,
                avatar.Name,
                avatar.Description,
                avatar.DefaultReferenceId,
                References = avatar.References.Select(r => new
                {
                    r.Id,
                    r.Name,
                    r.Text,
                    r.Language,
                    r.AudioPath
                })
            });
        }

        /// <summary>
        /// 获取指定音色的所有参考音频。
        /// </summary>
        /// <param name="avatarId">音色 ID</param>
        [HttpGet("avatars/{avatarId}/refs")]
        public IActionResult GetAvatarReferences(string avatarId)
        {
            var avatar = _sdk.GetAvatar(avatarId);
            if (avatar == null)
                return NotFound(new { error = $"Avatar '{avatarId}' not found." });

            return Ok(avatar.References.Select(r => new
            {
                r.Id,
                r.Name,
                r.Text,
                r.Language,
                r.AudioPath
            }));
        }

        /// <summary>
        /// 热重载配置。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("reload")]
        public async Task<IActionResult> Reload()
        {
            try
            {
                await _sdk.ReloadConfigAsync();
                return Ok(new { success = true, message = "Configuration reloaded successfully." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { success = false, message = ex.Message });
            }
        }

        /// <summary>
        /// 获取引擎信息。
        /// </summary>
        [HttpGet("info")]
        public IActionResult GetInfo()
        {
            return Ok(new
            {
                Engine = _sdk.Config.UseEngineV2 ? "V2 (GPT-SoVITS-based)" : "V1 (Genie-TTS)",
                SamplingRate = _sdk.SamplingRate,
                Device = "CPU", // 暂时硬编码，后续可从 SDK 获取
                DefaultAvatarId = _sdk.Config.DefaultAvatarId,
                Avatars = _sdk.Avatars.Select(a => new { a.Id, a.Name })
            });
        }

        /// <summary>
        /// 获取健康状态。
        /// </summary>
        [HttpGet("status")]
        public IActionResult GetStatus()
        {
            return Ok(new
            {
                Status = "Online",
                Timestamp = DateTime.UtcNow
            });
        }

        /// <summary>
        /// 获取完整配置。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("config")]
        public IActionResult GetConfig()
        {
            return Ok(_sdk.Config);
        }

        /// <summary>
        /// 更新并保存配置。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("config")]
        public async Task<IActionResult> UpdateConfig([FromBody] TTSConfig newConfig)
        {
            try
            {
                if (string.IsNullOrEmpty(TTSConfig.LoadedPath))
                    return StatusCode(500, new { success = false, message = "No config file loaded to save to." });

                newConfig.Save(TTSConfig.LoadedPath);
                await _sdk.ReloadConfigAsync();
                return Ok(new { success = true, message = "Configuration updated and saved successfully." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { success = false, message = ex.Message });
            }
        }

        /// <summary>
        /// 重置设置为默认配置
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("config/reset")]
        public async Task<IActionResult> ResetConfig()
        {
            try
            {
                if (string.IsNullOrEmpty(TTSConfig.LoadedPath))
                    return StatusCode(500, new { success = false, message = "No config file loaded." });

                var configDir = Path.GetDirectoryName(Path.GetFullPath(TTSConfig.LoadedPath));
                var templatePath = Path.Combine(configDir ?? "", "config.template.yaml");

                if (!System.IO.File.Exists(templatePath))
                {
                    return StatusCode(404, new { success = false, message = $"config.template.yaml not found at {templatePath}." });
                }

                System.IO.File.Copy(templatePath, TTSConfig.LoadedPath, true);
                await _sdk.ReloadConfigAsync();

                return Ok(new { success = true, message = "Configuration reset to default successfully." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { success = false, message = ex.Message });
            }
        }


        // ============================================================
        // Avatar 管理接口
        // ============================================================

        /// <summary>
        /// 添加或更新音色。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("avatars")]
        public async Task<IActionResult> AddOrUpdateAvatar([FromBody] Avatar avatar)
        {
            if (string.IsNullOrEmpty(avatar.Id)) return BadRequest("Avatar ID is required.");

            var existing = _sdk.Config.Avatars.Find(a => a.Id == avatar.Id);
            if (existing != null)
            {
                _sdk.Config.Avatars.Remove(existing);
            }
            _sdk.Config.Avatars.Add(avatar);

            return await SaveAndReloadConfig();
        }

        /// <summary>
        /// 删除音色。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpDelete("avatars/{id}")]
        public async Task<IActionResult> DeleteAvatar(string id)
        {
            var avatar = _sdk.Config.Avatars.Find(a => a.Id == id);
            if (avatar == null) return NotFound($"Avatar '{id}' not found.");

            _sdk.Config.Avatars.Remove(avatar);
            if (_sdk.Config.DefaultAvatarId == id)
            {
                _sdk.Config.DefaultAvatarId = _sdk.Config.Avatars.FirstOrDefault()?.Id ?? "";
            }

            return await SaveAndReloadConfig();
        }

        /// <summary>
        /// 为指定音色添加/更新参考音频。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpPost("avatars/{avatarId}/references")]
        public async Task<IActionResult> AddReference(string avatarId, [FromBody] ReferenceAudio reference)
        {
            var avatar = _sdk.Config.Avatars.Find(a => a.Id == avatarId);
            if (avatar == null) return NotFound($"Avatar '{avatarId}' not found.");

            var existing = avatar.References.Find(r => r.Id == reference.Id);
            if (existing != null)
            {
                avatar.References.Remove(existing);
            }
            avatar.References.Add(reference);

            return await SaveAndReloadConfig();
        }

        /// <summary>
        /// 删除参考音频。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpDelete("avatars/{avatarId}/references/{refId}")]
        public async Task<IActionResult> DeleteReference(string avatarId, string refId)
        {
            var avatar = _sdk.Config.Avatars.Find(a => a.Id == avatarId);
            if (avatar == null) return NotFound($"Avatar '{avatarId}' not found.");

            var reference = avatar.References.Find(r => r.Id == refId);
            if (reference == null) return NotFound($"Reference '{refId}' not found.");

            avatar.References.Remove(reference);
            return await SaveAndReloadConfig();
        }

        // ============================================================
        // 文件系统浏览接口
        // ============================================================

        /// <summary>
        /// 列出文件系统中可用的音色目录。
        /// 如果指定了 useV2 参数，会交叉验证 models_v1/models_v2 和 avatars 目录。
        /// </summary>
        /// <param name="useV2">null=返回所有avatars目录, true=交叉验证models_v2, false=交叉验证models_v1</param>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("fs/avatars")]
        public IActionResult ListAvatarDirectories([FromQuery] bool? useV2 = null)
        {
            var resourcesDir = _sdk.Config.ResourcesDir;
            var avatarsDir = _sdk.Config.AvatarsDir;

            // 如果没有指定引擎类型，返回 avatars 目录下所有子目录
            if (useV2 == null)
            {
                if (!Directory.Exists(avatarsDir))
                    return Ok(Array.Empty<object>());

                var all = Directory.GetDirectories(avatarsDir)
                    .Select(Path.GetFileName)
                    .Where(n => !string.IsNullOrEmpty(n))
                    .OrderBy(n => n)
                    .Select(n => new { id = n, hasModel = false, hasAvatar = true })
                    .ToArray();
                return Ok(all);
            }

            // 交叉验证：模型目录 + avatars 目录
            var modelsDir = Path.Combine(resourcesDir, useV2 == true ? "models_v2" : "models_v1");

            var modelIds = Directory.Exists(modelsDir)
                ? Directory.GetDirectories(modelsDir).Select(Path.GetFileName).Where(n => !string.IsNullOrEmpty(n)).Select(n => n!).ToHashSet()
                : new HashSet<string>();

            var avatarIds = Directory.Exists(avatarsDir)
                ? Directory.GetDirectories(avatarsDir).Select(Path.GetFileName).Where(n => !string.IsNullOrEmpty(n)).Select(n => n!).ToHashSet()
                : new HashSet<string>();

            // 合并所有发现的 ID
            var allIds = modelIds.Union(avatarIds).OrderBy(n => n);
            var result = allIds.Select(id => new
            {
                id,
                hasModel = modelIds.Contains(id),
                hasAvatar = avatarIds.Contains(id),
                valid = modelIds.Contains(id) && avatarIds.Contains(id)
            }).ToArray();

            return Ok(result);
        }

        /// <summary>
        /// 列出指定音色目录下 references/ 中的音频文件。
        /// </summary>
        /// <param name="avatarId">音色 ID / 目录名</param>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("fs/avatars/{avatarId}/references")]
        public IActionResult ListReferenceFiles(string avatarId)
        {
            var refsDir = Path.Combine(_sdk.Config.AvatarsDir, avatarId, "references");
            if (!Directory.Exists(refsDir))
                return Ok(Array.Empty<string>());

            var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".wav", ".mp3", ".flac", ".ogg" };
            var files = Directory.GetFiles(refsDir)
                .Where(f => exts.Contains(Path.GetExtension(f)))
                .Select(Path.GetFileName)
                .OrderBy(n => n)
                .ToArray();

            return Ok(files);
        }

        private async Task<IActionResult> SaveAndReloadConfig()
        {
            try
            {
                if (string.IsNullOrEmpty(TTSConfig.LoadedPath))
                    return StatusCode(500, new { success = false, message = "Config path is null." });

                _sdk.Config.Save(TTSConfig.LoadedPath);
                await _sdk.ReloadConfigAsync();
                return Ok(new { success = true, message = "Changes saved and reloaded." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { success = false, message = ex.Message });
            }
        }

        // =============================================================
        // Filesystem Browse & Converter Endpoints
        // =============================================================

        /// <summary>
        /// 浏览文件系统，列出文件和子目录。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("fs/browse")]
        public IActionResult BrowseFileSystem([FromQuery] string? path = null, [FromQuery] string? pattern = null)
        {
            try
            {
                var dir = string.IsNullOrWhiteSpace(path)
                    ? (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows) ? Environment.GetFolderPath(Environment.SpecialFolder.Desktop) : "/")
                    : path;

                // If default desktop doesn't exist (e.g., some Windows Server or headless), fallback to C: or /
                if (!Directory.Exists(dir))
                {
                    dir = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows) ? "C:\\" : "/";
                }

                if (!Directory.Exists(dir))
                    return BadRequest(new { error = "Directory does not exist", path = dir });

                var dirInfo = new DirectoryInfo(dir);
                var entries = new List<object>();

                // Subdirectories
                foreach (var d in dirInfo.GetDirectories().OrderBy(d => d.Name))
                {
                    try { entries.Add(new { name = d.Name, isDir = true, size = (long?)null }); }
                    catch { /* skip inaccessible */ }
                }

                // Files (optionally filtered)
                var files = string.IsNullOrWhiteSpace(pattern)
                    ? dirInfo.GetFiles()
                    : dirInfo.GetFiles(pattern);

                foreach (var f in files.OrderBy(f => f.Name))
                {
                    try { entries.Add(new { name = f.Name, isDir = false, size = (long?)f.Length }); }
                    catch { /* skip inaccessible */ }
                }

                // Drive list for root navigation
                var drives = DriveInfo.GetDrives()
                    .Where(d => d.IsReady)
                    .Select(d => d.Name)
                    .ToArray();

                return Ok(new
                {
                    current = dirInfo.FullName,
                    parent = dirInfo.Parent?.FullName,
                    drives,
                    entries
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = ex.Message });
            }
        }

        // OpenFolder API has been removed as it is unnecessary for headless/remote server usage
        /// <summary>
        /// 获取模型转换器状态
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("converter/status")]
        public IActionResult GetConverterStatus()
        {
            var converterDir = Path.GetFullPath(Path.Combine(_sdk.Config.ResourcesDir, "..", "tools", "converter"));
            string pythonPath;
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
            {
                pythonPath = Path.Combine(converterDir, "runtime", "python.exe");
            }
            else
            {
                // In Docker/Linux, we use the system python3
                // We fake the existence here, actual existence checks happen in ConvertModel using 'which' or direct invocation
                pythonPath = "/opt/venv/bin/python3"; // Or simply "python3"
            }

            var scriptPath = Path.Combine(converterDir, "v1_converter.py");
            var shellsDir = Path.Combine(converterDir, "templates");

            bool isPythonAvailable = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows)
                ? System.IO.File.Exists(pythonPath)
                : true; // In the Docker container, we assume it's installed via Dockerfile

            return Ok(new
            {
                available = isPythonAvailable && System.IO.File.Exists(scriptPath) && Directory.Exists(shellsDir),
                pythonPath,
                scriptPath,
                shellsDir,
                converterDir
            });
        }

        [HttpPost("fs/copy-file")]
        public IActionResult CopyFile([FromQuery] string sourcePath, [FromQuery] string targetDir)
        {
            try
            {
                if (!System.IO.File.Exists(sourcePath))
                    return BadRequest("Source file does not exist.");

                // 处理相对路径：如果以 resources/ 开头，则映射到 SDK 配置的物理目录
                string physicalTargetDir;
                if (!Path.IsPathRooted(targetDir))
                {
                    if (targetDir.Replace('\\', '/').StartsWith("resources/", StringComparison.OrdinalIgnoreCase))
                    {
                        // 去掉开头的 resources/，与 _sdk.Config.ResourcesDir 结合
                        var relativeToResources = targetDir.Substring("resources/".Length).TrimStart('\\', '/');
                        physicalTargetDir = Path.Combine(_sdk.Config.ResourcesDir, relativeToResources);
                    }
                    else
                    {
                        // 其它相对路径默认相对于程序运行目录的父级（如果是 bin/Debug/... 下运行）
                        physicalTargetDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, targetDir));
                    }
                }
                else
                {
                    physicalTargetDir = targetDir;
                }

                if (!Directory.Exists(physicalTargetDir))
                    Directory.CreateDirectory(physicalTargetDir);

                var fileName = Path.GetFileName(sourcePath);
                var destPath = Path.Combine(physicalTargetDir, fileName);

                System.IO.File.Copy(sourcePath, destPath, true);

                return Ok(new { success = true, fileName, destPath });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = ex.Message });
            }
        }


        /// <summary>
        /// 启动模型转换进程，通过 SSE 流式推送日志。
        /// </summary>
        [ApiExplorerSettings(IgnoreApi = true)]
        [HttpGet("converter/run")]
        public async Task ConverterRun(
            [FromQuery] string ckpt,
            [FromQuery] string pth,
            [FromQuery] string avatarId,
            [FromQuery] bool simplify = false,
            [FromQuery] bool quantize = false,
            [FromQuery] bool clean = true)
        {
            Response.ContentType = "text/event-stream";
            Response.Headers.Append("Cache-Control", "no-cache");
            Response.Headers.Append("X-Accel-Buffering", "no");

            async Task SendEvent(string type, string data)
            {
                await Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(new { type, data })}\n\n");
                await Response.Body.FlushAsync();
            }

            try
            {
                // Resolve paths
                var converterDir = Path.GetFullPath(Path.Combine(_sdk.Config.ResourcesDir, "..", "tools", "converter"));
                string pythonPath;
                if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
                {
                    pythonPath = Path.Combine(converterDir, "runtime", "python.exe");
                    if (!System.IO.File.Exists(pythonPath))
                    {
                        await SendEvent("error", "Windows Python runtime not found: " + pythonPath);
                        return;
                    }
                }
                else
                {
                    pythonPath = "/opt/venv/bin/python3"; // Or system "python3"
                }

                var scriptPath = Path.Combine(converterDir, "v1_converter.py");
                var shellsDir = Path.Combine(converterDir, "templates");
                var outDir = Path.Combine(_sdk.Config.ResourcesDir, "models_v1", avatarId);

                // Build arguments
                var args = $"\"{scriptPath}\" --ckpt \"{ckpt}\" --pth \"{pth}\" --shells \"{shellsDir}\" --out \"{outDir}\"";
                if (simplify) args += " --simplify";
                if (quantize) args += " --quantize";
                if (clean) args += " --clean";

                await SendEvent("info", $"Starting conversion for avatar '{avatarId}'...");
                await SendEvent("info", $"Output: {outDir}");
                await SendEvent("info", $"Command: python {Path.GetFileName(scriptPath)} ...");

                var psi = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = args,
                    WorkingDirectory = converterDir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8
                };
                // Ensure Python output is unbuffered and uses UTF-8
                psi.Environment["PYTHONUNBUFFERED"] = "1";
                psi.Environment["PYTHONIOENCODING"] = "utf-8";

                using var process = Process.Start(psi);
                if (process == null)
                {
                    await SendEvent("error", "Failed to start converter process.");
                    return;
                }

                // Read stdout and stderr concurrently
                var stdoutTask = Task.Run(async () =>
                {
                    while (await process.StandardOutput.ReadLineAsync() is { } line)
                    {
                        await SendEvent("log", line);
                    }
                });

                var stderrTask = Task.Run(async () =>
                {
                    while (await process.StandardError.ReadLineAsync() is { } line)
                    {
                        await SendEvent("warn", line);
                    }
                });

                await Task.WhenAll(stdoutTask, stderrTask);
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    // Auto-create references directory
                    var refsDir = Path.Combine(_sdk.Config.AvatarsDir, avatarId, "references");
                    Directory.CreateDirectory(refsDir);

                    await SendEvent("success", $"Conversion completed successfully! Exit code: {process.ExitCode}");
                    await SendEvent("refs_dir", refsDir);
                }
                else
                {
                    await SendEvent("error", $"Conversion failed with exit code: {process.ExitCode}");
                }

                await SendEvent("done", "Process finished.");
            }
            catch (Exception ex)
            {
                await SendEvent("error", $"Exception: {ex.Message}");
                await SendEvent("done", "Process finished with error.");
            }
        }
    }

}

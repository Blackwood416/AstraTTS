using Microsoft.AspNetCore.Mvc;
using AstraTTS.Core.Core;
using AstraTTS.Core.Utils;
using AstraTTS.Web.Models;
using AstraTTS.Web.Services;

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
                TopK = request.TopK ?? _sdk.Config.TopK
            };

            var audio = await _sdk.PredictAsync(request.Text, options, request.AvatarId, request.ReferenceId);

            using var ms = new MemoryStream();
            AudioHelper.SaveWav(ms, audio, _sdk.SamplingRate);

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
        /// <param name="chunkTokens">流式分块 Token 数（可选）</param>
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
            [FromQuery] int? chunkTokens = null)
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
                StreamingChunkTokens = chunkTokens ?? _sdk.Config.StreamingChunkTokens
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
                    request.StreamingChunkTokens ?? _sdk.Config.StreamingChunkTokens
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
                StreamingChunkTokens = session.StreamingChunkTokens
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
            return Ok(_sdk.Avatars.Select(a => new { a.Id, a.Name, a.Description, References = a.References.Select(r => new { r.Id, r.Name }) }));
        }

        /// <summary>
        /// 热重载配置。
        /// </summary>
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
    }
}

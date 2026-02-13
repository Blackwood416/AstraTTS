using System.Net;
using System.Text.Json;

namespace AstraTTS.Web.Middleware
{
    /// <summary>
    /// TTS 异常处理中间件，将核心层抛出的语种限制等异常映射为 HTTP 400。
    /// </summary>
    public class TtsExceptionMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<TtsExceptionMiddleware> _logger;

        public TtsExceptionMiddleware(RequestDelegate next, ILogger<TtsExceptionMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                await _next(context);
            }
            catch (InvalidOperationException ex)
            {
                // 核心层 MixedLanguageG2P 抛出的语种限制异常
                _logger.LogWarning($"[AstraTTS.Web] InvalidOperation: {ex.Message}");
                await HandleExceptionAsync(context, ex, HttpStatusCode.BadRequest);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[AstraTTS.Web] Unhandled Exception");
                await HandleExceptionAsync(context, ex, HttpStatusCode.InternalServerError);
            }
        }

        private static async Task HandleExceptionAsync(HttpContext context, Exception exception, HttpStatusCode code)
        {
            context.Response.ContentType = "application/json";
            context.Response.StatusCode = (int)code;

            var result = JsonSerializer.Serialize(new
            {
                success = false,
                message = exception.Message,
                type = exception.GetType().Name
            });

            await context.Response.WriteAsync(result);
        }
    }
}

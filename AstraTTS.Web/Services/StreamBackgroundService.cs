using AstraTTS.Core.Core;

namespace AstraTTS.Web.Services
{
    public class StreamBackgroundService : BackgroundService
    {
        private readonly StreamSessionManager _sessionManager;
        private readonly AstraTtsSdk _sdk;
        private readonly ILogger<StreamBackgroundService> _logger;

        public StreamBackgroundService(StreamSessionManager sessionManager, AstraTtsSdk sdk, ILogger<StreamBackgroundService> logger)
        {
            _sessionManager = sessionManager;
            _sdk = sdk;
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(1000, stoppingToken);

                    _sessionManager.CleanupOldSessions(TimeSpan.FromMinutes(5));
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in StreamBackgroundService");
                }
            }
        }
    }
}

using AstraTTS.Core.Core;
using Microsoft.Extensions.Hosting;

namespace AstraTTS.Web
{
    public class TtsWarmupService : IHostedService
    {
        private readonly AstraTtsSdk _sdk;

        public TtsWarmupService(AstraTtsSdk sdk)
        {
            _sdk = sdk;
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("[AstraTTS] Starting SDK Initialization...");
            await _sdk.InitializeAsync();
            Console.WriteLine("[AstraTTS] SDK Initialized Successfully.");
        }

        public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    }
}

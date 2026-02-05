using AstraTTS.Core.Core;
using AstraTTS.Core.Config;
using AstraTTS.Web;
using AstraTTS.Web.Services;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);

// Parse config path from args
string defaultConfig = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config.json");
string configPath = File.Exists("config.json") ? Path.GetFullPath("config.json") : defaultConfig;
for (int i = 0; i < args.Length; i++)
{
    string arg = args[i];
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
                    else
                    {
                        Console.WriteLine("Error: Missing value for config flag.");
                        return;
                    }
                }
                configPath = value;
                break;
                // 其他未知参数忽略，因为可能是 ASP.NET Core 的参数 (如 --urls, --environment)
        }
    }
}

Console.WriteLine($"[AstraTTS.Web] Loading config from: {configPath}");
var config = TTSConfig.LoadOrCreate(configPath);

// Add services to the container.
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.WriteIndented = true;
    });

// Register StreamSessionManager as Singleton
builder.Services.AddSingleton<StreamSessionManager>();

// Register AstraTTS.Core SDK as Singleton
builder.Services.AddSingleton<AstraTtsSdk>(sp =>
{
    var sdk = new AstraTtsSdk(config);
    // 异步初始化 (在后台运行，或由应用启动逻辑等待)
    // 注意：在 ASP.NET Core 中通常推荐在 Startup Filter 或 HostedService 中执行重量级异步初始化
    // 这里简单起见，我们确保第一次请求前初始化完成
    return sdk;
});

// 使用 HostedService 进行预热初始化
builder.Services.AddHostedService<TtsWarmupService>();
builder.Services.AddHostedService<StreamBackgroundService>();

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
app.UseStaticFiles();
app.MapOpenApi();
app.MapScalarApiReference(); // Mapping Scalar at /scalar
app.MapControllers();

app.Run();

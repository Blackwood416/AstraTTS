using System.Diagnostics;
using AstraTTS.Core.Config;
using AstraTTS.Core.Core;
using AstraTTS.Core.Utils;

namespace AstraTTSEmotionEval
{
    /// <summary>
    /// 情感注入端到端验证工具：
    ///   1) τ=0 (neutral/α=0) 应与基线一致
    ///   2) 随机 τ 应改变输出
    ///   3) α 网格扫描：生成音频 + 时长/F0 统计
    ///
    /// 用法:
    ///   dotnet run --project tools/emotion_eval -- --out E:\...\eval --emotions happy,angry ...
    /// </summary>
    internal static class Program
    {
        private static async Task<int> Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.InputEncoding = System.Text.Encoding.UTF8;

            var outDir = GetArg(args, "--out") ?? @"E:\AI\AstraTTS\_emotion_eval_cli";
            var text = GetArg(args, "--text") ?? "今天天气真好，我们一起去公园散步吧。";
            var avatarId = GetArg(args, "--avatar") ?? "chenxing";
            var emotions = (GetArg(args, "--emotions") ?? "neutral,happy,angry,sad,surprise,fearful,playfulness").Split(',');
            var alphas = (GetArg(args, "--alphas") ?? "0.1,0.3,0.5").Split(',');
            int repeats = int.TryParse(GetArg(args, "--repeats") ?? "1", out var rp) ? rp : 1;
            var configPath = GetArg(args, "--config");

            System.IO.Directory.CreateDirectory(outDir);

            var config = configPath != null ? TTSConfig.Load(configPath) : TTSConfig.LoadOrCreate(
                System.IO.Path.Combine(AppContext.BaseDirectory, "config.yaml"));

            // 强制启用情感控制（实验用）
            config.UseEmotionControl = true;
            if (GetArg(args, "--resources") is { } resDir) config.ResourcesDir = resDir;
            if (GetArg(args, "--tau") is { } tauDir) config.EmotionVectorPath = tauDir;

            Console.WriteLine($"ResourcesDir : {config.ResourcesDir}");
            Console.WriteLine($"EmotionPath  : {config.EmotionVectorFullPath}");
            Console.WriteLine($"Avatar       : {avatarId}");
            Console.WriteLine($"Text         : {text}");
            Console.WriteLine();

            using var sdk = new AstraTtsSdk(config);
            var sw = Stopwatch.StartNew();
            await sdk.InitializeAsync();
            Console.WriteLine($"SDK initialized in {sw.ElapsedMilliseconds} ms (sampling={sdk.SamplingRate})");

            var results = new List<(string emotion, float alpha, string file, int samples, double ms)>();

            async Task RunOne(string emotion, float? alpha, int rep = 0)
            {
                var opt = new TtsOptions
                {
                    EmotionId = emotion == "neutral" ? null : emotion,
                    EmotionAlpha = alpha
                };
                var sw2 = Stopwatch.StartNew();
                var r = await sdk.PredictAsync(text, opt, avatarId, null);
                sw2.Stop();
                var tag = alpha.HasValue ? $"{emotion}_a{alpha.Value:0.##}" : emotion;
                if (rep > 0) tag += $"_r{rep}";
                var file = System.IO.Path.Combine(outDir, $"{tag}.wav");
                AudioHelper.SaveWav(file, r.Audio, sdk.SamplingRate);
                double dur = r.Audio.Length / (double)sdk.SamplingRate;
                Console.WriteLine($"  {tag,-22} samples={r.Audio.Length,7} dur={dur:0.000}s tokens={r.TokenCount,4} " +
                                  $"wall={sw2.ElapsedMilliseconds,6}ms");
                results.Add((emotion, alpha ?? -1f, file, r.Audio.Length, sw2.ElapsedMilliseconds));
            }

            // 1. 基线（无情感）
            Console.WriteLine("== baseline (no injection) ==");
            await RunOne("neutral", null);
            for (int i = 1; i <= repeats; i++) await RunOne("neutral", null, i);

            // 2. 情感 × α 网格
            foreach (var emo in emotions)
            {
                if (emo.Trim() == "neutral") continue;
                Console.WriteLine($"== {emo} ==");
                foreach (var a in alphas)
                {
                    if (!float.TryParse(a, System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out var af)) continue;
                    await RunOne(emo.Trim(), af);
                    for (int i = 1; i <= repeats; i++) await RunOne(emo.Trim(), af, i);
                }
            }

            // 3. α=0 一致性测试
            Console.WriteLine("== alpha=0 (should match no-injection path) ==");
            await RunOne("happy", 0f);

            // summary
            var csv = System.IO.Path.Combine(outDir, "summary.csv");
            using (var w = new StreamWriter(csv, false, System.Text.Encoding.UTF8))
            {
                w.WriteLine("emotion,alpha,file,samples,ms");
                foreach (var r in results)
                    w.WriteLine($"{r.emotion},{r.alpha},{r.file},{r.samples},{r.ms}");
            }
            Console.WriteLine($"\nSaved {results.Count} samples to {outDir}");
            Console.WriteLine($"Summary: {csv}");
            return 0;
        }

        private static string? GetArg(string[] args, string name)
        {
            for (int i = 0; i < args.Length - 1; i++)
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
            return null;
        }
    }
}

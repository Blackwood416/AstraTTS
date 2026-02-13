using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

// ============================================================
// AstraTTS 不复用内存模式测试脚本 (C# .NET 10)
// ============================================================
// 场景: ReuseMemory = false
// 每个并发请求将使用独立的引擎实例（由于并发，这会触发多个模型副本的加载）
// 注意: 请监控显存/内存变化，可能会迅速占满
// ============================================================

string url = "http://localhost:5000/api/tts/predict";
int concurrency = 4; // 不复用模式下建议先从较小并发开始测试

Console.WriteLine("--- AstraTTS Independent Memory Test ---");
Console.WriteLine($"Target: {url}");
Console.WriteLine($"Concurrency: {concurrency}");
Console.WriteLine("Warning: ReuseMemory is assumed to be FALSE. Monitor your RAM/VRAM!");
Console.WriteLine("-----------------------------------------");

using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
var tasks = new List<Task>();
var swTotal = Stopwatch.StartNew();

for (int i = 0; i < concurrency; i++)
{
    int id = i + 1;
    tasks.Add(Task.Run(async () =>
    {
        var sw = Stopwatch.StartNew();
        try
        {
            string jsonBody = $"{{\"text\": \"这是第 {id} 路独立内存模式的推理请求。每个实例都拥有完整的模型副本。\", \"avatarId\": \"default\", \"languages\": [\"zh\"]}}";
            var content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

            var response = await httpClient.PostAsync(url, content);

            if (response.IsSuccessStatusCode)
            {
                byte[] audioData = await response.Content.ReadAsByteArrayAsync();
                await File.WriteAllBytesAsync($"test_independent_{id}.wav", audioData);

                sw.Stop();
                Console.WriteLine($"[Instance {id}] Success - Time: {sw.ElapsedMilliseconds}ms");
            }
            else
            {
                Console.WriteLine($"[Instance {id}] Failed - Status: {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Instance {id}] Error: {ex.Message}");
        }
    }));
}

await Task.WhenAll(tasks);
swTotal.Stop();

Console.WriteLine("-----------------------------------------");
Console.WriteLine($"Total Time: {swTotal.ElapsedMilliseconds}ms");

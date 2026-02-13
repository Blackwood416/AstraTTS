using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

// ============================================================
// AstraTTS 核心并发能力验证脚本 (C# .NET 10)
// ============================================================
// 1. 全量合成测试 (POST /api/tts/predict)
// 2. 直连流式测试 (GET /api/tts/predict-stream)
// 3. 会话流式测试 (POST /api/tts/stream/create -> GET /api/tts/stream/{id})
// ============================================================

string baseUrl = "http://localhost:5000/api/tts";

var testCases = new[]
{
    // 场景 1: 同 Avatar (default) 不同参考音频
    new { Name = "Default_Normal", Avatar = "default", Ref = "normal", Text = "你好，这是使用默认音色和正常风格的并发请求。", Langs = "[\"zh\"]" },
    new { Name = "Default_Smile", Avatar = "default", Ref = "smile", Text = "你好，这是使用默认音色和轻笑风格的并发请求。", Langs = "[\"zh\"]" },
    
    // 场景 2: 不同 Avatar，不同语言
    new { Name = "Mike_Default", Avatar = "mike", Ref = "default", Text = "I am Mike, testing concurrency and resource allocation.", Langs = "[\"en\"]" },
    new { Name = "Aima_Default", Avatar = "aima", Ref = "default", Text = "こんにちは、アイマです。", Langs = "[\"ja\"]" }
};

Console.WriteLine("--- AstraTTS Concurrency & Session Test ---");
Console.WriteLine($"Base URL: {baseUrl}");
Console.WriteLine("-------------------------------------------");

using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };

// --- 场景 1: 全量合成 (predict) ---
Console.WriteLine("\n[Scenario 1] Full Synthesis (POST /predict)");
var predictTasks = new List<Task>();
Stopwatch swS1 = Stopwatch.StartNew();

foreach (var test in testCases)
{
    predictTasks.Add(Task.Run(async () =>
    {
        var sw = Stopwatch.StartNew();
        try
        {
            // 精确匹配 TtsRequest 的 JSON 结构 (camelCase)
            string jsonBody = $"{{\"text\": \"{test.Text}\", \"avatarId\": \"{test.Avatar}\", \"referenceId\": \"{test.Ref}\", \"languages\": {test.Langs}}}";
            var content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

            var response = await httpClient.PostAsync($"{baseUrl}/predict", content);
            if (response.IsSuccessStatusCode)
            {
                byte[] data = await response.Content.ReadAsByteArrayAsync();
                await File.WriteAllBytesAsync($"test_full_{test.Name}.wav", data);
                Console.WriteLine($"[FullSucceed] {test.Name} - {sw.ElapsedMilliseconds}ms");
            }
            else
            {
                Console.WriteLine($"[FullFailed] {test.Name} - Status: {response.StatusCode}");
            }
        }
        catch (Exception ex) { Console.WriteLine($"[Error] {test.Name}: {ex.Message}"); }
    }));
}
await Task.WhenAll(predictTasks);
Console.WriteLine($"Scenario 1 Finished. Total: {swS1.ElapsedMilliseconds}ms");

// --- 场景 2: 直连流式 (predict-stream) ---
Console.WriteLine("\n[Scenario 2] Direct Stream (GET /predict-stream)");
var streamTasks = new List<Task>();
Stopwatch swS2 = Stopwatch.StartNew();

foreach (var test in testCases)
{
    streamTasks.Add(Task.Run(async () =>
    {
        var sw = Stopwatch.StartNew();
        try
        {
            // GET 参数拼接 (注意 URL 编码)
            string langsParam = test.Langs.Replace("[", "").Replace("]", "").Replace("\"", ""); // ja, en etc.
            string query = $"?text={Uri.EscapeDataString(test.Text)}&avatarId={test.Avatar}&referenceId={test.Ref}&languages={langsParam}";
            var response = await httpClient.GetAsync($"{baseUrl}/predict-stream{query}", HttpCompletionOption.ResponseHeadersRead);

            if (response.IsSuccessStatusCode)
            {
                using var stream = await response.Content.ReadAsStreamAsync();
                int totalBytes = 0;
                byte[] buffer = new byte[8192];
                int read;
                while ((read = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    totalBytes += read;
                    // 读取到第一块数据即视为响应开始 (First Byte)
                    if (totalBytes == read) Console.WriteLine($"[StreamStarted] {test.Name} - First Chunk: {sw.ElapsedMilliseconds}ms");
                }
                Console.WriteLine($"[StreamFinished] {test.Name} - Total: {sw.ElapsedMilliseconds}ms, Bytes: {totalBytes}");
            }
        }
        catch (Exception ex) { Console.WriteLine($"[Error] {test.Name}: {ex.Message}"); }
    }));
}
await Task.WhenAll(streamTasks);
Console.WriteLine($"Scenario 2 Finished. Total: {swS2.ElapsedMilliseconds}ms");

Console.WriteLine("\n-------------------------------------------");
Console.WriteLine("All tests finished. Check wav files and console output.");

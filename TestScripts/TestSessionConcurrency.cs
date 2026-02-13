using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

// ============================================================
// AstraTTS 会话级并发能力验证脚本 (C# .NET 10)
// ============================================================
// 场景: 创建 Session -> 获取流数据
// 这种模式最能体现“会话调度”的真实工作状态
// ============================================================

string baseUrl = "http://localhost:5000/api/tts";
int concurrency = 4;

var testSentences = new[]
{
    "正在通过会话模式进行并发测试，这是第一路流。",
    "会话调度器正在为这一路请求分配推理实例，请稍候。",
    "系统支持多个会话并行处理，当前正在进行压力验证。",
    "最后一路请求正在执行，即将完成本次并发会话测试。"
};

Console.WriteLine("--- AstraTTS Session-Based Concurrency Test ---");
Console.WriteLine($"Base URL: {baseUrl}");
Console.WriteLine("-----------------------------------------------");

using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
var tasks = new List<Task>();
var swTotal = Stopwatch.StartNew();

for (int i = 0; i < concurrency; i++)
{
    int id = i + 1;
    string text = testSentences[i % testSentences.Length];

    tasks.Add(Task.Run(async () =>
    {
        var sw = Stopwatch.StartNew();
        try
        {
            // 步骤 1: 创建会话 (POST /stream/create)
            string createUrl = $"{baseUrl}/stream/create";
            string jsonBody = $"{{\"text\": \"{text}\", \"avatarId\": \"default\", \"languages\": [\"zh\"]}}";
            var content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

            var createResp = await httpClient.PostAsync(createUrl, content);
            if (!createResp.IsSuccessStatusCode)
            {
                Console.WriteLine($"[SessionCreateFailed] {id} - Status: {createResp.StatusCode}");
                return;
            }

            var createResultJson = await createResp.Content.ReadAsStringAsync();
            // 手动解析 sessionId (简单处理，避免引用过多库)
            string sessionId = ExtractJsonValue(createResultJson, "sessionId");
            string streamUrl = $"{baseUrl}/stream/{sessionId}";

            Console.WriteLine($"[SessionCreated] {id} - ID: {sessionId}, Time: {sw.ElapsedMilliseconds}ms");

            // 步骤 2: 拉取音频流 (GET /stream/{sessionId})
            var streamResp = await httpClient.GetAsync(streamUrl, HttpCompletionOption.ResponseHeadersRead);
            if (streamResp.IsSuccessStatusCode)
            {
                using var stream = await streamResp.Content.ReadAsStreamAsync();
                int totalBytes = 0;
                byte[] buffer = new byte[8192];
                int read;
                while ((read = await stream.ReadAsync(buffer, 0, buffer.Length)) > 0)
                {
                    totalBytes += read;
                    if (totalBytes == read) Console.WriteLine($"[DataStarted] {id} - First Byte: {sw.ElapsedMilliseconds}ms");
                }
                Console.WriteLine($"[SessionFinished] {id} - Total: {sw.ElapsedMilliseconds}ms, Bytes: {totalBytes}");
            }
        }
        catch (Exception ex) { Console.WriteLine($"[Error] {id}: {ex.Message}"); }
    }));
}

await Task.WhenAll(tasks);
swTotal.Stop();

Console.WriteLine("-----------------------------------------------");
Console.WriteLine($"Test Finished. Total Time: {swTotal.ElapsedMilliseconds}ms");

// 简单的 JSON 字段提取工具
string ExtractJsonValue(string json, string key)
{
    int keyIdx = json.IndexOf($"\"{key}\"");
    if (keyIdx == -1) return "";
    int start = json.IndexOf("\"", keyIdx + key.Length + 2) + 1;
    int end = json.IndexOf("\"", start);
    return json.Substring(start, end - start);
}

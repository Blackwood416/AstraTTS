using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using SharpOpenJTalk.Lang;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 日语 G2P 处理器，使用 SharpOpenJTalk.Lang 提取音素。
    /// 注意：OpenJTalk 不是线程安全的，所有 API 调用都需要同步。
    /// </summary>
    public class JapaneseG2P : IG2P, IDisposable
    {
        private readonly OpenJTalkAPI _api;
        private readonly string _dictPath;
        private bool _initialized;
        private bool _disposed;

        // OpenJTalk 不是线程安全的，需要使用锁来同步所有 API 调用
        private static readonly object _lock = new object();

        // HTS 标签格式: xx^sil-k+o=N/A:...
        // 提取中间的音素 (在 - 和 + 之间)
        private static readonly Regex PhonemeExtractRegex = new Regex(@"-([a-zA-Z]+)\+", RegexOptions.Compiled);

        // 日语音素到 Symbols 的映射
        // 注意：OpenJTalk 输出的音素大小写敏感
        private static readonly Dictionary<string, string> PhonemeMapping = new Dictionary<string, string>()
        {
            // 特殊音素
            { "sil", "SP" },   // 静音
            { "pau", "," },    // 停顿 (修改为逗号以获得更好的稳定性)
            { "cl", "cl" },    // 促音
            
            // 元音 (小写)
            { "a", "a" }, { "i", "i" }, { "u", "u" }, { "e", "e" }, { "o", "o" },
            
            // 特殊元音 (大写 - OpenJTalk 有时输出大写)
            { "A", "a" }, { "I", "I" }, { "U", "U" }, { "E", "e" }, { "O", "o" },
            
            // 鼻音
            { "N", "N" }, { "n", "n" }, { "m", "m" }, { "ny", "ny" }, { "my", "my" },
            
            // 辅音
            { "k", "k" }, { "ky", "ky" },
            { "g", "g" }, { "gy", "gy" },
            { "s", "s" }, { "sh", "sh" },
            { "z", "z" }, { "j", "j" },
            { "t", "t" }, { "ts", "ts" }, { "ch", "ch" },
            { "d", "d" }, { "dy", "dy" },
            { "h", "h" }, { "hy", "hy" }, { "f", "f" },
            { "b", "b" }, { "by", "by" },
            { "p", "p" }, { "py", "py" },
            { "r", "r" }, { "ry", "ry" },
            { "w", "w" }, { "y", "y" }, { "v", "v" },
        };

        public JapaneseG2P(string dictPath, string? userDictPath = null)
        {
            _api = new OpenJTalkAPI();
            _dictPath = dictPath;

            if (!Directory.Exists(dictPath))
            {
                Console.WriteLine($"[JapaneseG2P] Warning: Dictionary not found at {dictPath}");
                return;
            }

            // 在锁内初始化以确保线程安全
            lock (_lock)
            {
                _initialized = _api.Initialize(dictPath, userDictPath ?? string.Empty);
            }

            if (_initialized)
            {
                Console.WriteLine($"[JapaneseG2P] Initialized with dictionary: {dictPath}");
            }
            else
            {
                Console.WriteLine($"[JapaneseG2P] Failed to initialize OpenJTalk");
            }
        }

        public G2PResult Process(string text)
        {
            if (!_initialized || string.IsNullOrWhiteSpace(text))
            {
                return new G2PResult
                {
                    NormalizedText = text,
                    Phones = new List<string> { "SP" },
                    PhoneIds = new long[] { Symbols.GetId("SP") },
                    Word2Ph = new int[] { 1 }
                };
            }

            // 1. 文本规范化 - 暂时禁用以测试 OpenJTalk
            // string normalized = TextNorm.JapaneseTextNormalizer.Normalize(text);
            string normalized = text; // 直接使用原文测试
            Console.WriteLine($"[JapaneseG2P] Processing: '{text}' -> normalized: '{normalized}'");

            // 2. 获取 HTS 标签
            // 为每次调用创建新的 OpenJTalkAPI 实例以避免状态问题
            List<string>? labelList = null;
            lock (_lock)
            {
                try
                {
                    using var api = new OpenJTalkAPI();
                    if (!api.Initialize(_dictPath, string.Empty))
                    {
                        Console.WriteLine($"[JapaneseG2P] Failed to re-initialize OpenJTalk");
                        return CreateFallbackResult(normalized);
                    }

                    Console.WriteLine($"[JapaneseG2P] Calling GetLabels with fresh API...");
                    var labels = api.GetLabels(normalized);
                    Console.WriteLine($"[JapaneseG2P] GetLabels returned: {(labels == null ? "null" : "IEnumerable")}");

                    if (labels != null)
                    {
                        labelList = labels.ToList();
                        Console.WriteLine($"[JapaneseG2P] Materialized {labelList.Count} labels");
                    }

                    api.Clear();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[JapaneseG2P] Error getting labels: {ex.Message}");
                    Console.WriteLine($"[JapaneseG2P] Stack trace: {ex.StackTrace}");
                }
            }

            if (labelList == null || labelList.Count == 0)
            {
                Console.WriteLine($"[JapaneseG2P] Warning: GetLabels returned null or empty for: {normalized}");
                return new G2PResult
                {
                    NormalizedText = normalized,
                    Phones = new List<string> { "SP" },
                    PhoneIds = new long[] { Symbols.GetId("SP") },
                    Word2Ph = new int[] { 1 }
                };
            }

            // Debug: 输出原始标签
            Console.WriteLine($"[JapaneseG2P] Input: {normalized}, Labels count: {labelList.Count}");
            if (labelList.Count > 0 && labelList.Count <= 20)
            {
                Console.WriteLine($"[JapaneseG2P] Labels: {string.Join(", ", labelList.Take(10).Select(l => l.Length > 50 ? l.Substring(0, 50) + "..." : l))}");
            }

            // 3. 从 HTS 标签中提取音素
            var phones = new List<string>();
            var word2ph = new List<int>();

            foreach (var label in labelList)
            {
                var match = PhonemeExtractRegex.Match(label);
                if (match.Success)
                {
                    string rawPhoneme = match.Groups[1].Value;

                    // 跳过静音标记 (句首/句尾的 sil)
                    if (rawPhoneme.Equals("sil", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    // 映射到 Symbols
                    string mappedPhoneme;
                    if (PhonemeMapping.TryGetValue(rawPhoneme, out var mapped))
                    {
                        mappedPhoneme = mapped;
                    }
                    else
                    {
                        // 未知音素，直接使用
                        mappedPhoneme = rawPhoneme;
                        Console.WriteLine($"[JapaneseG2P] Unknown phoneme: {rawPhoneme}");
                    }

                    phones.Add(mappedPhoneme);
                    word2ph.Add(1); // 日语每个音素对应一个"字"
                }
            }

            // 4. 确保音素列表不为空
            if (phones.Count == 0)
            {
                phones.Add("SP");
                word2ph.Add(1);
            }

            var phoneIds = Symbols.GetIds(phones);

            // Debug: 输出最终音素和 ID
            Console.WriteLine($"[JapaneseG2P] Final phones ({phones.Count}): {string.Join(", ", phones)}");
            Console.WriteLine($"[JapaneseG2P] Final IDs ({phoneIds.Length}): {string.Join(", ", phoneIds)}");

            return new G2PResult
            {
                NormalizedText = normalized,
                Phones = phones,
                PhoneIds = phoneIds,
                Word2Ph = word2ph.ToArray()
            };
        }

        private G2PResult CreateFallbackResult(string normalized)
        {
            return new G2PResult
            {
                NormalizedText = normalized,
                Phones = new List<string> { "SP" },
                PhoneIds = new long[] { Symbols.GetId("SP") },
                Word2Ph = new int[] { 1 }
            };
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _api?.Dispose();
                _disposed = true;
            }
        }
    }
}

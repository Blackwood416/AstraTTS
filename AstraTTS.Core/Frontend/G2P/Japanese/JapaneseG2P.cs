using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AstraTTS.Core.Frontend.TextNorm;
using SharpOpenJTalk.Lang;
using AstraTTS.Core.Frontend.G2P.Common;
using AstraTTS.Core.Core;

namespace AstraTTS.Core.Frontend.G2P.Japanese
{
    /// <summary>
    /// 日语 G2P 处理器，使用 SharpOpenJTalk.Lang 提取音素。
    /// 注意：OpenJTalk 不是线程安全的，所有 API 调用都需要同步。
    /// </summary>
    public class JapaneseG2P : IG2P, IDisposable
    {
        private readonly OpenJTalkAPI _api;
        private readonly string _dictPath;
        private readonly Dictionary<string, string[]> _customDict;
        private bool _initialized;
        private bool _disposed;
        public bool DebugMode { get; set; } = false;

        // OpenJTalk 不是线程安全的，需要使用锁来同步所有 API 调用
        private static readonly object _lock = new object();

        // 缓存频繁出现的短语或词组，避免重复调用昂贵的 OpenJTalk API
        private static readonly Dictionary<string, G2PResult> _cache = new Dictionary<string, G2PResult>(512);
        private const int MaxCacheSize = 512;

        // 日语音素到 Symbols 的映射
        private static readonly Dictionary<string, string> PhonemeMapping = new Dictionary<string, string>()
        {
            { "sil", "SP" }, { "pau", "," }, { "cl", "cl" },
            { "a", "a" }, { "i", "i" }, { "u", "u" }, { "e", "e" }, { "o", "o" },
            { "A", "a" }, { "I", "i" }, { "U", "u" }, { "E", "e" }, { "O", "o" },
            { "N", "N" }, { "n", "n" }, { "m", "m" }, { "ny", "ny" }, { "my", "my" },
            { "k", "k" }, { "ky", "ky" }, { "g", "g" }, { "gy", "gy" },
            { "s", "s" }, { "sh", "sh" }, { "z", "z" }, { "j", "j" },
            { "t", "t" }, { "ts", "ts" }, { "ch", "ch" }, { "d", "d" }, { "dy", "dy" },
            { "h", "h" }, { "hy", "hy" }, { "f", "f" }, { "b", "b" }, { "by", "by" },
            { "p", "p" }, { "py", "py" }, { "r", "r" }, { "ry", "ry" },
            { "w", "w" }, { "y", "y" }, { "v", "v" },
        };

        public JapaneseG2P(string dictPath, string? customDictPath = null)
        {
            _api = new OpenJTalkAPI();
            _dictPath = dictPath;
            _customDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);

            if (!Directory.Exists(dictPath))
            {
                Console.WriteLine($"[JapaneseG2P] Warning: Dictionary not found at {dictPath}");
                return;
            }

            lock (_lock)
            {
                // Note: userDictPath is for OpenJTalk's internal binary user dict, we use our own text-based dict
                _initialized = _api.Initialize(dictPath, string.Empty);
            }

            if (_initialized)
            {
                Console.WriteLine($"[JapaneseG2P] Initialized with dictionary: {dictPath}");
            }
            else
            {
                Console.WriteLine($"[JapaneseG2P] Failed to initialize OpenJTalk");
            }

            LoadCustomDictionary(customDictPath);
        }

        private void LoadCustomDictionary(string? path)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

            try
            {
                int count = 0;
                foreach (var line in File.ReadAllLines(path))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                    var parts = line.Split('\t');
                    if (parts.Length >= 2)
                    {
                        string word = parts[0].Trim();
                        string[] phones = parts[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        _customDict[word] = phones;
                        count++;
                    }
                }
                if (count > 0)
                    Console.WriteLine($"[JapaneseG2P] Loaded {count} custom Japanese words from {path}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[JapaneseG2P] Error loading custom dictionary: {ex.Message}");
            }
        }

        public G2PResult Process(string text, int? priorityMode = null)
        {
            if (string.IsNullOrWhiteSpace(text))
                return CreateFallbackResult("");

            var normalized = JapaneseTextNormalizer.Normalize(text);

            // 1. 检查缓存
            lock (_lock)
            {
                if (_cache.TryGetValue(normalized, out var cached)) return cached;
            }

            // 2. 检查自定义词典 (精确匹配)
            if (_customDict.TryGetValue(normalized, out var customPhones))
            {
                var customResult = new G2PResult
                {
                    NormalizedText = normalized,
                    Phones = customPhones.ToList(),
                    PhoneIds = Symbols.GetIds(customPhones),
                    Word2Ph = Enumerable.Repeat(1, customPhones.Length).ToArray()
                };
                return customResult;
            }

            // 3. 获取 HTS 标签
            List<string>? labelList = null;
            lock (_lock)
            {
                try
                {
                    if (_api == null) throw new InvalidOperationException("OpenJTalkAPI is not initialized.");
                    var labels = _api.GetLabels(normalized);
                    if (labels != null) labelList = labels.ToList();
                    else if (DebugMode) Console.WriteLine($"[JapaneseG2P] API returned NULL for input: '{normalized}'");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[JapaneseG2P Error] Extract labels failed: {ex.Message}");
                }
            }

            if (labelList == null || labelList.Count == 0) return CreateFallbackResult(normalized);

            // 3. 提取音素 (优化: 使用字符串操作代替 Regex)
            var phones = new List<string>(labelList.Count);
            foreach (var label in labelList)
            {
                int start = label.IndexOf('-');
                int end = label.IndexOf('+', start + 1);
                if (start >= 0 && end > start)
                {
                    string rawPhoneme = label.Substring(start + 1, end - start - 1);
                    if (PhonemeMapping.TryGetValue(rawPhoneme, out string? mapped)) phones.Add(mapped);
                    else if (DebugMode) Console.WriteLine($"[JapaneseG2P] Warning: Unknown phoneme '{rawPhoneme}'");
                }
            }

            if (phones.Count == 0) phones.Add("SP");

            // 过滤首尾多余的 SP
            while (phones.Count > 1 && phones[0] == "SP") phones.RemoveAt(0);
            while (phones.Count > 1 && phones[phones.Count - 1] == "SP") phones.RemoveAt(phones.Count - 1);
            if (phones.Count == 0) phones.Add("SP");

            var result = new G2PResult
            {
                NormalizedText = normalized,
                Phones = phones,
                PhoneIds = Symbols.GetIds(phones),
                Word2Ph = Enumerable.Repeat(1, phones.Count).ToArray()
            };

            // 4. 更新缓存
            lock (_lock)
            {
                if (_cache.Count >= MaxCacheSize) _cache.Remove(_cache.Keys.First());
                _cache[normalized] = result;

                if (DebugMode)
                {
                    Console.WriteLine($"[JapaneseG2P] Processed: '{normalized}', Phones: {string.Join(" ", phones)}");
                }
            }

            return result;
        }

        private G2PResult CreateFallbackResult(string normalized)
        {
            return new G2PResult
            {
                NormalizedText = normalized,
                Phones = new List<string> { "SP" },
                PhoneIds = Symbols.GetIds(new List<string> { "SP" }),
                Word2Ph = new[] { 1 }
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

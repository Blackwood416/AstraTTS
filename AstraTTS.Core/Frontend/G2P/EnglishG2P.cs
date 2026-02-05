using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AstraTTS.Core.Core;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 英文 G2P 处理器，使用 CMUDict 将英文单词转换为 ARPAbet 音素。
    /// </summary>
    public class EnglishG2P : IG2P
    {
        // Word -> Phonemes (e.g. "hello" -> ["HH", "AH0", "L", "OW1"])
        private readonly Dictionary<string, string[]> _cmuDict;

        // Neural network G2P for OOV words
        private readonly NeuralG2P? _neuralG2P;

        // 外部自定义词典 (用户配置)
        private readonly Dictionary<string, string[]> _customDict;

        // 手动发音词典 (处理特定专有名词或缩写)
        private static readonly Dictionary<string, string[]> _manualDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            // === AI/ML 相关 ===
            { "gpt", new[] { "JH", "IY1", "P", "IY1", "T", "IY1" } },
            { "openai", new[] { "OW1", "P", "AH0", "N", "EY0", "AY1" } },
            { "ai", new[] { "EY0", "AY1" } },
            { "api", new[] { "EY1", "P", "IY1", "AY1" } },
            { "llm", new[] { "EH1", "L", "EH1", "L", "EH1", "M" } },
            { "ml", new[] { "EH1", "M", "EH1", "L" } },
            
            // === 网络/协议 ===
            { "http", new[] { "EY1", "CH", "T", "IY1", "T", "IY1", "P", "IY1" } },
            { "https", new[] { "EY1", "CH", "T", "IY1", "T", "IY1", "P", "IY1", "EH1", "S" } },
            { "url", new[] { "Y", "UW1", "AA1", "R", "EH1", "L" } },
            { "ip", new[] { "AY1", "P", "IY1" } },
            { "tcp", new[] { "T", "IY1", "S", "IY1", "P", "IY1" } },
            { "udp", new[] { "Y", "UW1", "D", "IY1", "P", "IY1" } },
            { "dns", new[] { "D", "IY1", "EH1", "N", "EH1", "S" } },
            { "ssh", new[] { "EH1", "S", "EH1", "S", "EY1", "CH" } },
            { "ftp", new[] { "EH1", "F", "T", "IY1", "P", "IY1" } },
            
            // === 硬件 ===
            { "cpu", new[] { "S", "IY1", "P", "IY1", "Y", "UW1" } },
            { "gpu", new[] { "JH", "IY1", "P", "IY1", "Y", "UW1" } },
            { "ram", new[] { "R", "AE1", "M" } },
            { "rom", new[] { "R", "AA1", "M" } },
            { "ssd", new[] { "EH1", "S", "EH1", "S", "D", "IY1" } },
            { "hdd", new[] { "EY1", "CH", "D", "IY1", "D", "IY1" } },
            { "usb", new[] { "Y", "UW1", "EH1", "S", "B", "IY1" } },
            
            // === 编程/开发 ===
            { "sdk", new[] { "EH1", "S", "D", "IY1", "K", "EY1" } },
            { "ide", new[] { "AY1", "D", "IY1", "IY1" } },
            { "ui", new[] { "Y", "UW1", "AY1" } },
            { "ux", new[] { "Y", "UW1", "EH1", "K", "S" } },
            { "ios", new[] { "AY1", "OW1", "EH1", "S" } },
            { "css", new[] { "S", "IY1", "EH1", "S", "EH1", "S" } },
            { "html", new[] { "EY1", "CH", "T", "IY1", "EH1", "M", "EH1", "L" } },
            { "sql", new[] { "EH1", "S", "K", "Y", "UW1", "EH1", "L" } },
            { "json", new[] { "JH", "EY1", "S", "AA0", "N" } },
            { "xml", new[] { "EH1", "K", "S", "EH1", "M", "EH1", "L" } },
            
            // === 公司/品牌 ===
            { "google", new[] { "G", "UW1", "G", "AH0", "L" } },
            { "microsoft", new[] { "M", "AY1", "K", "R", "OW0", "S", "AO1", "F", "T" } },
            { "nvidia", new[] { "EH0", "N", "V", "IY1", "D", "IY0", "AH0" } },
            { "intel", new[] { "IH1", "N", "T", "EH0", "L" } },
            { "amd", new[] { "EY1", "EH1", "M", "D", "IY1" } },
            
            // === 编程语言/框架 ===
            { "python", new[] { "P", "AY1", "TH", "AA0", "N" } },
            { "java", new[] { "JH", "AA1", "V", "AH0" } },
            { "javascript", new[] { "JH", "AA1", "V", "AH0", "S", "K", "R", "IH1", "P", "T" } },
            { "typescript", new[] { "T", "AY1", "P", "S", "K", "R", "IH1", "P", "T" } },
            { "golang", new[] { "G", "OW1", "L", "AE1", "NG" } },
            { "rust", new[] { "R", "AH1", "S", "T" } },
            { "kotlin", new[] { "K", "AA1", "T", "L", "IH0", "N" } },
            { "swift", new[] { "S", "W", "IH1", "F", "T" } },
            { "ruby", new[] { "R", "UW1", "B", "IY0" } },
            { "php", new[] { "P", "IY1", "EY1", "CH", "P", "IY1" } },
            { "sharp", new[] { "SH", "AA1", "R", "P" } },  // for "C sharp"
            { "plus", new[] { "P", "L", "AH1", "S" } },    // for "C plus plus"
            { "dot", new[] { "D", "AA1", "T" } },         // for ".NET"
            { "net", new[] { "N", "EH1", "T" } },
            
            // === 常见技术词汇 ===
            { "email", new[] { "IY1", "M", "EY1", "L" } },
            { "wifi", new[] { "W", "AY1", "F", "AY1" } },
            { "bluetooth", new[] { "B", "L", "UW1", "T", "UW2", "TH" } },
            { "github", new[] { "G", "IH1", "T", "HH", "AH1", "B" } },
            { "gitlab", new[] { "G", "IH1", "T", "L", "AE1", "B" } },
            { "docker", new[] { "D", "AA1", "K", "ER0" } },
            { "kubernetes", new[] { "K", "UW0", "B", "ER0", "N", "EH1", "T", "IY0", "Z" } },
            { "linux", new[] { "L", "IH1", "N", "AH0", "K", "S" } },
            { "ubuntu", new[] { "UW0", "B", "UH1", "N", "T", "UW0" } },
            { "windows", new[] { "W", "IH1", "N", "D", "OW0", "Z" } },
            { "macos", new[] { "M", "AE1", "K", "OW1", "EH1", "S" } },
            { "android", new[] { "AE1", "N", "D", "R", "OY2", "D" } },
            { "iphone", new[] { "AY1", "F", "OW2", "N" } },
            { "ipad", new[] { "AY1", "P", "AE2", "D" } },
            
            // === 社交媒体/应用 ===
            { "tiktok", new[] { "T", "IH1", "K", "T", "AA1", "K" } },
            { "chatgpt", new[] { "CH", "AE1", "T", "JH", "IY1", "P", "IY1", "T", "IY1" } },
            { "youtube", new[] { "Y", "UW1", "T", "UW1", "B" } },
            { "twitter", new[] { "T", "W", "IH1", "T", "ER0" } },
            { "facebook", new[] { "F", "EY1", "S", "B", "UH2", "K" } },
            { "instagram", new[] { "IH1", "N", "S", "T", "AH0", "G", "R", "AE2", "M" } },
            { "whatsapp", new[] { "W", "AA1", "T", "S", "AE2", "P" } },
            { "telegram", new[] { "T", "EH1", "L", "AH0", "G", "R", "AE2", "M" } },
            { "discord", new[] { "D", "IH1", "S", "K", "AO1", "R", "D" } },
            { "slack", new[] { "S", "L", "AE1", "K" } },
            { "zoom", new[] { "Z", "UW1", "M" } },
            
            // === 单字母 (英文字母表发音) ===
            // A = "ay" 
            { "a", new[] { "EY1" } },
            // B = "bee"
            { "b", new[] { "B", "IY1" } },
            // C = "see"
            { "c", new[] { "S", "IY1" } },
            // D = "dee"
            { "d", new[] { "D", "IY1" } },
            // E = "ee"
            { "e", new[] { "IY1", "IY0", "SP"} },
            // F = "ef"
            { "f", new[] { "EH1", "F" } },
            // G = "jee"
            { "g", new[] { "JH", "IY1" } },
            // H = "aych"
            { "h", new[] { "EY1", "CH" } },
            // I = "ai"
            { "i", new[] { "AY1" } },
            // J = "jay"
            { "j", new[] { "JH", "EY1" } },
            // K = "kay"
            { "k", new[] { "K", "EY1" } },
            // L = "el"
            { "l", new[] { "EH1", "L" } },
            // M = "em"
            { "m", new[] { "EH1", "M" } },
            // N = "en"
            { "n", new[] { "EH1", "N" } },
            // O = "oh"
            { "o", new[] { "OW1", "SP" } },
            // P = "pee"
            { "p", new[] { "P", "IY1" } },
            // Q = "cue"
            { "q", new[] { "K", "Y", "Y", "UW1" } },
            // R = "ar"
            { "r", new[] { "AA1", "R" } },
            // S = "es"
            { "s", new[] { "EH1", "S" } },
            // T = "tee"
            { "t", new[] { "T", "IY1" } },
            // U = "you"
            { "u", new[] { "Y", "UW1" } },
            // V = "vee"
            { "v", new[] { "V", "IY1" } },
            // W = "double-you"
            { "w", new[] { "D", "AH1", "B", "L", "Y", "UW1" } },
            // X = "ex"
            { "x", new[] { "EH1", "K", "S" } },
            // Y = "why"
            { "y", new[] { "W", "AY1" } },
            // Z = "zee" - 使用 SP 分隔避免与后续内容连接
            { "z", new[] { "Z", "IY1", "SP" } }
        };

        public EnglishG2P(string cmuDictPath, string? neuralG2PModelPath = null, string? customDictPath = null)
        {
            _cmuDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            _customDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);

            // Initialize NeuralG2P if model path is provided
            if (!string.IsNullOrEmpty(neuralG2PModelPath) && File.Exists(neuralG2PModelPath))
            {
                _neuralG2P = new NeuralG2P();
                _neuralG2P.LoadModel(neuralG2PModelPath);
            }

            // Load CMUDict
            if (File.Exists(cmuDictPath))
            {
                Console.WriteLine($"[EnglishG2P] Loading CMUDict from {cmuDictPath}...");
                foreach (var line in File.ReadLines(cmuDictPath))
                {
                    if (line.StartsWith(";;;") || string.IsNullOrWhiteSpace(line)) continue;
                    var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2) continue;
                    string word = parts[0];
                    int parenIdx = word.IndexOf('(');
                    if (parenIdx > 0) word = word.Substring(0, parenIdx);
                    if (_cmuDict.ContainsKey(word)) continue;
                    _cmuDict[word] = parts.Skip(1).ToArray();
                }
                Console.WriteLine($"[EnglishG2P] Loaded {_cmuDict.Count} words from CMUDict.");
            }

            // Load Custom Dict
            if (!string.IsNullOrEmpty(customDictPath) && File.Exists(customDictPath))
            {
                Console.WriteLine($"[EnglishG2P] Loading custom dictionary from {customDictPath}...");
                foreach (var line in File.ReadAllLines(customDictPath))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                    // Format: word\tPHONEME1 PHONEME2
                    var parts = line.Split('\t');
                    if (parts.Length >= 2)
                    {
                        string word = parts[0].Trim();
                        string[] phonemes = parts[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        _customDict[word] = phonemes;
                    }
                }
                Console.WriteLine($"[EnglishG2P] Loaded {_customDict.Count} custom English words.");
            }
        }

        /// <summary>
        /// 处理纯英文文本。
        /// </summary>
        public G2PResult Process(string text)
        {
            // 0. 文本正规化 (数字 -> 英文)
            text = TextNorm.EnglishTextNormalizer.Normalize(text);

            // 0.5 保护手动发音词典中的词 (防止被驼峰分割逻辑拆开)
            // 将 manualDict 中的词在文本中替换为小写，这样 CamelCase 正则就不会匹配到它们
            foreach (var key in _manualDict.Keys)
            {
                // 使用不区分大小写的替换
                text = Regex.Replace(text, Regex.Escape(key), key.ToLowerInvariant(), RegexOptions.IgnoreCase);
            }

            // 1. 分割驼峰式复合词 (OpenAI -> Open AI, GPT4 -> GPT 4)
            // 注意：已经变成小写的词 (如 openai) 不会被此正则匹配
            text = Regex.Replace(text, @"([a-z])([A-Z])", "$1 $2");
            text = Regex.Replace(text, @"([A-Z]+)([A-Z][a-z])", "$1 $2");

            // 2. Normalize: lowercase, split by non-alphabetic
            text = text.ToLowerInvariant();
            var words = Regex.Split(text, @"([^a-z']+)");

            List<string> phones = new List<string>();
            List<int> word2ph = new List<int>();

            foreach (var word in words)
            {
                if (string.IsNullOrWhiteSpace(word)) continue;

                // Punctuation
                if (Symbols.Punctuation.Contains(word))
                {
                    phones.Add(word);
                    word2ph.Add(1);
                    continue;
                }

                // Non-alphabetic characters (spaces, etc.)
                if (!Regex.IsMatch(word, @"[a-z]"))
                {
                    // Skip or add as pause
                    continue;
                }


                // 3. Lookup in Custom Dictionary (High Priority)
                if (_customDict.TryGetValue(word, out var customPhones))
                {
                    if (InferenceEngineV1.DebugMode)
                        Console.WriteLine($"[EnglishG2P] Custom dict matched: {word} -> {string.Join(" ", customPhones)}");
                    phones.AddRange(customPhones);
                    word2ph.Add(customPhones.Length);
                    continue;
                }

                // 4. Lookup in Manual Dictionary
                if (_manualDict.TryGetValue(word, out var manualPhones))
                {
                    if (InferenceEngineV1.DebugMode)
                        Console.WriteLine($"[EnglishG2P] Manual dict matched: {word} -> {string.Join(" ", manualPhones)}");
                    phones.AddRange(manualPhones);
                    word2ph.Add(manualPhones.Length);
                    continue;
                }

                // 5. Lookup in CMUDict
                // 常见单字母词白名单 (即使只有 1 个音素也直接使用)
                bool isCommonShortWord = word == "i" || word == "a" || word == "o" ||
                                         word == "an" || word == "am" || word == "is" ||
                                         word == "it" || word == "in" || word == "on" ||
                                         word == "or" || word == "at" || word == "to";

                // 如果 CMUDict 返回的音素太少 (< 2) 且不是常见词，改用字母拼读
                if (_cmuDict.TryGetValue(word, out var phonemes) && (phonemes.Length >= 2 || isCommonShortWord))
                {
                    phones.AddRange(phonemes);
                    word2ph.Add(phonemes.Length);
                }
                else
                {
                    // OOV: Use NeuralG2P if available, otherwise spell out
                    List<string> oovPhones;

                    if (_neuralG2P != null)
                    {
                        oovPhones = _neuralG2P.Predict(word);
                        if (oovPhones.Count > 0 && InferenceEngineV1.DebugMode)
                        {
                            Console.WriteLine($"[EnglishG2P] NeuralG2P: {word} -> {string.Join(" ", oovPhones)}");
                        }
                    }
                    else
                    {
                        oovPhones = new List<string>();
                    }

                    // Fallback to letter spelling if NeuralG2P returns empty
                    if (oovPhones.Count == 0)
                    {
                        if (InferenceEngineV1.DebugMode)
                            Console.WriteLine($"[EnglishG2P] OOV word: {word}, spelling out...");
                        foreach (char c in word.ToLowerInvariant())
                        {
                            // Use _manualDict for letter phonemes (contains a-z with SP separators)
                            if (_manualDict.TryGetValue(c.ToString(), out var letterPhones))
                            {
                                oovPhones.AddRange(letterPhones);
                            }
                        }
                    }

                    if (oovPhones.Count == 0)
                    {
                        phones.Add("SP");
                        word2ph.Add(1);
                    }
                    else
                    {
                        phones.AddRange(oovPhones);
                        word2ph.Add(oovPhones.Count);
                    }
                }
            }

            // 如果整个输出的音素太少 (< 3)，添加 SP padding 避免 T2S 输入过短
            while (phones.Count < 3)
            {
                phones.Add("SP");
                word2ph.Add(1);
            }

            // 注意：不再在此处添加尾部标点
            // 由 MixedLanguageG2P 统一处理最终输出的尾部标点

            return new G2PResult
            {
                NormalizedText = text,
                Phones = phones,
                PhoneIds = Symbols.GetIds(phones),
                Word2Ph = word2ph.ToArray()
            };
        }
    }
}

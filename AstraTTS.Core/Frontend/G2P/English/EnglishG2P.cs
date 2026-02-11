using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AstraTTS.Core.Core;

using AstraTTS.Core.Frontend.G2P.Common;

namespace AstraTTS.Core.Frontend.G2P.English
{
    /// <summary>
    /// 英文 G2P 处理器，使用 CMUDict 将英文单词转换为 ARPAbet 音素。
    /// </summary>
    public class EnglishG2P : IG2P
    {
        // Word -> Phonemes (e.g. "hello" -> ["HH", "AH0", "L", "OW1"])
        private readonly Dictionary<string, string[]> _cmuDict;

        // 外部自定义词典 (用户配置)
        private readonly Dictionary<string, string[]> _customDict;

        // 进阶语言学组件
        private readonly EnglishPosTagger? _posTagger;
        private readonly EnglishWordSegmenter? _wordSegmenter;
        private readonly EnglishNeuralG2P? _neuralG2P;

        public bool HasPosTagger => _posTagger != null;
        public bool HasWordSegmenter => _wordSegmenter != null;
        public bool HasNeuralG2P => _neuralG2P != null;

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

        public EnglishG2P(string cmuDictPath, string? neuralG2PModelPath = null, string? customDictPath = null,
            string? posTaggerDir = null, string? wordSegmentDir = null)
        {
            _cmuDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            _customDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);

            // Initialize POS Tagger
            if (!string.IsNullOrEmpty(posTaggerDir) && Directory.Exists(posTaggerDir))
            {
                _posTagger = new EnglishPosTagger(posTaggerDir);
            }

            // Initialize Word Segmenter
            if (!string.IsNullOrEmpty(wordSegmentDir) && Directory.Exists(wordSegmentDir))
            {
                _wordSegmenter = new EnglishWordSegmenter(wordSegmentDir);
            }

            // Initialize NeuralG2P if model path is provided
            if (!string.IsNullOrEmpty(neuralG2PModelPath) && File.Exists(neuralG2PModelPath))
            {
                _neuralG2P = new EnglishNeuralG2P();
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

                    // Filter problematic words as in Genie-TTS
                    string lowerWord = word.ToLowerInvariant();
                    if (lowerWord == "ae" || lowerWord == "ai" || lowerWord == "ar" ||
                        lowerWord == "ios" || lowerWord == "hud" || lowerWord == "os") continue;

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

            // 1. 分割驼峰式复合词 (OpenAI -> Open AI, GPT4 -> GPT 4)
            // 注意：已经变成小写的词 (如 openai) 不会被此正则匹配
            text = Regex.Replace(text, @"([a-z])([A-Z])", "$1 $2");
            text = Regex.Replace(text, @"([A-Z]+)([A-Z][a-z])", "$1 $2");

            // 2. Split by non-alphabetic but preserve apostrophes for possessives
            var words = Regex.Split(text, @"([^a-zA-Z']+)").Select(w => w.Trim()).Where(w => !string.IsNullOrEmpty(w)).ToArray();

            List<string> phones = new List<string>();
            List<int> word2ph = new List<int>();

            // 2.5 词性标注 (仅当 posTagger 可用且有单词时)
            var alphabeticWords = words.Where(w => Regex.IsMatch(w, @"[a-z]", RegexOptions.IgnoreCase)).ToArray();
            string[] tags = _posTagger != null && alphabeticWords.Length > 0
                ? _posTagger.Tag(alphabeticWords)
                : Array.Empty<string>();

            int alphaIdx = 0;

            for (int i = 0; i < words.Length; i++)
            {
                var word = words[i].ToLowerInvariant();
                bool isAlpha = Regex.IsMatch(word, @"[a-z]");
                string? currentTag = isAlpha && alphaIdx < tags.Length ? tags[alphaIdx++] : null;

                // Punctuation
                if (Symbols.Punctuation.Contains(word))
                {
                    phones.Add(word);
                    word2ph.Add(1);
                    continue;
                }

                // Non-alphabetic characters (spaces, etc.)
                if (!isAlpha)
                {
                    continue;
                }

                // 3. Process the word (possessives, homographs, dict lookup, etc.)
                var wordPhones = QueryWord(word, currentTag);
                if (wordPhones.Count > 0)
                {
                    phones.AddRange(wordPhones);
                    word2ph.Add(wordPhones.Count);
                }
            }

            // 如果整个输出的音素太少 (< 3)，添加 SP padding 避免 T2S 输入过短
            while (phones.Count < 3)
            {
                phones.Add("SP");
                word2ph.Add(1);
            }

            if (InferenceEngineV1.DebugMode)
            {
                Console.WriteLine($"[EnglishG2P] Final phones ({phones.Count}): {string.Join(", ", phones)}");
            }

            return new G2PResult
            {
                NormalizedText = text,
                Phones = phones,
                PhoneIds = Symbols.GetIds(phones),
                Word2Ph = word2ph.ToArray()
            };
        }

        private List<string> QueryWord(string word, string? tag = null)
        {
            string lower = word.ToLowerInvariant();

            // 1. Lookup in Custom Dictionary
            if (_customDict.TryGetValue(lower, out var customPhones)) return customPhones.ToList();

            // 2. Lookup in Manual Dictionary
            if (_manualDict.TryGetValue(lower, out var manualPhones)) return manualPhones.ToList();

            // 3. Advanced Homograph Handling using POS tags
            if (tag != null)
            {
                if (lower == "read")
                {
                    Console.WriteLine($"[EnglishG2P] Disambiguating 'read' with tag: {tag}");
                    // VBD (past) or VBN (past participle) -> /rɛd/
                    if (tag == "VBD" || tag == "VBN")
                        return new List<string> { "R", "EH1", "D" };
                    // Default to /riːd/ for VB, VBP, VBZ, etc.
                    return new List<string> { "R", "IY1", "D" };
                }

                if (lower == "lead")
                {
                    // NN (lead metal) -> /lɛd/
                    if (tag == "NN")
                        return new List<string> { "L", "EH1", "D" };
                    // VB (to lead) -> /liːd/
                    return new List<string> { "L", "IY1", "D" };
                }

                if (lower == "live")
                {
                    // JJ (live show), RB (broadcast live) -> /laɪv/
                    if (tag.StartsWith("JJ") || tag.StartsWith("RB"))
                        return new List<string> { "L", "AY1", "V" };
                    // VB, VBP (I live here) -> /lɪv/
                    return new List<string> { "L", "IH1", "V" };
                }

                if (lower == "record" || lower == "records")
                {
                    bool isPlural = lower.EndsWith("s");
                    // Verb (re-CORD) -> /rɪˈkɔːrd/
                    if (tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "R", "IH0", "K", "AO1", "R", "D" };
                        if (isPlural) p.Add("Z");
                        return p;
                    }
                    // Noun (RE-cord) -> /ˈrɛkərd/
                    var n = new List<string> { "R", "EH1", "K", "ER0", "D" };
                    if (isPlural) n.Add("Z");
                    return n;
                }

                if (lower == "object" || lower == "objects")
                {
                    bool isPlural = lower.EndsWith("s");
                    // Verb (ob-JECT) -> /əbˈdʒɛkt/
                    if (tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "AH0", "B", "JH", "EH1", "K", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    // Noun (OB-ject) -> /ˈɒbdʒɪkt/
                    var n = new List<string> { "AA1", "B", "JH", "EH0", "K", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }

                if (lower == "desert" || lower == "deserts")
                {
                    bool isPlural = lower.EndsWith("s");
                    // Verb (de-SERT) -> /dɪˈzɜːrt/
                    if (tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "D", "IH0", "Z", "ER1", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    // Noun (DES-ert) -> /ˈdɛzərt/
                    var n = new List<string> { "D", "EH1", "Z", "ER0", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }

                if (lower == "content" || lower == "contents")
                {
                    bool isPlural = lower.EndsWith("s");
                    // Adj/Verb (con-TENT) -> /kənˈtɛnt/
                    // e.g. "I am content", "contented"
                    if (tag.StartsWith("JJ") || tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "K", "AH0", "N", "T", "EH1", "N", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    // Noun (CON-tent) -> /ˈkɒntɛnt/
                    var n = new List<string> { "K", "AA1", "N", "T", "EH0", "N", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }

                if (lower == "present" || lower == "presents")
                {
                    // Verb (pre-SENT) vs Noun/Adj (PRE-sent)
                    bool isVerb = tag.StartsWith("VB");
                    bool isPlural = lower.EndsWith("s");

                    if (isVerb) // [P R IY0 Z EH1 N T]
                    {
                        var p = new List<string> { "P", "R", "IY0", "Z", "EH1", "N", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    // Noun/Adj: [P R EH1 Z AH0 N T]
                    var n = new List<string> { "P", "R", "EH1", "Z", "AH0", "N", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }
            }

            // 4. Special Case: 'A' as AH0 (usually when used as a determiner)
            if (lower == "a") return new List<string> { "AH0" };

            // 4. CMUDict Lookup
            if (_cmuDict.TryGetValue(lower, out var phonemes))
            {
                // Common short word check to avoid spell-out logic below
                bool isCommonShortWord = lower == "i" || lower == "a" || lower == "o" ||
                                         lower == "an" || lower == "am" || lower == "is" ||
                                         lower == "it" || lower == "in" || lower == "on" ||
                                         lower == "or" || lower == "at" || lower == "to";

                if (phonemes.Length >= 2 || isCommonShortWord) return phonemes.ToList();
            }

            // 6. Compound Word Segmentation for OOV
            if (_wordSegmenter != null)
            {
                var segments = _wordSegmenter.Segment(lower);
                if (segments.Length > 1)
                {
                    List<string> combinedPhones = new List<string>();
                    foreach (var segment in segments)
                    {
                        // Sub-segments usually don't have tags
                        var segPhones = QueryWord(segment, null);
                        combinedPhones.AddRange(segPhones.Where(p => p != "SP"));
                    }
                    if (combinedPhones.Count > 0) return combinedPhones;
                }
            }

            // 7. Possessives ('s)
            if (lower.EndsWith("'s") && lower.Length > 2)
            {
                var basePron = QueryWord(lower.Substring(0, lower.Length - 2), tag);
                if (basePron.Count > 0)
                {
                    string lastPh = basePron.Last();
                    // Morphophonemic rules for 's
                    // Sibilants: S, Z, SH, ZH, CH, JH -> AH0 Z
                    if (new[] { "S", "Z", "SH", "ZH", "CH", "JH" }.Any(s => lastPh.StartsWith(s)))
                        basePron.AddRange(new[] { "AH0", "Z" });
                    // Voiceless: P, T, K, F, TH -> S
                    else if (new[] { "P", "T", "K", "F", "TH" }.Any(s => lastPh.StartsWith(s)))
                        basePron.Add("S");
                    // Others: -> Z
                    else
                        basePron.Add("Z");

                    return basePron;
                }
            }

            // 6. Neural G2P or Spell-out
            List<string> oovPhones = new List<string>();
            if (_neuralG2P != null)
            {
                oovPhones = _neuralG2P.Predict(lower);
            }

            if (oovPhones.Count == 0)
            {
                foreach (char c in lower)
                {
                    if (_manualDict.TryGetValue(c.ToString(), out var letterPhones))
                        oovPhones.AddRange(letterPhones);
                }
            }

            return oovPhones.Count > 0 ? oovPhones : new List<string> { "SP" };
        }
    }
}

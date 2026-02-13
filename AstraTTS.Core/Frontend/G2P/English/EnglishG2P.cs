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

        // 手动特殊词典 (从 en_special_words.txt 加载)
        private readonly Dictionary<string, string[]> _specialDict;

        // 进阶语言学组件
        private readonly EnglishPosTagger? _posTagger;
        private readonly EnglishWordSegmenter? _wordSegmenter;
        private readonly EnglishNeuralG2P? _neuralG2P;

        // 优先级模式：0-词典优先, 1-仅词典, 2-模型优先
        public int PriorityMode { get; set; } = 0;

        public bool HasPosTagger => _posTagger != null;
        public bool HasWordSegmenter => _wordSegmenter != null;
        public bool HasNeuralG2P => _neuralG2P != null;

        // 单字母发音词典 (用于字母拼读保底)
        private static readonly Dictionary<string, string[]> _manualDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            { "a", new[] { "EY1" } },
            { "b", new[] { "B", "IY1" } },
            { "c", new[] { "S", "IY1" } },
            { "d", new[] { "D", "IY1" } },
            { "e", new[] { "IY1", "IY0", "SP"} },
            { "f", new[] { "EH1", "F" } },
            { "g", new[] { "JH", "IY1" } },
            { "h", new[] { "EY1", "CH" } },
            { "i", new[] { "AY1" } },
            { "j", new[] { "JH", "EY1" } },
            { "k", new[] { "K", "EY1" } },
            { "l", new[] { "EH1", "L" } },
            { "m", new[] { "EH1", "M" } },
            { "n", new[] { "EH1", "N" } },
            { "o", new[] { "OW1", "SP" } },
            { "p", new[] { "P", "IY1" } },
            { "q", new[] { "K", "Y", "Y", "UW1" } },
            { "r", new[] { "AA1", "R" } },
            { "s", new[] { "EH1", "S" } },
            { "t", new[] { "T", "IY1" } },
            { "u", new[] { "Y", "UW1" } },
            { "v", new[] { "V", "IY1" } },
            { "w", new[] { "D", "AH1", "B", "L", "Y", "UW1" } },
            { "x", new[] { "EH1", "K", "S" } },
            { "y", new[] { "W", "AY1" } },
            { "z", new[] { "Z", "IY1", "SP" } }
        };

        public EnglishG2P(string cmuDictPath, string? neuralG2PModelPath = null, string? customDictPath = null,
            string? posTaggerDir = null, string? wordSegmentDir = null, string? specialDictPath = null, int priorityMode = 0)
        {
            _cmuDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            _customDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            _specialDict = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            PriorityMode = priorityMode;

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

            // Load Special Dict
            if (!string.IsNullOrEmpty(specialDictPath) && File.Exists(specialDictPath))
            {
                Console.WriteLine($"[EnglishG2P] Loading special dictionary from {specialDictPath}...");
                foreach (var line in File.ReadAllLines(specialDictPath))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                    var parts = line.Split('\t');
                    if (parts.Length >= 2)
                    {
                        string word = parts[0].Trim();
                        string[] phonemes = parts[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        _specialDict[word] = phonemes;
                    }
                }
                Console.WriteLine($"[EnglishG2P] Loaded {_specialDict.Count} special English words/abbreviations.");
            }
        }

        public bool DebugMode { get; set; } = false;

        /// <summary>
        /// 处理纯英文文本。
        /// </summary>
        public G2PResult Process(string text, int? priorityMode = null)
        {
            // Use override if provided, otherwise use property
            int effectiveMode = priorityMode ?? PriorityMode;
            // 0. 文本正规化 (数字 -> 英文)
            text = TextNorm.EnglishTextNormalizer.Normalize(text);

            // 2. Split by whitespace and punctuation, but KEEPing them as tokens
            var words = Regex.Split(text, @"([\s,\.!?;:\(\)\[\]])").Where(w => !string.IsNullOrEmpty(w)).ToArray();

            List<string> phones = new List<string>();
            List<int> word2ph = new List<int>();

            // 2.5 词性标注 (仅针对字母词)
            var alphabeticWords = words.Where(w => Regex.IsMatch(w, @"[a-zA-Z]", RegexOptions.IgnoreCase)).Select(w => w.Trim()).Where(w => !string.IsNullOrEmpty(w)).ToArray();
            string[] tags = _posTagger != null && alphabeticWords.Length > 0
                ? _posTagger.Tag(alphabeticWords)
                : Array.Empty<string>();

            if (DebugMode)
            {
                Console.WriteLine($"[EnglishG2P] Input Text: \"{text}\"");
                Console.WriteLine($"[EnglishG2P] Split Words: [{string.Join(" | ", words)}]");
                Console.WriteLine($"[EnglishG2P] AlphaWords: [{string.Join(", ", alphabeticWords)}]");
                Console.WriteLine($"[EnglishG2P] Tags: [{string.Join(", ", tags)}]");
            }

            int alphaIdx = 0;

            for (int i = 0; i < words.Length; i++)
            {
                var word = words[i];
                if (string.IsNullOrWhiteSpace(word)) continue;

                bool isAlpha = Regex.IsMatch(word, @"[a-z]", RegexOptions.IgnoreCase);
                string? currentTag = isAlpha && alphaIdx < tags.Length ? tags[alphaIdx++] : null;

                List<string> qRes;
                if (Symbols.Punctuation.Contains(word))
                {
                    qRes = new List<string> { word
    };
                }
                else if (!isAlpha)
                {
                    qRes = new List<string>(); // Non-alphabetic, non-punctuation words are skipped for phonemization
                }
                else
                {
                    qRes = QueryWord(word, currentTag, effectiveMode);
                }

                if (qRes.Count > 0)
                {
                    phones.AddRange(qRes);

                    // 记录当前词对应的音素数量
                    int phCount = qRes.Count;

                    // 在单词之间添加 SP (如果不是最后一个单词且下一个不是标点)
                    if (i < words.Length - 1)
                    {
                        var nextWord = words[i + 1].Trim();
                        // Only add SP if the next "word" is not empty and is not a punctuation mark
                        if (!string.IsNullOrEmpty(nextWord) && !Regex.IsMatch(nextWord, @"^[\.,!?;:\(\)\[\]]$"))
                        {
                            phones.Add("SP");
                            phCount++;
                        }
                    }

                    word2ph.Add(phCount);
                }
            }

            // 补全静音音素 (用于对齐 BERT 等后续模块的最小长度要求)
            while (phones.Count < 3)
            {
                phones.Add("SP");
                word2ph.Add(1);
            }

            if (true)
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

        private List<string> QueryWord(string word, string? tag, int priorityMode)
        {
            string lower = word.ToLowerInvariant();

            // 1. Lookup in Custom Dictionary
            if (_customDict.TryGetValue(lower, out var customPhones)) return customPhones.ToList();

            // 2. Lookup in Special Dictionary (Technical terms, abbreviations)
            if (_specialDict.TryGetValue(lower, out var specialPhones)) return specialPhones.ToList();

            // 3. Model-First Mode optimization
            if (priorityMode == 2 && _neuralG2P != null)
            {
                var modelPhones = _neuralG2P.Predict(lower);
                if (modelPhones.Count > 0) return modelPhones;
            }



            // 4. Advanced Homograph Handling
            if (PriorityMode != 2 && tag != null)
            {
                if (lower == "read")
                {
                    if (tag == "VBD" || tag == "VBN")
                        return new List<string> { "R", "EH1", "D" };
                    return new List<string> { "R", "IY1", "D" };
                }
                if (lower == "lead")
                {
                    if (tag == "NN") return new List<string> { "L", "EH1", "D" };
                    return new List<string> { "L", "IY1", "D" };
                }
                if (lower == "live")
                {
                    if (tag.StartsWith("JJ") || tag.StartsWith("RB"))
                        return new List<string> { "L", "AY1", "V" };
                    return new List<string> { "L", "IH1", "V" };
                }
                if (lower == "record" || lower == "records")
                {
                    bool isPlural = lower.EndsWith("s");
                    if (tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "R", "IH0", "K", "AO1", "R", "D" };
                        if (isPlural) p.Add("Z");
                        return p;
                    }
                    var n = new List<string> { "R", "EH1", "K", "ER0", "D" };
                    if (isPlural) n.Add("Z");
                    return n;
                }
                if (lower == "object" || lower == "objects")
                {
                    bool isPlural = lower.EndsWith("s");
                    if (tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "AH0", "B", "JH", "EH1", "K", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    var n = new List<string> { "AA1", "B", "JH", "EH0", "K", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }
                if (lower == "desert" || lower == "deserts")
                {
                    bool isPlural = lower.EndsWith("s");
                    if (tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "D", "IH0", "Z", "ER1", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    var n = new List<string> { "D", "EH1", "Z", "ER0", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }
                if (lower == "content" || lower == "contents")
                {
                    bool isPlural = lower.EndsWith("s");
                    if (tag.StartsWith("JJ") || tag.StartsWith("VB"))
                    {
                        var p = new List<string> { "K", "AH0", "N", "T", "EH1", "N", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    var n = new List<string> { "K", "AA1", "N", "T", "EH0", "N", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }
                if (lower == "present" || lower == "presents")
                {
                    bool isVerb = tag.StartsWith("VB");
                    bool isPlural = lower.EndsWith("s");
                    if (isVerb)
                    {
                        var p = new List<string> { "P", "R", "IY0", "Z", "EH1", "N", "T" };
                        if (isPlural) p.Add("S");
                        return p;
                    }
                    var n = new List<string> { "P", "R", "EH1", "Z", "AH0", "N", "T" };
                    if (isPlural) n.Add("S");
                    return n;
                }
            }

            if (lower == "a") return new List<string> { "AH0" };

            // 5. CMUDict Lookup
            if (_cmuDict.TryGetValue(lower, out var phonemes))
            {
                bool isCommonShortWord = lower == "i" || lower == "a" || lower == "o" ||
                                         lower == "an" || lower == "am" || lower == "is" ||
                                         lower == "it" || lower == "in" || lower == "on" ||
                                         lower == "or" || lower == "at" || lower == "to";
                if (phonemes.Length >= 2 || isCommonShortWord) return phonemes.ToList();
            }

            // 6. Manual Dict (Single Letters)
            if (_manualDict.TryGetValue(lower, out var manualPhones)) return manualPhones.ToList();

            // 6.5 CamelCase Fallback (e.g. "OpenAI" -> "Open", "AI")
            // Try only if the word actually contains camelCase patterns
            string split = Regex.Replace(word, @"([a-z])([A-Z])", "$1 $2");
            split = Regex.Replace(split, @"([A-Z]+)([A-Z][a-z])", "$1 $2");
            if (split != word)
            {
                var parts = split.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length > 1)
                {
                    List<string> combined = new List<string>();
                    for (int j = 0; j < parts.Length; j++)
                    {
                        var partPhones = QueryWord(parts[j], null, priorityMode);
                        combined.AddRange(partPhones);
                        // Add SP between parts if not already there and if not at the end
                        if (j < parts.Length - 1 && partPhones.Count > 0 && partPhones.Last() != "SP")
                        {
                            combined.Add("SP");
                        }
                    }
                    return combined;
                }
            }

            // 7. Compound Word Segmentation
            if (_wordSegmenter != null)
            {
                var segments = _wordSegmenter.Segment(lower);
                if (segments.Length > 1)
                {
                    List<string> combinedPhones = new List<string>();
                    foreach (var segment in segments)
                    {
                        var segPhones = QueryWord(segment, null, priorityMode);
                        combinedPhones.AddRange(segPhones.Where(p => p != "SP"));
                    }
                    if (combinedPhones.Count > 0) return combinedPhones;
                }
            }

            // 8. Possessives
            if (lower.EndsWith("'s") && lower.Length > 2)
            {
                var basePron = QueryWord(lower.Substring(0, lower.Length - 2), tag, priorityMode);
                if (basePron.Count > 0)
                {
                    string lastPh = basePron.Last();
                    if (new[] { "S", "Z", "SH", "ZH", "CH", "JH" }.Any(s => lastPh.StartsWith(s)))
                        basePron.AddRange(new[] { "AH0", "Z" });
                    else if (new[] { "P", "T", "K", "F", "TH" }.Any(s => lastPh.StartsWith(s)))
                        basePron.Add("S");
                    else basePron.Add("Z");
                    return basePron;
                }
            }

            // 9. Neural G2P or Spell-out
            List<string> oovPhones = new List<string>();
            if (priorityMode != 1 && priorityMode != 2 && _neuralG2P != null)
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

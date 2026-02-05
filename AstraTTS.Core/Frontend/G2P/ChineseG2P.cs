using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AstraTTS.Core.Utils;

namespace AstraTTS.Core.Frontend.G2P
{
    public class ChineseG2P : IG2P
    {
        private readonly SimpleMaxMatchSegmenter _segmenter;
        private readonly ToneSandhi _toneModifier;

        // Pinyin -> Phonemes (e.g. "ni" -> ["n", "i"])
        // Loaded from opencpop-strict.txt
        private readonly Dictionary<string, string[]> _pinyinToPhonemes;

        // Word -> Pinyin[] (e.g. "你好" -> ["ni3", "hao3"])
        // Loaded from mandarin_pinyin.dict
        private readonly Dictionary<string, string[]> _wordToPinyin;

        // User Custom Dictionary
        private readonly Dictionary<string, string[]> _customWordToPinyin;

        public ChineseG2P(string vocabPath, string? pinyinDictPath = null, string? customDictPath = null)
        {
            _segmenter = new SimpleMaxMatchSegmenter();
            _toneModifier = new ToneSandhi();
            _pinyinToPhonemes = new Dictionary<string, string[]>();
            _wordToPinyin = new Dictionary<string, string[]>();
            _customWordToPinyin = new Dictionary<string, string[]>();

            // 1. Load Pinyin -> Phoneme mapping (opencpop-strict.txt)
            if (File.Exists(vocabPath))
            {
                foreach (var line in File.ReadAllLines(vocabPath))
                {
                    var parts = line.Split(new[] { '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                    {
                        // parts[0] = pinyin (e.g., "a", "ba")
                        // parts[1] = phonemes (e.g., "AA a", "b a")
                        _pinyinToPhonemes[parts[0]] = parts[1].Split(' ');
                    }
                }
            }
            else
            {
                Console.WriteLine($"[ChineseG2P Warning] Vocab file not found: {vocabPath}");
            }

            // 2. Load Word -> Pinyin mapping (mandarin_pinyin.dict)
            if (!string.IsNullOrEmpty(pinyinDictPath) && File.Exists(pinyinDictPath))
            {
                Console.WriteLine($"[ChineseG2P] Loading dictionary from {pinyinDictPath}...");
                var dictLines = File.ReadAllLines(pinyinDictPath);
                foreach (var line in dictLines)
                {
                    // Format: Word\tPinyin1 Pinyin2...
                    var parts = line.Split('\t');
                    if (parts.Length >= 2)
                    {
                        string word = parts[0];
                        string[] pinyins = parts[1].Split(' ');
                        _wordToPinyin[word] = pinyins;
                    }
                }
                // Load vocab into segmenter
                _segmenter.LoadVocabulary(_wordToPinyin.Keys);
                Console.WriteLine($"[ChineseG2P] Loaded {_wordToPinyin.Count} words into dictionary.");
            }
            else
            {
                Console.WriteLine("[ChineseG2P Warning] Pinyin dictionary not found. Fallback to basic mode.");
            }

            // 3. Load User Custom Dictionary
            if (!string.IsNullOrEmpty(customDictPath) && File.Exists(customDictPath))
            {
                Console.WriteLine($"[ChineseG2P] Loading custom dictionary from {customDictPath}...");
                foreach (var line in File.ReadAllLines(customDictPath))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                    var parts = line.Split('\t');
                    if (parts.Length >= 2)
                    {
                        string word = parts[0];
                        string[] pinyins = parts[1].Split(' ');
                        _customWordToPinyin[word] = pinyins;
                        _segmenter.AddWord(word); // Ensure custom words are segmented correctly
                    }
                }
                Console.WriteLine($"[ChineseG2P] Loaded {_customWordToPinyin.Count} custom words.");
            }
        }

        public G2PResult Process(string text)
        {
            // 0. 文本正规化 (数字 -> 中文)
            text = TextNorm.ChineseTextNormalizer.Normalize(text);

            // 1. 基础归一化
            string normalized = text.Replace("，", ",").Replace("。", ".").Replace("！", "!").Replace("？", "?");

            // 2. 分词
            var words = _segmenter.Cut(normalized);

            List<string> finalPhones = new List<string>();
            List<int> word2ph = new List<int>();

            foreach (var word in words)
            {
                // 标点处理
                if (Symbols.Punctuation.Contains(word))
                {
                    finalPhones.Add(word);
                    word2ph.Add(1);
                    continue;
                }

                // 获取拼音 (优先级：自定义词典 > 系统词典)
                List<string> pinyins = new List<string>();
                if (_customWordToPinyin.TryGetValue(word, out var customPinyins))
                {
                    pinyins.AddRange(customPinyins);
                }
                else if (_wordToPinyin.TryGetValue(word, out var dictPinyins))
                {
                    pinyins.AddRange(dictPinyins);
                }
                else
                {
                    // 未登录词：逐字尝试
                    foreach (char c in word)
                    {
                        string s = c.ToString();
                        if (_customWordToPinyin.TryGetValue(s, out var charCustomPinyins))
                            pinyins.Add(charCustomPinyins[0]);
                        else if (_wordToPinyin.TryGetValue(s, out var charPinyins))
                            pinyins.Add(charPinyins[0]); // 取第一个读音
                        else
                            pinyins.Add("SP"); // 未知字符
                    }
                }

                if (pinyins.Count > 0)
                {
                    // 变调处理 (针对 '不', '一' 等)
                    // 注意：jieba 字典中的拼音已经是带声调的 (ni3)，ToneSandhi 需要处理这种格式
                    pinyins = _toneModifier.ModifyTones(word, "n", pinyins);

                    foreach (var py in pinyins)
                    {
                        if (py == "SP")
                        {
                            finalPhones.Add("SP");
                            word2ph.Add(1);
                            continue;
                        }

                        // 提取声调
                        // py 格式: "ni3", "hao3", "a5" (轻声)
                        string tone = "";
                        string purePy = py;
                        if (char.IsDigit(py.Last()))
                        {
                            tone = py.Last().ToString();
                            purePy = py.Substring(0, py.Length - 1);
                        }

                        if (_pinyinToPhonemes.TryGetValue(purePy, out var phones))
                        {
                            // 将声调附加到韵母 (第二个音素) 上
                            // Opencpop 映射通常是 [Initial, Final]
                            for (int i = 0; i < phones.Length; i++)
                            {
                                if (i == phones.Length - 1 && !string.IsNullOrEmpty(tone))
                                    finalPhones.Add(phones[i] + tone);
                                else
                                    finalPhones.Add(phones[i]);
                            }
                            word2ph.Add(phones.Length);
                        }
                        else
                        {
                            // 拼音未找到 (可能是错误或特殊发音)
                            Console.WriteLine($"[ChineseG2P Warning] Unknown pinyin: {purePy} (Raw: {py})");
                            finalPhones.Add("SP");
                            word2ph.Add(1);
                        }
                    }
                }
            }

            return new G2PResult
            {
                NormalizedText = normalized,
                Phones = finalPhones,
                PhoneIds = Symbols.GetIds(finalPhones),
                Word2Ph = word2ph.ToArray()
            };
        }
    }
}

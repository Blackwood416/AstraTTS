using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AstraTTS.Core.Utils;

using AstraTTS.Core.Frontend.G2P.Common;

namespace AstraTTS.Core.Frontend.G2P.Chinese
{
    /// <summary>
    /// 增强版本的中文 G2P 处理器。
    /// 集成了 Jieba 分词、变调处理、多音字修正、儿化处理以及 Native Mode。
    /// </summary>
    public class ChineseG2P : IG2P
    {
        private readonly JiebaSegmenter _segmenter;
        private readonly ToneSandhi _toneSandhi;
        private readonly CorrectPronunciation _corrector;
        private readonly ErhuaProcessor _erhuaProcessor;

        // Pinyin -> Phonemes (e.g. "ni" -> ["n", "i"])
        private readonly Dictionary<string, string[]> _pinyinToPhonemes;

        // Word -> Pinyin[] (Mandarin basic dict)
        private readonly Dictionary<string, string[]> _wordToPinyin;

        // User Custom Dictionary
        private readonly Dictionary<string, string[]> _customWordToPinyin;

        public ChineseG2P(string vocabPath, string pinyinDictPath, string customDictPath, string polyphonicJsonPath, string jiebaDictDir)
        {
            _segmenter = new JiebaSegmenter(jiebaDictDir);
            _toneSandhi = new ToneSandhi();
            _corrector = new CorrectPronunciation(polyphonicJsonPath);
            _erhuaProcessor = new ErhuaProcessor();

            _pinyinToPhonemes = new Dictionary<string, string[]>();
            _wordToPinyin = new Dictionary<string, string[]>();
            _customWordToPinyin = new Dictionary<string, string[]>();

            LoadVocab(vocabPath);
            LoadDictionary(pinyinDictPath);
            LoadCustomDictionary(customDictPath);
        }

        private void LoadVocab(string vocabPath)
        {
            if (!File.Exists(vocabPath)) return;
            foreach (var line in File.ReadAllLines(vocabPath))
            {
                var parts = line.Split('\t');
                if (parts.Length >= 2)
                    _pinyinToPhonemes[parts[0]] = parts[1].Split(' ');
            }
        }

        private void LoadDictionary(string pinyinDictPath)
        {
            if (!File.Exists(pinyinDictPath)) return;
            foreach (var line in File.ReadAllLines(pinyinDictPath))
            {
                var parts = line.Split('\t');
                if (parts.Length >= 2)
                    _wordToPinyin[parts[0]] = parts[1].Split(' ');
            }
        }

        private void LoadCustomDictionary(string customDictPath)
        {
            if (!File.Exists(customDictPath)) return;
            foreach (var line in File.ReadAllLines(customDictPath))
            {
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                var parts = line.Split('\t');
                if (parts.Length >= 2)
                    _customWordToPinyin[parts[0]] = parts[1].Split(' ');
            }
        }

        public G2PResult Process(string text)
        {
            // --- 1. Native Mode Check ---
            // If text is wrapped in { }, treat as direct phonemes
            if (text.StartsWith("{") && text.EndsWith("}"))
            {
                return ProcessNativeMode(text);
            }

            // --- 2. Normalization ---
            string normalized = TextNorm.ChineseTextNormalizer.Normalize(text);
            // Replace full-width punctuation for inner processing
            string processed = normalized.Replace("，", ",").Replace("。", ".").Replace("！", "!").Replace("？", "?");

            // --- 3. Segmentation with POS ---
            var rawSegments = _segmenter.Cut(processed);

            // --- 4. Pre-Merge (Sentence level) ---
            var segments = _toneSandhi.PreMergeForModify(rawSegments);

            List<string> finalPhones = new List<string>();
            List<int> word2phList = new List<int>();

            foreach (var (word, pos) in segments)
            {
                // Punctuation handling
                if (Symbols.Punctuation.Contains(word) || word.All(c => char.IsPunctuation(c) || char.IsSeparator(c)))
                {
                    finalPhones.Add(word == " " ? "SP" : word);
                    word2phList.Add(1);
                    continue;
                }

                // Get initial pinyin
                List<string> pinyins = GetPinyin(word);
                if (pinyins.Count == 0 || pinyins[0] == "SP")
                {
                    foreach (var c in word)
                    {
                        finalPhones.Add("SP");
                        word2phList.Add(1);
                    }
                    continue;
                }

                // --- 5. Correct Pronunciation (Polyphonic) ---
                pinyins = _corrector.Correct(word, pinyins);

                // --- 6. Tone Sandhi ---
                pinyins = _toneSandhi.ModifyTones(word, pos, pinyins);

                // --- 7. Erhua Processing (R-coloring) ---
                // Need to split into initials/finals for Erhua
                var (initials, finals) = SplitPinyin(pinyins);
                (initials, finals) = _erhuaProcessor.MergeErhua(initials, finals, word, pos);

                // --- 8. Final Phoneme Mapping ---
                for (int i = 0; i < finals.Count; i++)
                {
                    string ini = initials[i];
                    string fin = finals[i];

                    int phoneCount = 0;
                    if (!string.IsNullOrEmpty(ini))
                    {
                        finalPhones.Add(ini);
                        phoneCount++;
                    }
                    if (!string.IsNullOrEmpty(fin))
                    {
                        finalPhones.Add(fin);
                        phoneCount++;
                    }
                    word2phList.Add(phoneCount);
                }
            }

            return new G2PResult
            {
                NormalizedText = normalized,
                Phones = finalPhones,
                PhoneIds = Symbols.GetIds(finalPhones),
                Word2Ph = word2phList.ToArray()
            };
        }

        private G2PResult ProcessNativeMode(string text)
        {
            // Input: {ni3}{hao3} or {n i3 h ao3}
            // Logic: Strip { } and split by } { or spaces.
            string content = text.Trim('{', '}');
            var units = content.Split(new[] { "}{", " " }, StringSplitOptions.RemoveEmptyEntries);

            List<string> phones = new List<string>();
            List<int> w2p = new List<int>();

            foreach (var unit in units)
            {
                // If it's a pinyin like "ni3"
                if (char.IsDigit(unit[^1]) || _pinyinToPhonemes.ContainsKey(unit))
                {
                    string py = unit;
                    char tone = char.IsDigit(py[^1]) ? py[^1] : '5';
                    string pure = char.IsDigit(py[^1]) ? py[..^1] : py;

                    if (_pinyinToPhonemes.TryGetValue(pure, out var ps))
                    {
                        phones.Add(ps[0]); // initial
                        phones.Add(ps[1] + tone); // final+tone
                        w2p.Add(2);
                    }
                    else
                    {
                        phones.Add(unit);
                        w2p.Add(1);
                    }
                }
                else
                {
                    phones.Add(unit);
                    w2p.Add(1);
                }
            }

            return new G2PResult
            {
                NormalizedText = text,
                Phones = phones,
                PhoneIds = Symbols.GetIds(phones),
                Word2Ph = w2p.ToArray()
            };
        }

        private List<string> GetPinyin(string word)
        {
            if (_customWordToPinyin.TryGetValue(word, out var cp)) return new List<string>(cp);
            if (_wordToPinyin.TryGetValue(word, out var dp)) return new List<string>(dp);

            // Fallback: character by character
            var result = new List<string>();
            foreach (var c in word)
            {
                string s = c.ToString();
                if (_customWordToPinyin.TryGetValue(s, out var ccp)) result.Add(ccp[0]);
                else if (_wordToPinyin.TryGetValue(s, out var cdp)) result.Add(cdp[0]);
                else result.Add("SP");
            }
            return result;
        }

        private (List<string> initials, List<string> finals) SplitPinyin(List<string> pinyins)
        {
            var initials = new List<string>();
            var finals = new List<string>();

            foreach (var py in pinyins)
            {
                if (py == "SP")
                {
                    initials.Add("");
                    finals.Add("SP");
                    continue;
                }

                char tone = char.IsDigit(py[^1]) ? py[^1] : '5';
                string pure = char.IsDigit(py[^1]) ? py[..^1] : py;

                if (_pinyinToPhonemes.TryGetValue(pure, out var ps))
                {
                    // Opencpop: [Initial, Final]
                    initials.Add(ps.Length > 1 ? ps[0] : "");
                    finals.Add((ps.Length > 1 ? ps[1] : ps[0]) + tone);
                }
                else
                {
                    initials.Add("");
                    finals.Add(py);
                }
            }
            return (initials, finals);
        }
    }
}

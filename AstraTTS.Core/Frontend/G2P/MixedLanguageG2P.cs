using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AstraTTS.Core.Core;
using AstraTTS.Core.Frontend.G2P.Common;
using AstraTTS.Core.Frontend.G2P.Chinese;
using AstraTTS.Core.Frontend.G2P.English;
using AstraTTS.Core.Frontend.G2P.Japanese;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 混合语言 G2P 处理器，自动检测中英日文片段并路由到对应的 G2P。
    /// </summary>
    public class MixedLanguageG2P : IG2P
    {
        private readonly ChineseG2P _chineseG2P;
        private readonly EnglishG2P _englishG2P;
        private readonly JapaneseG2P? _japaneseG2P;

        public MixedLanguageG2P(ChineseG2P chineseG2P, EnglishG2P englishG2P, JapaneseG2P? japaneseG2P = null)
        {
            _chineseG2P = chineseG2P;
            _englishG2P = englishG2P;
            _japaneseG2P = japaneseG2P;
        }

        /// <summary>
        /// 处理混合语言文本（自动检测语言）。
        /// </summary>
        public G2PResult Process(string text) => Process(text, null);

        /// <summary>
        /// 处理文本，可选择显式指定语言。
        /// </summary>
        /// <param name="text">要处理的文本</param>
        /// <param name="explicitLanguage">显式语言代码 (zh, en, jp/ja)，null 表示自动检测</param>
        public G2PResult Process(string text, string? explicitLanguage)
        {
            // 1. 使用 TaggedTextParser 解析文本标签 ([lang] 和 {ph})
            var textSegments = TaggedTextParser.Parse(text);

            // 如果指定了全局日语标记，或原始文本包含假名，则后续汉字检测可能偏向日语
            bool forceJapanese = IsJapaneseLanguageCode(explicitLanguage)
                                 || (explicitLanguage == null && ContainsJapaneseKana(text));

            List<string> allPhones = new List<string>();
            List<int> allWord2Ph = new List<int>();
            List<PhoneLanguage> languageTags = new List<PhoneLanguage>();
            List<LanguageSegment> segmentList = new List<LanguageSegment>();
            StringBuilder normalizedBuilder = new StringBuilder();

            Language? prevLang = null;

            foreach (var textSeg in textSegments)
            {
                if (string.IsNullOrEmpty(textSeg.Text)) continue;

                // 2. 根据片段属性决定如何处理
                var subSegments = new List<(string segment, Language lang)>();

                if (textSeg.Language != null)
                {
                    // 强制指定了语言
                    Language l = Language.Chinese;
                    if (IsJapaneseLanguageCode(textSeg.Language)) l = Language.Japanese;
                    else if (textSeg.Language.ToLowerInvariant() == "en") l = Language.English;

                    subSegments.Add((textSeg.Text, l));
                }
                else if (textSeg.Type == TextSegmentType.Native)
                {
                    // 原生音素模式：自动识别目标 G2P (简单启发式：包含数字通常是拼音/汉音)
                    Language l = Language.Chinese;
                    if (Regex.IsMatch(textSeg.Text, @"[0-9]")) l = Language.Chinese;
                    else if (prevLang.HasValue) l = prevLang.Value; // 继承语境

                    subSegments.Add((textSeg.Text, l));
                }
                else
                {
                    // 普通文本：进行正常的语种分割
                    string normalizedSeg = TextNorm.EnglishTextNormalizer.NormalizeSymbols(textSeg.Text);
                    subSegments.AddRange(SplitByLanguage(normalizedSeg, forceJapanese));
                }

                // 3. 路由到具体 G2P
                foreach (var (segment, lang) in subSegments)
                {
                    if (string.IsNullOrWhiteSpace(segment)) continue;

                    if (InferenceEngineV1.DebugMode)
                    {
                        Console.WriteLine($"[MixedG2P] Segment: '{segment}', Detected Language: {lang}");
                    }

                    G2PResult result;
                    PhoneLanguage phoneLang;

                    if (lang == Language.Chinese)
                    {
                        result = _chineseG2P.Process(segment);
                        phoneLang = PhoneLanguage.Chinese;
                        prevLang = Language.Chinese;
                    }
                    else if (lang == Language.English)
                    {
                        if (prevLang == Language.Chinese || prevLang == Language.Japanese)
                        {
                            allPhones.Add("SP");
                            allWord2Ph.Add(1);
                            languageTags.Add(PhoneLanguage.Other);
                        }
                        result = _englishG2P.Process(segment);
                        phoneLang = PhoneLanguage.English;
                        prevLang = Language.English;
                    }
                    else if (lang == Language.Japanese)
                    {
                        if (_japaneseG2P == null)
                        {
                            Console.WriteLine($"[MixedG2P] Warning: Japanese processor missing. Falling back to Chinese for: '{segment}'");
                            result = _chineseG2P.Process(segment);
                            phoneLang = PhoneLanguage.Chinese;
                        }
                        else
                        {
                            if (prevLang.HasValue && prevLang != Language.Japanese)
                            {
                                allPhones.Add("SP");
                                allWord2Ph.Add(1);
                                languageTags.Add(PhoneLanguage.Other);
                            }
                            if (InferenceEngineV1.DebugMode)
                                Console.WriteLine($"[MixedG2P] Using Japanese processor for: '{segment}'");
                            result = _japaneseG2P.Process(segment);
                            phoneLang = PhoneLanguage.Japanese;
                        }
                        prevLang = Language.Japanese;
                    }
                    else
                    {
                        if (Symbols.Punctuation.Contains(segment))
                        {
                            allPhones.Add(segment);
                            allWord2Ph.Add(1);
                            languageTags.Add(PhoneLanguage.Other);
                        }
                        normalizedBuilder.Append(segment);
                        continue;
                    }

                    int startIdx = allPhones.Count;
                    allPhones.AddRange(result.Phones);
                    allWord2Ph.AddRange(result.Word2Ph);
                    for (int i = 0; i < result.Phones.Count; i++)
                    {
                        languageTags.Add(phoneLang);
                    }
                    normalizedBuilder.Append(result.NormalizedText);

                    segmentList.Add(new LanguageSegment
                    {
                        Text = result.NormalizedText,
                        Language = phoneLang,
                        StartPhoneIndex = startIdx,
                        PhoneCount = result.Phones.Count,
                        Word2Ph = result.Word2Ph
                    });
                }
            }

            // 添加尾部标点，防止模型截断最后一个音素
            if (allPhones.Count > 0 && !Symbols.Punctuation.Contains(allPhones.Last()))
            {
                allPhones.Add(".");
                allWord2Ph.Add(1);
                languageTags.Add(PhoneLanguage.Other);
                normalizedBuilder.Append("."); // 同步更新文本，确保与 Word2Ph 长度一致
            }

            // 检查音素数量是否足够 (太短的输入可能导致模型输出异常)
            const int MIN_PHONES = 6;
            int paddingCount = 0;
            if (allPhones.Count < MIN_PHONES)
            {
                if (InferenceEngineV1.DebugMode)
                    Console.WriteLine($"[MixedG2P] Warning: Input too short ({allPhones.Count} phonemes). Padding to {MIN_PHONES}.");
                while (allPhones.Count < MIN_PHONES)
                {
                    // 在句首添加 SP 作为 padding
                    allPhones.Insert(0, "SP");
                    allWord2Ph.Insert(0, 1);
                    languageTags.Insert(0, PhoneLanguage.Other);
                    normalizedBuilder.Insert(0, " "); // 同步更新文本，插入空格占据一个 Token 位置
                    paddingCount++;
                }

                // 更新所有片段的 startPhoneIndex（因为在开头插入了 padding）
                for (int i = 0; i < segmentList.Count; i++)
                {
                    var seg = segmentList[i];
                    seg.StartPhoneIndex += paddingCount;
                    segmentList[i] = seg;
                }
            }

            return new G2PResult
            {
                NormalizedText = normalizedBuilder.ToString(),
                Phones = allPhones,
                PhoneIds = Symbols.GetIds(allPhones),
                Word2Ph = allWord2Ph.ToArray(),
                LanguageTags = languageTags.ToArray(),
                Segments = segmentList
            };
        }

        private enum Language { Chinese, English, Japanese, Numeric, Other }

        private static bool IsJapaneseLanguageCode(string? langCode)
        {
            if (string.IsNullOrEmpty(langCode)) return false;
            var code = langCode.ToLowerInvariant();
            return code == "jp" || code == "ja" || code == "jpn" || code == "japanese";
        }

        private bool ContainsJapaneseKana(string text)
        {
            foreach (char c in text)
            {
                if (c >= 0x3040 && c <= 0x309F) return true;
                if (c >= 0x30A0 && c <= 0x30FF) return true;
                if (c >= 0x31F0 && c <= 0x31FF) return true;
            }
            return false;
        }

        private List<(string segment, Language lang)> SplitByLanguage(string text, bool forceJapanese = false)
        {
            var result = new List<(string, Language)>();
            var currentSegment = new StringBuilder();
            Language? currentLang = null;

            bool hasCJK = text.Any(c => (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF));
            Language lastDefinitiveLang = forceJapanese ? Language.Japanese : (hasCJK ? Language.Chinese : Language.English);

            foreach (char c in text)
            {
                Language charLang = DetectCharLanguage(c, forceJapanese);

                if (charLang == Language.Numeric)
                {
                    if (currentLang.HasValue && (currentLang == Language.Chinese || currentLang == Language.English || currentLang == Language.Japanese))
                    {
                        charLang = currentLang.Value;
                    }
                    else
                    {
                        charLang = lastDefinitiveLang;
                    }
                }

                if (charLang == Language.Chinese || charLang == Language.English || charLang == Language.Japanese)
                {
                    lastDefinitiveLang = charLang;
                }

                if (currentLang == null)
                {
                    currentLang = charLang;
                    currentSegment.Append(c);
                }
                else if (charLang == currentLang || charLang == Language.Other)
                {
                    currentSegment.Append(c);
                }
                else
                {
                    if (currentSegment.Length > 0)
                    {
                        result.Add((currentSegment.ToString(), currentLang.Value == Language.Other ? lastDefinitiveLang : currentLang.Value));
                        currentSegment.Clear();
                    }
                    currentLang = charLang;
                    currentSegment.Append(c);
                }
            }

            if (currentSegment.Length > 0 && currentLang.HasValue)
            {
                result.Add((currentSegment.ToString(), currentLang.Value == Language.Other ? lastDefinitiveLang : currentLang.Value));
            }

            return result;
        }

        private Language DetectCharLanguage(char c, bool forceJapanese = false)
        {
            if (c >= 0x3040 && c <= 0x309F) return Language.Japanese;
            if (c >= 0x30A0 && c <= 0x30FF) return Language.Japanese;
            if (c >= 0x31F0 && c <= 0x31FF) return Language.Japanese;
            if (c == 0x30FC) return Language.Japanese;

            if (c >= 0x4E00 && c <= 0x9FFF) return forceJapanese ? Language.Japanese : Language.Chinese;
            if (c >= 0x3400 && c <= 0x4DBF) return forceJapanese ? Language.Japanese : Language.Chinese;

            if ((c >= '0' && c <= '9') || c == '.') return Language.Numeric;

            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return Language.English;

            if (c == '$' || c == '£' || c == '€') return Language.English;

            return Language.Other;
        }
    }
}

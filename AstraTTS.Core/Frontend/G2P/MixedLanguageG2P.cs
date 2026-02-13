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
        private readonly HashSet<Language> _allowedLanguages;
        public bool DebugMode { get; set; } = false;

        public MixedLanguageG2P(ChineseG2P chineseG2P, EnglishG2P englishG2P, JapaneseG2P? japaneseG2P = null, IEnumerable<string>? allowedLanguages = null)
        {
            _chineseG2P = chineseG2P;
            _englishG2P = englishG2P;
            _japaneseG2P = japaneseG2P;
            _allowedLanguages = ParseLanguages(allowedLanguages);
        }

        private HashSet<Language> ParseLanguages(IEnumerable<string>? langs)
        {
            var result = new HashSet<Language>();
            if (langs != null)
            {
                foreach (var l in langs)
                {
                    var lower = l.ToLowerInvariant();
                    if (lower == "zh" || lower == "chs") result.Add(Language.Chinese);
                    else if (lower == "en") result.Add(Language.English);
                    else if (lower == "ja" || lower == "jp") result.Add(Language.Japanese);
                }
            }

            // 默认兜底 [zh, en]
            if (result.Count == 0)
            {
                result.Add(Language.Chinese);
                result.Add(Language.English);
            }

            return result;
        }

        /// <summary>
        /// 处理混合语言文本（自动检测语言）。
        /// </summary>
        public G2PResult Process(string text, int? priorityMode = null) => Process(text, null, priorityMode);

        /// <summary>
        /// 处理文本，可选择显式指定语言。
        /// </summary>
        /// <param name="text">要处理的文本</param>
        /// <param name="explicitLanguage">显式语言代码 (zh, en, jp/ja)，null 表示自动检测</param>
        /// <remarks>支持文本中的强制标签，例如 {zh 文本} 或 {en text}</remarks>
        /// <param name="priorityMode">G2P 优先级模式</param>
        public G2PResult Process(string text, string? explicitLanguage, int? priorityMode = null)
        {
            // 1. 使用 TaggedTextParser 解析文本标签 ([lang] 和 {ph})
            var textSegments = TaggedTextParser.Parse(text);

            // 显式指定语言代码
            bool isExplicitJapanese = IsJapaneseLanguageCode(explicitLanguage);
            bool isExplicitChinese = explicitLanguage != null && (explicitLanguage.ToLowerInvariant() == "zh" || explicitLanguage.ToLowerInvariant() == "chs");
            bool isExplicitEnglish = explicitLanguage != null && explicitLanguage.ToLowerInvariant() == "en";

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
                    Language l = Language.English; // 默认英文
                    string langCode = textSeg.Language.ToLowerInvariant();
                    if (IsJapaneseLanguageCode(langCode)) l = Language.Japanese;
                    else if (langCode == "zh" || langCode == "chs") l = Language.Chinese;
                    else if (langCode == "en") l = Language.English;

                    // 既然是强制指定的，直接添加，不再进行二次分割
                    subSegments.Add((textSeg.Text, l));
                }
                else if (textSeg.Type == TextSegmentType.Native)
                {
                    // 原生音素模式
                    Language l = Language.Chinese;
                    if (Regex.IsMatch(textSeg.Text, @"[0-9]")) l = Language.Chinese;
                    else if (prevLang.HasValue) l = prevLang.Value;

                    subSegments.Add((textSeg.Text, l));
                }
                else
                {
                    // 普通文本：进行正常的语种分割
                    string normalizedSeg = TextNorm.EnglishTextNormalizer.NormalizeSymbols(textSeg.Text);

                    // 确定该段落的语言倾向
                    Language? preference = null;
                    if (isExplicitJapanese) preference = Language.Japanese;
                    else if (isExplicitChinese) preference = Language.Chinese;
                    else if (isExplicitEnglish) preference = Language.English;

                    subSegments.AddRange(SplitByLanguage(normalizedSeg, preference));
                }

                // 3. 路由到具体 G2P
                foreach (var (segment, lang) in subSegments)
                {
                    if (string.IsNullOrWhiteSpace(segment)) continue;



                    G2PResult result;
                    PhoneLanguage phoneLang;

                    if (lang == Language.Chinese)
                    {
                        result = _chineseG2P.Process(segment, priorityMode);
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
                        result = _englishG2P.Process(segment, priorityMode);
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
                            if (DebugMode)
                                Console.WriteLine($"[MixedG2P] Using Japanese processor for: '{segment}'");
                            result = _japaneseG2P.Process(segment, priorityMode);
                            phoneLang = PhoneLanguage.Japanese;
                        }
                        prevLang = Language.Japanese;
                    }
                    else if (lang == Language.Punctuation || lang == Language.Symbol)
                    {
                        string punc = segment;
                        // 映射常见标点到静态音素表支持的符号
                        string mapped = punc switch
                        {
                            "，" or "、" or "；" or "：" => ",",
                            "。" or "！" or "？" or "!" or "?" or "." => punc == "。" ? "." : punc,
                            "《" or "》" or "【" or "】" or "（" or "）" or "(" or ")" or "[" or "]" => " ", // 结构化符号转为空格/边界
                            _ => punc
                        };

                        if (Symbols.Punctuation.Contains(mapped))
                        {
                            allPhones.Add(mapped);
                            allWord2Ph.Add(1);
                            languageTags.Add(PhoneLanguage.Other);
                        }
                        else if (mapped == " ")
                        {
                            // 结构化符号产生一个极其微小的边界感，不增加音素但更新文本
                        }

                        normalizedBuilder.Append(segment);
                        continue;
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
                if (DebugMode)
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

        private enum Language { Chinese, English, Japanese, Han, Numeric, Punctuation, Symbol, Other }

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

        private List<(string segment, Language lang)> SplitByLanguage(string text, Language? preference = null)
        {
            var result = new List<(string, Language)>();
            if (string.IsNullOrEmpty(text)) return result;

            // 1. 初步脚本识别
            var chars = new List<(char c, Language lang)>();
            foreach (char c in text)
            {
                chars.Add((c, DetectCharLanguage(c)));
            }

            // 2. 识别汉字片段的归属
            var segments = new List<(string text, Language lang)>();
            if (chars.Count == 0) return result;

            StringBuilder sb = new StringBuilder();
            Language currentType = chars[0].lang;
            sb.Append(chars[0].c);

            for (int i = 1; i < chars.Count; i++)
            {
                if (chars[i].lang == currentType)
                {
                    sb.Append(chars[i].c);
                }
                else
                {
                    segments.Add((sb.ToString(), currentType));
                    sb.Clear();
                    currentType = chars[i].lang;
                    sb.Append(chars[i].c);
                }
            }
            segments.Add((sb.ToString(), currentType));

            // 3. 上下文修正
            for (int i = 0; i < segments.Count; i++)
            {
                var seg = segments[i];
                if (seg.lang == Language.Han)
                {
                    Language detected = Language.Chinese; // 默认中文
                    if (preference == Language.Japanese && _allowedLanguages.Contains(Language.Japanese))
                        detected = Language.Japanese;
                    else
                    {
                        // 改进的上下文检测：检查前后是否有日语假名 (跳过标点和空格)
                        bool hasJaContext = false;

                        // 向上搜索最近的正文语种
                        for (int j = i - 1; j >= 0; j--)
                        {
                            if (segments[j].lang == Language.Japanese) { hasJaContext = true; break; }
                            // 如果遇到明确的其他语种，则停止搜索（除非它是标点/空格）
                            if (segments[j].lang == Language.Han || segments[j].lang == Language.Chinese || segments[j].lang == Language.English) break;
                        }

                        if (!hasJaContext)
                        {
                            // 向下搜索
                            for (int j = i + 1; j < segments.Count; j++)
                            {
                                if (segments[j].lang == Language.Japanese) { hasJaContext = true; break; }
                                if (segments[j].lang == Language.Han || segments[j].lang == Language.Chinese || segments[j].lang == Language.English) break;
                            }
                        }

                        // 兜底策略：如果整个输入包含假名，且由于标点切分导致局部探测失败，则倾向于日语
                        bool textHasKana = ContainsJapaneseKana(text);

                        if ((hasJaContext || textHasKana) && _allowedLanguages.Contains(Language.Japanese))
                            detected = Language.Japanese;
                        else
                            detected = Language.Chinese;
                    }
                    segments[i] = (seg.text, detected);
                }
                else if (seg.lang == Language.Numeric || seg.lang == Language.Other)
                {
                    // === 主体语言加权决策策略 (仅在允许范围内选择) ===
                    int zhCount = 0, enCount = 0, jaCount = 0;
                    foreach (var s in segments)
                    {
                        if ((s.lang == Language.Chinese || s.lang == Language.Han) && _allowedLanguages.Contains(Language.Chinese))
                            zhCount += s.text.Length * 3;
                        else if (s.lang == Language.English && _allowedLanguages.Contains(Language.English))
                            enCount += s.text.Where(c => !char.IsWhiteSpace(c)).Count();
                        else if (s.lang == Language.Japanese && _allowedLanguages.Contains(Language.Japanese))
                            jaCount += s.text.Length * 3;
                    }

                    Language dominantLang;
                    if (preference.HasValue && _allowedLanguages.Contains(preference.Value))
                        dominantLang = preference.Value;
                    else if (jaCount > 0 && jaCount >= zhCount && jaCount >= enCount)
                        dominantLang = Language.Japanese;
                    else if (zhCount >= enCount && zhCount > 0)
                        dominantLang = Language.Chinese;
                    else if (enCount > 0)
                        dominantLang = Language.English;
                    else
                        dominantLang = _allowedLanguages.Contains(Language.English) ? Language.English : _allowedLanguages.First();

                    Language context;

                    if (seg.lang == Language.Numeric)
                    {
                        bool hasCurrencyPrefix = false;
                        if (i > 0)
                        {
                            string prevText = segments[i - 1].text.Trim();
                            if ((prevText == "$" || prevText == "£" || prevText == "€") && _allowedLanguages.Contains(Language.English))
                                hasCurrencyPrefix = true;
                        }
                        context = hasCurrencyPrefix ? Language.English : dominantLang;
                    }
                    else
                    {
                        // Other (空格/符号)
                        context = Language.Other;
                        Language? prevLangCandidate = null, nextLangCandidate = null;

                        for (int j = i - 1, skipped = 0; j >= 0 && skipped < 2; j--)
                        {
                            var l = segments[j].lang;
                            if (l == Language.Han) l = Language.Chinese;
                            if (_allowedLanguages.Contains(l)) { prevLangCandidate = l; break; }
                            skipped++;
                        }

                        for (int j = i + 1, skipped = 0; j < segments.Count && skipped < 2; j++)
                        {
                            var l = segments[j].lang;
                            if (l == Language.Han) l = Language.Chinese;
                            if (_allowedLanguages.Contains(l)) { nextLangCandidate = l; break; }
                            skipped++;
                        }

                        if (prevLangCandidate.HasValue && nextLangCandidate.HasValue)
                            context = (prevLangCandidate == nextLangCandidate) ? prevLangCandidate.Value : dominantLang;
                        else if (prevLangCandidate.HasValue)
                            context = prevLangCandidate.Value;
                        else if (nextLangCandidate.HasValue)
                            context = nextLangCandidate.Value;
                        else
                            context = dominantLang;
                    }
                    segments[i] = (seg.text, context);
                }
            }

            // 4. 合并相同语言的相邻片段
            if (segments.Count == 0) return result;

            string currentText = segments[0].text;
            Language currentLang = segments[0].lang;

            for (int i = 1; i < segments.Count; i++)
            {
                if (segments[i].lang == currentLang)
                {
                    currentText += segments[i].text;
                }
                else
                {
                    result.Add((currentText, currentLang));
                    currentText = segments[i].text;
                    currentLang = segments[i].lang;
                }
            }
            result.Add((currentText, currentLang));

            // 5. 语言限制校验
            var detectedLangs = result
                .Select(r => r.Item2)
                .Where(l => l == Language.Chinese || l == Language.English || l == Language.Japanese)
                .Distinct()
                .ToList();

            if (detectedLangs.Count > 2)
            {
                throw new InvalidOperationException($"Input text contains too many languages ({string.Join(", ", detectedLangs)}). Maximum 2 languages are allowed.");
            }

            bool isExplicitJapanese = preference == Language.Japanese;
            bool isExplicitChinese = preference == Language.Chinese;
            bool isExplicitEnglish = preference == Language.English;

            foreach (var l in detectedLangs)
            {
                if (!_allowedLanguages.Contains(l))
                {
                    // === 豁免逻辑 ===
                    // 如果语种是显式指定的 (通过 explicitLanguage 参数)，则允许处理
                    bool isExplicit = (l == Language.Japanese && isExplicitJapanese) ||
                                     (l == Language.Chinese && isExplicitChinese) ||
                                     (l == Language.English && isExplicitEnglish);

                    if (!isExplicit)
                    {
                        throw new InvalidOperationException($"Language '{l}' is detected but not allowed in current configuration. Allowed: {string.Join(", ", _allowedLanguages)}");
                    }
                }
            }

            return result;
        }

        private Language DetectCharLanguage(char c)
        {
            if ((c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) ||
                (c >= 0x31F0 && c <= 0x31FF) || c == 0x30FC)
                return Language.Japanese;

            if ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF))
                return Language.Han;

            if (char.IsDigit(c) || c == '.') return Language.Numeric;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return Language.English;
            if (c == '$' || c == '£' || c == '€' || c == '@' || c == '&') return Language.English;

            string s = c.ToString();
            if (Symbols.Punctuation.Contains(s))
            {
                // 区分“呼吸/停顿”标点和“结构”标点
                if ("，。！？；：, . ! ? ; : 、…".Contains(s)) return Language.Punctuation;
                return Language.Symbol;
            }

            return Language.Other;
        }
    }
}

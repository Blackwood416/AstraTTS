using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AstraTTS.Core.Core;

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
            // 如果指定了日语，将所有汉字视为日语处理
            // 如果未指定语言但文本包含假名，也将汉字视为日语（上下文感知）
            bool forceJapanese = IsJapaneseLanguageCode(explicitLanguage)
                                 || (explicitLanguage == null && ContainsJapaneseKana(text));

            // 0. 先进行英文特殊符号规范化 (C# -> C sharp, .NET -> dot net 等)
            // 这样在分割语言时，特殊符号已经被转换为普通文本
            text = TextNorm.EnglishTextNormalizer.Normalize(text);

            // 1. 分割文本为语言片段
            var segments = SplitByLanguage(text, forceJapanese);

            List<string> allPhones = new List<string>();
            List<int> allWord2Ph = new List<int>();
            List<PhoneLanguage> languageTags = new List<PhoneLanguage>();  // 语言标记
            List<LanguageSegment> segmentList = new List<LanguageSegment>();  // 片段信息
            StringBuilder normalizedBuilder = new StringBuilder();

            Language? prevLang = null;

            foreach (var (segment, lang) in segments)
            {
                if (string.IsNullOrWhiteSpace(segment)) continue;

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
                    // 仅在 中文/日文->英文 过渡时添加 SP
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
                    // 日语处理
                    if (_japaneseG2P == null)
                    {
                        // 回退到中文 G2P（汉字部分可能有效）
                        result = _chineseG2P.Process(segment);
                        phoneLang = PhoneLanguage.Chinese;
                    }
                    else
                    {
                        // 语言过渡时添加 SP
                        if (prevLang.HasValue && prevLang != Language.Japanese)
                        {
                            allPhones.Add("SP");
                            allWord2Ph.Add(1);
                            languageTags.Add(PhoneLanguage.Other);
                        }
                        result = _japaneseG2P.Process(segment);
                        phoneLang = PhoneLanguage.Japanese;
                    }
                    prevLang = Language.Japanese;
                }
                else
                {
                    // 标点或其他字符
                    if (Symbols.Punctuation.Contains(segment))
                    {
                        allPhones.Add(segment);
                        allWord2Ph.Add(1);
                        languageTags.Add(PhoneLanguage.Other);
                    }
                    normalizedBuilder.Append(segment);
                    continue;
                }

                // 记录片段信息 (用于分段 BERT)
                int startIdx = allPhones.Count;
                int phoneCount = result.Phones.Count;

                // 添加音素和对应的语言标记
                allPhones.AddRange(result.Phones);
                allWord2Ph.AddRange(result.Word2Ph);
                for (int i = 0; i < phoneCount; i++)
                {
                    languageTags.Add(phoneLang);
                }
                normalizedBuilder.Append(result.NormalizedText);

                // 保存片段信息
                segmentList.Add(new LanguageSegment
                {
                    Text = result.NormalizedText,
                    Language = phoneLang,
                    StartPhoneIndex = startIdx,
                    PhoneCount = phoneCount,
                    Word2Ph = result.Word2Ph
                });
            }

            // 添加尾部标点，防止模型截断最后一个音素
            if (allPhones.Count > 0 && !Symbols.Punctuation.Contains(allPhones.Last()))
            {
                allPhones.Add(".");
                allWord2Ph.Add(1);
                languageTags.Add(PhoneLanguage.Other);
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

        private enum Language { Chinese, English, Japanese, Other }

        /// <summary>
        /// 检测语言代码是否为日语。
        /// </summary>
        private static bool IsJapaneseLanguageCode(string? langCode)
        {
            if (string.IsNullOrEmpty(langCode)) return false;
            var code = langCode.ToLowerInvariant();
            return code == "jp" || code == "ja" || code == "jpn" || code == "japanese";
        }

        /// <summary>
        /// 将文本分割为语言片段。
        /// </summary>
        /// <param name="text">要分割的文本</param>
        /// <param name="forceJapanese">如果为 true，将所有 CJK 汉字视为日语</param>
        private List<(string segment, Language lang)> SplitByLanguage(string text, bool forceJapanese = false)
        {
            var result = new List<(string, Language)>();
            var currentSegment = new StringBuilder();
            Language? currentLang = null;

            foreach (char c in text)
            {
                Language charLang = DetectCharLanguage(c, forceJapanese);

                if (currentLang == null)
                {
                    currentLang = charLang;
                    currentSegment.Append(c);
                }
                else if (charLang == currentLang || charLang == Language.Other)
                {
                    // 同语言或标点，继续累积
                    currentSegment.Append(c);
                }
                else
                {
                    // 语言切换，保存当前片段
                    if (currentSegment.Length > 0)
                    {
                        result.Add((currentSegment.ToString(), currentLang.Value));
                        currentSegment.Clear();
                    }
                    currentLang = charLang;
                    currentSegment.Append(c);
                }
            }

            // 保存最后一个片段
            if (currentSegment.Length > 0 && currentLang.HasValue)
            {
                result.Add((currentSegment.ToString(), currentLang.Value));
            }

            return result;
        }
        /// <summary>
        /// 检测文本中是否包含日语假名（平假名或片假名）。
        /// 保留方法供将来使用。
        /// </summary>
        private bool ContainsJapaneseKana(string text)
        {
            foreach (char c in text)
            {
                // 平假名 (3040-309F)
                if (c >= 0x3040 && c <= 0x309F) return true;
                // 片假名 (30A0-30FF)
                if (c >= 0x30A0 && c <= 0x30FF) return true;
                // 片假名扩展 (31F0-31FF)
                if (c >= 0x31F0 && c <= 0x31FF) return true;
            }
            return false;
        }

        /// <summary>
        /// 检测单个字符的语言。
        /// </summary>
        /// <param name="c">要检测的字符</param>
        /// <param name="forceJapanese">如果为 true，将 CJK 汉字视为日语</param>
        private Language DetectCharLanguage(char c, bool forceJapanese = false)
        {
            // 日语平假名 (3040-309F)
            if (c >= 0x3040 && c <= 0x309F) return Language.Japanese;
            // 日语片假名 (30A0-30FF)
            if (c >= 0x30A0 && c <= 0x30FF) return Language.Japanese;
            // 日语片假名扩展 (31F0-31FF)
            if (c >= 0x31F0 && c <= 0x31FF) return Language.Japanese;
            // 日语长音符号
            if (c == 0x30FC) return Language.Japanese;

            // CJK 汉字范围 - 根据 forceJapanese 决定语言
            if (c >= 0x4E00 && c <= 0x9FFF)
            {
                return forceJapanese ? Language.Japanese : Language.Chinese;
            }
            // CJK Extension A
            if (c >= 0x3400 && c <= 0x4DBF)
            {
                return forceJapanese ? Language.Japanese : Language.Chinese;
            }

            // 数字和小数点 - 根据 forceJapanese 决定
            if ((c >= '0' && c <= '9') || c == '.')
            {
                return forceJapanese ? Language.Japanese : Language.Chinese;
            }

            // ASCII 字母
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return Language.English;

            // 其他 (标点、空格等)
            return Language.Other;
        }
    }
}

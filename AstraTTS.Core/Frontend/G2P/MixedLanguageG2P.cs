using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AstraTTS.Core.Core;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 混合语言 G2P 处理器，自动检测中英文片段并路由到对应的 G2P。
    /// </summary>
    public class MixedLanguageG2P : IG2P
    {
        private readonly ChineseG2P _chineseG2P;
        private readonly EnglishG2P _englishG2P;

        public MixedLanguageG2P(ChineseG2P chineseG2P, EnglishG2P englishG2P)
        {
            _chineseG2P = chineseG2P;
            _englishG2P = englishG2P;
        }

        /// <summary>
        /// 处理混合语言文本。
        /// </summary>
        public G2PResult Process(string text)
        {
            // 0. 先进行英文特殊符号规范化 (C# -> C sharp, .NET -> dot net 等)
            // 这样在分割语言时，特殊符号已经被转换为普通文本
            text = TextNorm.EnglishTextNormalizer.Normalize(text);

            // 1. 分割文本为中英文片段
            var segments = SplitByLanguage(text);

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
                    // 仅在 中文->英文 过渡时添加 SP
                    if (prevLang == Language.Chinese)
                    {
                        allPhones.Add("SP");
                        allWord2Ph.Add(1);
                        languageTags.Add(PhoneLanguage.Other);  // SP 标记为 Other
                    }
                    result = _englishG2P.Process(segment);
                    phoneLang = PhoneLanguage.English;
                    prevLang = Language.English;
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

        private enum Language { Chinese, English, Other }

        /// <summary>
        /// 将文本分割为语言片段。
        /// </summary>
        private List<(string segment, Language lang)> SplitByLanguage(string text)
        {
            var result = new List<(string, Language)>();
            var currentSegment = new StringBuilder();
            Language? currentLang = null;

            foreach (char c in text)
            {
                Language charLang = DetectCharLanguage(c);

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
        /// 检测单个字符的语言。
        /// </summary>
        private Language DetectCharLanguage(char c)
        {
            // 中文字符范围 (CJK Unified Ideographs)
            if (c >= 0x4E00 && c <= 0x9FFF) return Language.Chinese;
            // CJK Extension A
            if (c >= 0x3400 && c <= 0x4DBF) return Language.Chinese;

            // 数字和小数点 - 归类为中文，让 ChineseTextNormalizer 处理
            if ((c >= '0' && c <= '9') || c == '.') return Language.Chinese;

            // ASCII 字母
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return Language.English;

            // 其他 (标点、空格等)
            return Language.Other;
        }
    }
}

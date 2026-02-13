using System.Linq;

namespace AstraTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// 语言检测和文本预处理工具
    /// </summary>
    public static class LanguageDetector
    {
        /// <summary>
        /// 语言模式
        /// </summary>
        public enum LanguageMode
        {
            Chinese,  // 纯中文
            English,  // 纯英文
            Mixed,    // 中英混合
            Japanese  // 包含日语假名
        }

        /// <summary>
        /// 检测文本的语言模式
        /// </summary>
        public static LanguageMode DetectMode(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return LanguageMode.English;

            bool hasChinese = false;
            bool hasEnglish = false;
            bool hasJapanese = false;

            foreach (char c in text)
            {
                // 日语平假名 (U+3040-U+309F)
                if (c >= 0x3040 && c <= 0x309F)
                {
                    hasJapanese = true;
                }
                // 日语片假名 (U+30A0-U+30FF)
                else if (c >= 0x30A0 && c <= 0x30FF)
                {
                    hasJapanese = true;
                }
                // 中文字符 (CJK Unified Ideographs + Extension A)
                else if ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF))
                {
                    hasChinese = true;
                }
                // 英文字母
                else if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                {
                    hasEnglish = true;
                }
                // 其他字符（数字、标点等）不参与判断
            }

            // 日语优先级最高（包含假名即视为日语）
            if (hasJapanese) return LanguageMode.Japanese;

            if (hasChinese && hasEnglish) return LanguageMode.Mixed;
            if (hasChinese && !hasEnglish) return LanguageMode.Chinese;
            if (hasEnglish && !hasChinese) return LanguageMode.English;

            // 既没中文也没英文（纯数字/标点），默认英文模式
            return LanguageMode.English;
        }

        /// <summary>
        /// 将全角标点转换为半角标点
        /// </summary>
        public static string NormalizePunctuation(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            return text
                .Replace('\uFF0C', ',')  // \uFF0C = \uFF0C
                .Replace('\u3002', '.')  // \u3002 = \u3002
                .Replace('\uFF01', '!')  // \uFF01 = \uFF01
                .Replace('\uFF1F', '?')  // \uFF1F = \uFF1F
                .Replace('\uFF1A', ':')  // \uFF1A = \uFF1A
                .Replace('\uFF1B', ';')  // \uFF1B = \uFF1B
                .Replace('\u201C', '"')  // \u201C = \u201C
                .Replace('\u201D', '"')  // \u201D = \u201D
                .Replace('\u2018', '\'')  // \u2018 = \u2018
                .Replace('\u2019', '\'')  // \u2019 = \u2019
                .Replace('\uFF08', '(')  // \uFF08 = \uFF08
                .Replace('\uFF09', ')')  // \uFF09 = \uFF09
                .Replace('\u3010', '[')  // \u3010 = \u3010
                .Replace('\u3011', ']')  // \u3011 = \u3011
                .Replace('\u300A', '<')  // \u300A = \u300A
                .Replace('\u300B', '>'); // \u300B = \u300B
        }

        /// <summary>
        /// 将长文本智能切分为句子列表，支持中英文标点及换行
        /// </summary>
        public static System.Collections.Generic.List<string> SplitSentences(string text)
        {
            var sentences = new System.Collections.Generic.List<string>();
            if (string.IsNullOrWhiteSpace(text)) return sentences;

            var sb = new System.Text.StringBuilder();
            char[] splitChars = { '。', '！', '？', '.', '!', '?', '\n', '\r' };
            bool inTag = false;

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                // 追踪是否在标签内 {}
                if (c == '{') inTag = true;
                else if (c == '}') inTag = false;

                sb.Append(c);

                if (!inTag && System.Array.IndexOf(splitChars, c) >= 0)
                {
                    // 小数点逻辑：如果是 . 且前后都是数字，则不视为句号
                    if (c == '.')
                    {
                        bool isDigitBefore = i > 0 && char.IsDigit(text[i - 1]);
                        bool isDigitAfter = i < text.Length - 1 && char.IsDigit(text[i + 1]);
                        if (isDigitBefore && isDigitAfter) continue;
                    }

                    string s = sb.ToString().Trim();
                    if (!string.IsNullOrEmpty(s))
                    {
                        sentences.Add(s);
                    }
                    sb.Clear();
                }
            }

            string remaining = sb.ToString().Trim();
            if (!string.IsNullOrEmpty(remaining))
            {
                sentences.Add(remaining);
            }

            return sentences;
        }
    }
}

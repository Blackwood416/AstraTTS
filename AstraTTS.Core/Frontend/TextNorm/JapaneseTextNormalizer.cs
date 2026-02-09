using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// 日语文本规范化器
    /// </summary>
    public static class JapaneseTextNormalizer
    {
        // 特殊符号转换为日语读法
        private static readonly (Regex Pattern, string Replacement)[] SymbolReplacements =
        {
            (new Regex("%"), "パーセント"),
            (new Regex("％"), "パーセント"),
            (new Regex("&"), "アンド"),
            (new Regex("＆"), "アンド"),
        };

        // 全角标点 -> 半角标点
        private static readonly Dictionary<char, char> PunctuationMap = new Dictionary<char, char>
        {
            { '。', '.' },
            { '、', ',' },
            { '！', '!' },
            { '？', '?' },
            { '：', ':' },
            { '；', ';' },
            { '\u201C', '"' },  // "
            { '\u201D', '"' },  // "
            { '\u2018', '\'' }, // '
            { '\u2019', '\'' }, // '
            { '（', '(' },
            { '）', ')' },
            { '【', '[' },
            { '】', ']' },
            { '《', '<' },
            { '》', '>' },
            { '「', '"' },
            { '」', '"' },
            { '『', '"' },
            { '』', '"' },
            { '〜', '~' },
            { '～', '~' },
        };

        // 连续标点合并正则
        private static readonly Regex ConsecutivePunctuation = new Regex(@"([,./?!~…・])\1+");

        /// <summary>
        /// 规范化日语文本
        /// </summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return text;

            // 1. 特殊符号转换
            foreach (var (pattern, replacement) in SymbolReplacements)
            {
                text = pattern.Replace(text, replacement);
            }

            // 2. 全角标点转半角
            var chars = text.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
            {
                if (PunctuationMap.TryGetValue(chars[i], out char replacement))
                {
                    chars[i] = replacement;
                }
            }
            text = new string(chars);

            // 3. 合并连续标点
            text = ConsecutivePunctuation.Replace(text, "$1");

            return text;
        }

        /// <summary>
        /// 标点后处理替换
        /// </summary>
        public static string PostReplacePunctuation(string phoneme)
        {
            return phoneme switch
            {
                "：" or "；" or "，" or "、" or "·" => ",",
                "。" => ".",
                "！" => "!",
                "？" => "?",
                "\n" => ".",
                "..." => "…",
                _ => phoneme
            };
        }
    }
}

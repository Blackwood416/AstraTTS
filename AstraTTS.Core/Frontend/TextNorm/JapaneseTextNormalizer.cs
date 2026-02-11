using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// 日语文本规范化器
    /// </summary>
    public static class JapaneseTextNormalizer
    {
        // 简中 -> 日文汉字常见变体转换 (防止 OpenJTalk 字典查表失败)
        private static readonly Dictionary<char, char> KanjiVariantMap = new Dictionary<char, char>
        {
            { '场', '場' }, { '语', '語' }, { '发', '発' }, { '经', '経' },
            { '过', '過' }, { '进', '進' }, { '开', '開' }, { '实', '実' },
            { '现', '現' }, { '对', '対' }, { '说', '説' }, { '话', '話' },
            { '长', '長' }, { '门', '門' }, { '间', '間' }, { '问', '問' },
            { '题', '題' }, { '东', '東' }, { '应', '応' }, { '该', '該' },
        };

        // 连续标点合并正则 (防止过长的停顿导致模型异常)
        private static readonly Regex ConsecutivePunctuation = new Regex(@"([,./?!~…・。、！？：；〜～])\1+", RegexOptions.Compiled);

        /// <summary>
        /// 规范化日语文本
        /// </summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return text;

            // 1. 快速符号替换 (避免正则开销)
            text = text.Replace("%", "パーセント").Replace("％", "パーセント")
                       .Replace("&", "アンド").Replace("＆", "アンド");

            // 2. 汉字变体转换 (简中 -> 日文汉字) 和 标点还原 (全角)
            // 使用 StringBuilder 减少分配
            var sb = new StringBuilder(text.Length);
            foreach (char c in text)
            {
                if (KanjiVariantMap.TryGetValue(c, out char replaced)) sb.Append(replaced);
                else
                {
                    // 标点统一还原为全角，OpenJTalk 对全角标点支持更好
                    char target = c switch
                    {
                        '.' => '。',
                        ',' => '、',
                        '!' => '！',
                        '?' => '？',
                        ':' => '：',
                        ';' => '；',
                        _ => c
                    };
                    sb.Append(target);
                }
            }
            text = sb.ToString();

            // 3. 合并连续标点
            text = ConsecutivePunctuation.Replace(text, "$1");

            return text;
        }

        /// <summary>
        /// 标点后处理替换 (用于音素映射阶段)
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

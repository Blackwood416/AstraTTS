using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.G2P.English
{
    public enum TextSegmentType { Normal, Native }

    public class TextSegment
    {
        public string Text { get; set; } = string.Empty;
        public string? Language { get; set; }
        public TextSegmentType Type { get; set; } = TextSegmentType.Normal;
    }

    /// <summary>
    /// 解析带标签的文本，识别 {zh 内容} 语言切换及 {ni3}{hao3} 原生音素块。
    /// </summary>
    public static class TaggedTextParser
    {
        // 匹配 {lang 内容}，lang 为 2-3 位字母，内容可以包含空格，非贪婪匹配
        private static readonly Regex UnifiedTagRegex = new Regex(@"{([a-zA-Z]{2,3})\s+(.*?)}", RegexOptions.Compiled | RegexOptions.Singleline);
        // 匹配 {ni3} 原生音素，由 MixedLanguageG2P 最终分发
        private static readonly Regex NativeTagRegex = new Regex(@"({[^{}\s]+})", RegexOptions.Compiled);

        public static List<TextSegment> Parse(string text)
        {
            var result = new List<TextSegment>();
            if (string.IsNullOrEmpty(text)) return result;

            int lastIdx = 0;
            var matches = UnifiedTagRegex.Matches(text);



            foreach (Match match in matches)
            {
                if (match.Index > lastIdx)
                {
                    ProcessNative(text.Substring(lastIdx, match.Index - lastIdx), null, result);
                }

                string lang = match.Groups[1].Value;
                string content = match.Groups[2].Value;



                // 语言标签内部的内容也可能包含原生音素块，例如 {zh GPT {ni3}}
                // 但为了简单，目前暂不支持嵌套，仅作为普通文本处理
                result.Add(new TextSegment { Text = content, Language = lang, Type = TextSegmentType.Normal });

                lastIdx = match.Index + match.Length;
            }

            if (lastIdx < text.Length)
            {
                ProcessNative(text.Substring(lastIdx), null, result);
            }

            return result;
        }

        private static void ProcessNative(string text, string? lang, List<TextSegment> result)
        {
            if (string.IsNullOrEmpty(text)) return;

            int lastIdx = 0;
            var matches = NativeTagRegex.Matches(text);
            foreach (Match match in matches)
            {
                if (match.Index > lastIdx)
                {
                    result.Add(new TextSegment { Text = text.Substring(lastIdx, match.Index - lastIdx), Language = lang, Type = TextSegmentType.Normal });
                }
                result.Add(new TextSegment { Text = match.Value, Language = lang, Type = TextSegmentType.Native });
                lastIdx = match.Index + match.Length;
            }
            if (lastIdx < text.Length)
            {
                result.Add(new TextSegment { Text = text.Substring(lastIdx), Language = lang, Type = TextSegmentType.Normal });
            }
        }
    }
}

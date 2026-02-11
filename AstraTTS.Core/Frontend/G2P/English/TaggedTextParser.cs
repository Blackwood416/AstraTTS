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
    /// 解析带标签的文本，识别 [lang=zh]...[/lang] 及 {ni3}{hao3} 原生音素块。
    /// </summary>
    public static class TaggedTextParser
    {
        private static readonly Regex LangTagRegex = new Regex(@"\[lang=([a-zA-Z]+)\](.*?)\[/lang\]", RegexOptions.Compiled | RegexOptions.Singleline);
        private static readonly Regex NativeTagRegex = new Regex(@"({[^{}]+})", RegexOptions.Compiled);

        public static List<TextSegment> Parse(string text)
        {
            var result = new List<TextSegment>();
            if (string.IsNullOrEmpty(text)) return result;

            int lastIdx = 0;
            var matches = LangTagRegex.Matches(text);
            foreach (Match match in matches)
            {
                if (match.Index > lastIdx)
                {
                    ProcessNormalAndNative(text.Substring(lastIdx, match.Index - lastIdx), null, result);
                }

                string lang = match.Groups[1].Value;
                string content = match.Groups[2].Value;
                ProcessNormalAndNative(content, lang, result);
                lastIdx = match.Index + match.Length;
            }

            if (lastIdx < text.Length)
            {
                ProcessNormalAndNative(text.Substring(lastIdx), null, result);
            }

            return result;
        }

        private static void ProcessNormalAndNative(string text, string? lang, List<TextSegment> result)
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

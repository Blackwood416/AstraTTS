using System;
using System.Text;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// 中文文本正规化器，将阿拉伯数字、百分比、日期等转换为中文读法。
    /// </summary>
    public static class ChineseTextNormalizer
    {
        private static readonly string[] ChineseDigits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
        private static readonly string[] ChineseUnits = { "", "十", "百", "千" };
        private static readonly string[] ChineseBigUnits = { "", "万", "亿", "万亿" };

        /// <summary>
        /// 主入口：对文本进行全面正规化处理。
        /// </summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            // 1. 百分比 (e.g., 50% -> 百分之五十)
            text = Regex.Replace(text, @"(\d+(?:\.\d+)?)\s*[%％]", m => ConvertPercentage(m.Groups[1].Value));

            // 2. 年份 (e.g., 2023年 -> 二零二三年)
            text = Regex.Replace(text, @"(\d{4})年", m => ConvertYear(m.Groups[1].Value) + "年");

            // 3. 日期 月日 (e.g., 10月1日 -> 十月一日)
            text = Regex.Replace(text, @"(\d{1,2})月(\d{1,2})日", m => 
                ConvertCardinal(m.Groups[1].Value) + "月" + ConvertCardinal(m.Groups[2].Value) + "日");

            // 4. 序数词 (e.g., 第1名 -> 第一名)
            text = Regex.Replace(text, @"第(\d+)", m => "第" + ConvertCardinal(m.Groups[1].Value));

            // 5. 小数 (e.g., 3.14 -> 三点一四)
            text = Regex.Replace(text, @"(\d+)\.(\d+)", m => ConvertDecimal(m.Groups[1].Value, m.Groups[2].Value));

            // 6. 普通整数 (e.g., 123 -> 一百二十三)
            // 必须放在最后，避免干扰上面的模式
            text = Regex.Replace(text, @"\d+", m => ConvertCardinal(m.Value));

            return text;
        }

        /// <summary>
        /// 转换普通整数 (基数词)。
        /// 例如: 123 -> 一百二十三
        /// </summary>
        public static string ConvertCardinal(string numStr)
        {
            if (string.IsNullOrEmpty(numStr)) return "";

            // 去除前导零
            numStr = numStr.TrimStart('0');
            if (numStr.Length == 0) return "零";

            if (!long.TryParse(numStr, out long num)) return numStr;

            if (num == 0) return "零";
            if (num < 0) return "负" + ConvertCardinal((-num).ToString());

            var result = new StringBuilder();
            int groupIndex = 0;

            while (num > 0)
            {
                int group = (int)(num % 10000);
                if (group > 0)
                {
                    string groupStr = ConvertFourDigits(group);
                    result.Insert(0, groupStr + ChineseBigUnits[groupIndex]);
                }
                else if (result.Length > 0 && !result.ToString().StartsWith("零"))
                {
                    result.Insert(0, "零");
                }
                num /= 10000;
                groupIndex++;
            }

            // 清理多余的零
            string final = result.ToString();
            final = Regex.Replace(final, "零+", "零");
            final = final.TrimEnd('零');
            if (final.Length == 0) final = "零";

            // 特殊规则: 一十 -> 十 (仅用于 10-19)
            if (final.StartsWith("一十") && final.Length <= 3)
            {
                final = final.Substring(1);
            }

            return final;
        }

        /// <summary>
        /// 转换四位以内的数字组。
        /// </summary>
        private static string ConvertFourDigits(int num)
        {
            if (num == 0) return "";

            var sb = new StringBuilder();
            int[] digits = new int[4];
            for (int i = 0; i < 4; i++)
            {
                digits[3 - i] = num % 10;
                num /= 10;
            }

            bool needZero = false;
            for (int i = 0; i < 4; i++)
            {
                if (digits[i] != 0)
                {
                    if (needZero) sb.Append("零");
                    sb.Append(ChineseDigits[digits[i]]);
                    sb.Append(ChineseUnits[3 - i]);
                    needZero = false;
                }
                else
                {
                    if (sb.Length > 0) needZero = true;
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// 转换小数。
        /// 例如: 3.14 -> 三点一四
        /// </summary>
        public static string ConvertDecimal(string intPart, string decPart)
        {
            var sb = new StringBuilder();
            sb.Append(ConvertCardinal(intPart));
            sb.Append("点");
            
            // 小数部分逐位读
            foreach (char c in decPart)
            {
                if (char.IsDigit(c))
                {
                    sb.Append(ChineseDigits[c - '0']);
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// 转换年份 (逐位读)。
        /// 例如: 2023 -> 二零二三
        /// </summary>
        public static string ConvertYear(string yearStr)
        {
            var sb = new StringBuilder();
            foreach (char c in yearStr)
            {
                if (char.IsDigit(c))
                {
                    sb.Append(ChineseDigits[c - '0']);
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// 转换百分比。
        /// 例如: 50 -> 百分之五十
        /// </summary>
        public static string ConvertPercentage(string numStr)
        {
            if (numStr.Contains("."))
            {
                var parts = numStr.Split('.');
                return "百分之" + ConvertDecimal(parts[0], parts[1]);
            }
            return "百分之" + ConvertCardinal(numStr);
        }
    }
}

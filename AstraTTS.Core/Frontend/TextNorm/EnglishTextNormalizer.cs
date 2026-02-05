using System;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// 英文文本正规化器，将数字转换为英文读法。
    /// </summary>
    public static class EnglishTextNormalizer
    {
        private static readonly string[] Ones = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                                  "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                                  "seventeen", "eighteen", "nineteen" };
        private static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

        /// <summary>
        /// 正规化英文文本中的数字和特殊符号。
        /// </summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            // === 特殊符号转换 ===
            // 编程语言名称 - 使用 (?![A-Za-z]) 替代 \b，因为中文在.NET正则中被视为word character
            text = Regex.Replace(text, @"(?<![A-Za-z])C#(?![A-Za-z])", "C sharp", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"(?<![A-Za-z])F#(?![A-Za-z])", "F sharp", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"(?<![A-Za-z])C\+\+(?![A-Za-z])", "C plus plus", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.NET(?![A-Za-z])", "dot net", RegexOptions.IgnoreCase);
            
            // === URL/域名/邮箱处理 ===
            // URL 协议
            text = Regex.Replace(text, @"https://", "H T T P S ", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"http://", "H T T P ", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"www\.", "W W W dot ", RegexOptions.IgnoreCase);
            
            // 常见顶级域名 (必须在通用 @ 转换之前)
            text = Regex.Replace(text, @"\.com\b", " dot com", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.org\b", " dot org", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.net\b", " dot net", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.io\b", " dot I O", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.cn\b", " dot C N", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.edu\b", " dot E D U", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.gov\b", " dot gov", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.co\b", " dot C O", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.ai\b", " dot A I", RegexOptions.IgnoreCase);
            
            // 常见符号
            text = Regex.Replace(text, @"@", " at ");
            text = Regex.Replace(text, @"&", " and ");
            text = Regex.Replace(text, @"\+", " plus ");
            text = Regex.Replace(text, @"=", " equals ");
            text = Regex.Replace(text, @"%", " percent ");
            text = Regex.Replace(text, @"\$", " dollar ");
            text = Regex.Replace(text, @"#(?!\s|$)", " hash ");  // # 后面不是空格或结尾
            text = Regex.Replace(text, @"#(?=\s|$)", " ");       // 单独的 # 静音
            
            // === 版本号处理 ===
            // 语义版本号 x.y.z (e.g., v0.24.0 -> version zero point two four point zero)
            text = Regex.Replace(text, @"\bv?(\d+)\.(\d+)\.(\d+)\b", m => 
                "version " + ConvertVersionDigits(m.Groups[1].Value) + 
                " point " + ConvertVersionDigits(m.Groups[2].Value) + 
                " point " + ConvertVersionDigits(m.Groups[3].Value), RegexOptions.IgnoreCase);
            
            // 简单版本号 v1.0 (e.g., v1.0 -> version one point zero)
            text = Regex.Replace(text, @"\bv(\d+)\.(\d+)\b", m => 
                "version " + ConvertVersionDigits(m.Groups[1].Value) + 
                " point " + ConvertVersionDigits(m.Groups[2].Value), RegexOptions.IgnoreCase);
            
            // 单版本号 v2 (e.g., v2 -> version two)
            text = Regex.Replace(text, @"\bv(\d+)\b", m => 
                "version " + ConvertNumber(m.Groups[1].Value), RegexOptions.IgnoreCase);

            // === 数字转换 (仅限紧邻英文字母的数字) ===
            // 小数紧邻英文 (e.g., GPT3.5 -> GPT three point five)
            text = Regex.Replace(text, @"(?<=[A-Za-z])(\d+)\.(\d+)", m => " " + ConvertDecimal(m.Groups[1].Value, m.Groups[2].Value) + " ");
            text = Regex.Replace(text, @"(\d+)\.(\d+)(?=[A-Za-z])", m => " " + ConvertDecimal(m.Groups[1].Value, m.Groups[2].Value) + " ");

            // 整数紧邻英文 (e.g., GPT4 -> GPT four, v2 -> v two)
            // 数字后面紧跟英文字母 或 英文字母后面紧跟数字
            text = Regex.Replace(text, @"(?<=[A-Za-z])(\d+)", m => " " + ConvertNumber(m.Groups[1].Value));
            text = Regex.Replace(text, @"(\d+)(?=[A-Za-z])", m => ConvertNumber(m.Groups[1].Value) + " ");
            
            // 清理多余空格
            text = Regex.Replace(text, @"\s+", " ").Trim();

            return text;
        }

        private static string ConvertNumber(string numStr)
        {
            if (!long.TryParse(numStr, out long num)) return numStr;
            if (num == 0) return "zero";

            if (num < 0) return "negative " + ConvertNumber((-num).ToString());

            string result = "";

            if (num >= 1000000000)
            {
                result += ConvertNumber((num / 1000000000).ToString()) + " billion ";
                num %= 1000000000;
            }
            if (num >= 1000000)
            {
                result += ConvertNumber((num / 1000000).ToString()) + " million ";
                num %= 1000000;
            }
            if (num >= 1000)
            {
                result += ConvertNumber((num / 1000).ToString()) + " thousand ";
                num %= 1000;
            }
            if (num >= 100)
            {
                result += Ones[num / 100] + " hundred ";
                num %= 100;
            }
            if (num >= 20)
            {
                result += Tens[num / 10] + " ";
                num %= 10;
            }
            if (num > 0)
            {
                result += Ones[num] + " ";
            }

            return result.Trim();
        }

        private static string ConvertDecimal(string intPart, string decPart)
        {
            string result = ConvertNumber(intPart) + " point";
            foreach (char c in decPart)
            {
                if (char.IsDigit(c))
                {
                    int digit = c - '0';
                    // 特别处理 0，因为 Ones[0] 是空字符串
                    result += " " + (digit == 0 ? "zero" : Ones[digit]);
                }
            }
            return result.Trim();
        }

        /// <summary>
        /// 转换版本号数字：逐位读出 (24 -> two four, 0 -> zero)
        /// </summary>
        private static string ConvertVersionDigits(string numStr)
        {
            if (string.IsNullOrEmpty(numStr)) return "";
            
            var result = new System.Text.StringBuilder();
            foreach (char c in numStr)
            {
                if (char.IsDigit(c))
                {
                    int digit = c - '0';
                    if (result.Length > 0) result.Append(" ");
                    result.Append(digit == 0 ? "zero" : Ones[digit]);
                }
            }
            return result.ToString();
        }
    }
}

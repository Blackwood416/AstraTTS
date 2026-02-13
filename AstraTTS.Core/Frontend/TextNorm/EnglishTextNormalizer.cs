using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.TextNorm
{
    /// <summary>
    /// 英文文本正规化器，将数字、货币、日期、缩写等转换为英文读法。
    /// 移植自 Genie-TTS 的 Normalization.py。
    /// </summary>
    public static class EnglishTextNormalizer
    {
        private static readonly string[] Ones = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                                  "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                                  "seventeen", "eighteen", "nineteen" };
        private static readonly string[] Tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
        private static readonly string[] Thousands = { "", "thousand", "million", "billion", "trillion" };

        private static readonly Dictionary<string, (string singular, string plural)> MeasurementMap = new Dictionary<string, (string, string)>
        {
            { "km/h", ("kilometer per hour", "kilometers per hour") },
            { "mph", ("mile per hour", "miles per hour") },
            { "°C", ("degree celsius", "degrees celsius") },
            { "°F", ("degree fahrenheit", "degrees fahrenheit") },
            { "tbsp", ("tablespoon", "tablespoons") },
            { "tsp", ("teaspoon", "teaspoons") },
            { "km", ("kilometer", "kilometers") },
            { "kg", ("kilogram", "kilograms") },
            { "min", ("minute", "minutes") },
            { "ft", ("foot", "feet") },
            { "cm", ("centimeter", "centimeters") },
            { "m", ("meter", "meters") },
            { "L", ("liter", "liters") },
            { "h", ("hour", "hours") },
            { "s", ("second", "seconds") },
        };

        private static readonly Dictionary<string, string> AbbreviationMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "Mr.", "Mister" },
            { "Mrs.", "Missus" },
            { "Dr.", "Doctor" },
            { "Prof.", "Professor" },
            { "St.", "Street" },
            { "Co.", "Company" },
            { "Ltd.", "Limited" },
            { "e.g.", "for example" },
            { "i.e.", "that is" },
        };

        private static readonly Dictionary<string, string> RomanMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "ii", "two" }, { "iii", "three" }, { "iv", "four" }, { "v", "five" }, { "vi", "six" }, { "vii", "seven" },
            { "viii", "eight" }, { "ix", "nine" }, { "x", "ten" }, { "xi", "eleven" }, { "xii", "twelve" },
            { "xiii", "thirteen" }, { "xiv", "fourteen" }, { "xv", "fifteen" }, { "xvi", "sixteen" },
            { "xvii", "seventeen" }, { "xviii", "eighteen" }, { "xix", "nineteen" }
        };

        private static readonly string MeasurementUnitsRegexPart = string.Join("|", MeasurementMap.Keys);

        /// <summary>
        /// 全量正规化：处理缩写、货币、手机号、日期、时间、度量衡、数字等。
        /// </summary>
        public static string Normalize(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;

            // 1. 去除重音符号 (简化的 NFD 处理)
            text = text.Normalize(NormalizationForm.FormD);
            StringBuilder sb = new StringBuilder();
            foreach (char c in text)
            {
                if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
                    sb.Append(c);
            }
            text = sb.ToString().Normalize(NormalizationForm.FormC);

            // 2. 符号归一化 (为语种分割做准备的部分)
            text = NormalizeSymbols(text);

            // 3. 高级正则处理
            text = ExpandAbbreviations(text);
            text = ExpandCurrencySuffix(text); // $5 million
            text = ExpandPhoneNumbers(text);
            text = ExpandDimensions(text); // 10x20
            text = ExpandRomanNumerals(text);
            text = ExpandDecades(text); // 1990s
            text = ExpandScores(text); // 3-1
            text = ExpandDates(text); // 10/1/2023
            text = ExpandTimes(text); // 10:30pm
            text = ExpandCommaNumbers(text); // 1,000
            text = ExpandCurrencies(text); // $5.20
            text = ExpandMeasurements(text); // 5km
            text = ExpandFractions(text); // 1/2
            text = ExpandDecimals(text); // 3.14
            text = ExpandAlphanumeric(text); // GPT4
            // text = ExpandAcronyms(text); // HTML - Deprecated, G2P handles this better

            // 4. 数字归一化 (兜底)
            text = NormalizeNumbers(text);

            // 保持原文，仅清理不可见的控制字符，由后续各自的 G2P 决定清洗策略
            text = Regex.Replace(text, @"[\u0000-\u001F\u007F-\u009F]", " ");
            text = Regex.Replace(text, @"\s+", " ").Trim();

            return text;
        }

        public static string NormalizeSymbols(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;
            text = Regex.Replace(text, @"(?<![A-Za-z])C#(?![A-Za-z])", "C sharp", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"(?<![A-Za-z])F#(?![A-Za-z])", "F sharp", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"(?<![A-Za-z])C\+\+(?![A-Za-z])", "C plus plus", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"\.NET(?![A-Za-z])", "dot net", RegexOptions.IgnoreCase);
            text = Regex.Replace(text, @"@", " at ");
            text = Regex.Replace(text, @"&", " and ");
            text = Regex.Replace(text, @"%", " percent ");
            return text;
        }

        public static string NormalizeNumbers(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;
            // 兜底处理所有剩余数字
            return Regex.Replace(text, @"(?<!\w)-?\d+(?!\w)", m => " " + ConvertFullNumber(m.Value) + " ");
        }

        #region Regex Expansions

        private static string ExpandAbbreviations(string text)
        {
            foreach (var kvp in AbbreviationMap)
            {
                text = Regex.Replace(text, $@"\b{Regex.Escape(kvp.Key)}(?=[\s,.]|\b)", kvp.Value, RegexOptions.IgnoreCase);
            }
            return text;
        }

        private static string ExpandCurrencySuffix(string text)
        {
            // ([£$€])([\d,.]*\d)\s*(million|billion|thousand)\b
            return Regex.Replace(text, @"([£$€])([\d,.]*\d)\s*(million|billion|thousand)\b", m =>
            {
                string symbol = m.Groups[1].Value;
                string amount = m.Groups[2].Value.Replace(",", "");
                string suffix = m.Groups[3].Value;
                string major = symbol == "$" ? "dollars" : (symbol == "£" ? "pounds" : "euros");
                return $"{ConvertNumber(amount)} {suffix} {major}";
            }, RegexOptions.IgnoreCase);
        }

        private static string ExpandPhoneNumbers(string text)
        {
            // (\+?\d{1,3}-)?\b(\d{3})-(?:(\d{3})-)?(\d{4})\b
            return Regex.Replace(text, @"(\+?\d{1,3}-)?\b(\d{3})-(?:(\d{3})-)?(\d{4})\b", m =>
            {
                List<string> parts = new List<string>();
                if (m.Groups[1].Success)
                {
                    string country = m.Groups[1].Value.Trim('-');
                    if (country.StartsWith("+")) parts.Add("plus");
                    foreach (char c in country.TrimStart('+')) if (char.IsDigit(c)) parts.Add(ConvertNumber(c.ToString()));
                }
                foreach (char c in m.Groups[2].Value) parts.Add(ConvertNumber(c.ToString()));
                if (m.Groups[3].Success) foreach (char c in m.Groups[3].Value) parts.Add(ConvertNumber(c.ToString()));
                foreach (char c in m.Groups[4].Value) parts.Add(ConvertNumber(c.ToString()));
                return string.Join(" ", parts);
            });
        }

        private static string ExpandDimensions(string text)
        {
            // \b(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?:\s*x\s*(\d+(?:\.\d+)?))?\b
            return Regex.Replace(text, @"\b(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)(?:\s*x\s*(\d+(?:\.\d+)?))?\b", m =>
            {
                string p1 = ConvertNumber(m.Groups[1].Value);
                string p2 = ConvertNumber(m.Groups[2].Value);
                if (m.Groups[3].Success) return $"{p1} by {p2} by {ConvertNumber(m.Groups[3].Value)}";
                return $"{p1} by {p2}";
            }, RegexOptions.IgnoreCase);
        }

        private static string ExpandRomanNumerals(string text)
        {
            // \b(XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II)\b
            return Regex.Replace(text, @"\b(XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|IX|VIII|VII|VI|V|IV|III|II)\b", m =>
            {
                if (RomanMap.TryGetValue(m.Value, out var val)) return val;
                return m.Value;
            }, RegexOptions.IgnoreCase);
        }

        private static string ExpandDecades(string text)
        {
            // \b((?:1[89]|20)\d0)s\b
            return Regex.Replace(text, @"\b((?:1[89]|20)\d0)s\b", m =>
            {
                string yearWords = ConvertYear(m.Groups[1].Value);
                if (yearWords.EndsWith("ty")) return yearWords.Substring(0, yearWords.Length - 1) + "ies";
                return yearWords + "s";
            });
        }

        private static string ExpandScores(string text)
        {
            // \b(\d{1,2})-(\d{1,2})\b
            return Regex.Replace(text, @"\b(\d{1,2})-(\d{1,2})\b", m =>
            {
                return $"{ConvertNumber(m.Groups[1].Value)} to {ConvertNumber(m.Groups[2].Value)}";
            });
        }

        private static string ExpandDates(string text)
        {
            // \b(0?[1-9]|1[0-2])/([0-2]?\d|3[01])/(\d{2,4})\b
            return Regex.Replace(text, @"\b(0?[1-9]|1[0-2])/([0-2]?\d|3[01])/(\d{2,4})\b", m =>
            {
                int month = int.Parse(m.Groups[1].Value);
                string day = m.Groups[2].Value;
                string year = m.Groups[3].Value;
                string monthName = CultureInfo.InvariantCulture.DateTimeFormat.GetMonthName(month);
                string dayOrdinal = ConvertOrdinal(day);
                int yearNum = int.Parse(year);
                if (year.Length == 2) yearNum += yearNum < 50 ? 2000 : 1900;
                return $"{monthName} {dayOrdinal}, {ConvertYear(yearNum.ToString())}";
            });
        }

        private static string ExpandTimes(string text)
        {
            // \b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(\s*(?:a\.?m\.?|p\.?m\.?))?\b
            return Regex.Replace(text, @"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(\s*(?:a\.?m\.?|p\.?m\.?))?\b", m =>
            {
                int h = int.Parse(m.Groups[1].Value);
                int min = int.Parse(m.Groups[2].Value);
                string ampm = m.Groups[4].Value.ToLower();

                string hWord = ConvertNumber((h > 12 && !string.IsNullOrEmpty(ampm)) ? (h - 12).ToString() : h.ToString());
                if (h == 0 && !string.IsNullOrEmpty(ampm)) hWord = "twelve";

                string mWord = "";
                if (min > 0) mWord = (min < 10 ? " oh " : " ") + ConvertNumber(min.ToString());

                string result = hWord + mWord;
                if (m.Groups[3].Success) result += " and " + ConvertNumber(m.Groups[3].Value) + " seconds";
                if (!string.IsNullOrEmpty(ampm)) result += (ampm.Contains("p") ? " pm" : " am");
                return result;
            }, RegexOptions.IgnoreCase);
        }

        private static string ExpandCommaNumbers(string text)
        {
            return Regex.Replace(text, @"(\d[\d,]+\d)", m => m.Value.Replace(",", ""));
        }

        private static string ExpandCurrencies(string text)
        {
            // ([£$€])(\d*\.?\d+)|(\d*\.?\d+)\s*([£$€])
            return Regex.Replace(text, @"([£$€])(\d*\.?\d+)|(\d*\.?\d+)\s*([£$€])", m =>
            {
                string symbol = m.Groups[1].Success ? m.Groups[1].Value : m.Groups[4].Value;
                string amountStr = m.Groups[2].Success ? m.Groups[2].Value : m.Groups[3].Value;
                amountStr = amountStr.Replace(",", "");
                if (amountStr.StartsWith(".")) amountStr = "0" + amountStr;

                var majorMap = new Dictionary<string, (string, string)> { { "$", ("dollar", "dollars") }, { "£", ("pound", "pounds") }, { "€", ("euro", "euros") } };
                var minorMap = new Dictionary<string, (string, string)> { { "$", ("cent", "cents") }, { "£", ("penny", "pence") }, { "€", ("cent", "cents") } };

                var (majorS, majorP) = majorMap.GetValueOrDefault(symbol, ("", ""));
                string[] parts = amountStr.Split('.');
                long majorVal = long.Parse(parts[0]);
                long minorVal = parts.Length > 1 ? long.Parse(parts[1].PadRight(2, '0').Substring(0, 2)) : 0;

                List<string> result = new List<string>();
                if (majorVal > 0) result.Add($"{ConvertNumber(majorVal.ToString())} {(majorVal == 1 ? majorS : majorP)}");
                if (minorVal > 0)
                {
                    var (minorS, minorP) = minorMap.GetValueOrDefault(symbol, ("", ""));
                    result.Add($"{ConvertNumber(minorVal.ToString())} {(minorVal == 1 ? minorS : minorP)}");
                }
                return string.Join(" and ", result) == "" ? $"zero {majorP}" : string.Join(" and ", result);
            });
        }

        private static string ExpandMeasurements(string text)
        {
            // (?<!\w)(-?(?:\d+/\d+|\d+(?:\.\d+)?))\s*({units})\b
            string pattern = $@"(?<!\w)(-?(?:\d+/\d+|\d+(?:\.\d+)?))\s*({MeasurementUnitsRegexPart})\b";
            return Regex.Replace(text, pattern, m =>
            {
                string numStr = m.Groups[1].Value;
                string unit = m.Groups[2].Value;
                bool isNeg = numStr.StartsWith("-");
                if (isNeg) numStr = numStr.Substring(1);

                string numWord;
                bool isPlural;
                if (numStr.Contains("/"))
                {
                    numWord = ExpandSingleFraction(numStr);
                    isPlural = true;
                }
                else
                {
                    numWord = ConvertNumber(numStr);
                    isPlural = double.Parse(numStr) != 1.0;
                }

                var (singular, plural) = MeasurementMap[unit];
                string result = $"{numWord} {(isPlural ? plural : singular)}";
                return isNeg ? "minus " + result : result;
            });
        }

        private static string ExpandFractions(string text)
        {
            return Regex.Replace(text, @"\b(\d+)/(\d+)\b", m => ExpandSingleFraction(m.Value));
        }

        private static string ExpandSingleFraction(string frac)
        {
            var parts = frac.Split('/');
            int n = int.Parse(parts[0]);
            int d = int.Parse(parts[1]);
            if (d == 0) return frac;
            if (n == 1 && d == 2) return "one half";
            if (n == 1 && d == 4) return "one quarter";
            if (n == 3 && d == 4) return "three quarters";
            return $"{ConvertNumber(n.ToString())} over {ConvertNumber(d.ToString())}";
        }

        private static string ExpandDecimals(string text)
        {
            return Regex.Replace(text, @"(\d+\.\d+)", m =>
            {
                var parts = m.Value.Split('.');
                string intPart = ConvertNumber(parts[0]);
                StringBuilder decPart = new StringBuilder();
                foreach (char c in parts[1]) decPart.Append(" " + (c == '0' ? "zero" : Ones[c - '0']));
                return $"{intPart} point{decPart}";
            });
        }

        private static string ExpandAlphanumeric(string text)
        {
            // \b([a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+)\b
            return Regex.Replace(text, @"\b([a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+)\b", m =>
            {
                var parts = Regex.Matches(m.Value, @"[a-zA-Z]+|[0-9]+");
                List<string> expanded = new List<string>();
                foreach (Match part in parts)
                {
                    expanded.Add(part.Value);
                }
                return string.Join(" ", expanded);
            });
        }

        private static string ExpandAcronyms(string text)
        {
            return Regex.Replace(text, @"\b[A-Z]{2,}\b", m => string.Join(" ", m.Value.ToCharArray()));
        }

        #endregion

        #region Core Conversions

        private static string ConvertNumber(string numStr)
        {
            if (!long.TryParse(numStr, out long num)) return numStr;
            if (num == 0) return "zero";
            if (num < 0) return "negative " + ConvertNumber((-num).ToString());

            string result = "";
            int groupIndex = 0;
            while (num > 0)
            {
                int group = (int)(num % 1000);
                if (group > 0)
                {
                    string groupStr = ConvertThreeDigits(group);
                    result = groupStr + (Thousands[groupIndex] == "" ? "" : " " + Thousands[groupIndex]) + (result == "" ? "" : " " + result);
                }
                num /= 1000;
                groupIndex++;
            }
            return result.Trim();
        }

        private static string ConvertThreeDigits(int n)
        {
            string res = "";
            if (n >= 100)
            {
                res += Ones[n / 100] + " hundred";
                n %= 100;
                if (n > 0) res += " ";
            }
            if (n >= 20)
            {
                res += Tens[n / 10];
                if (n % 10 > 0) res += " " + Ones[n % 10];
            }
            else if (n > 0)
            {
                res += Ones[n];
            }
            return res.Trim();
        }

        private static string ConvertFullNumber(string numStr)
        {
            if (numStr.StartsWith("-")) return "minus " + ConvertNumberPositive(numStr.Substring(1));
            return ConvertNumberPositive(numStr);
        }

        private static string ConvertNumberPositive(string numStr)
        {
            if (!long.TryParse(numStr, out long num)) return numStr;
            // 处理年份等特殊规律
            if (num >= 2000 && num < 2010) return $"two thousand and {ConvertNumber((num % 100).ToString())}";
            if (num >= 1100 && num < 2100 && num % 100 != 0)
                return $"{ConvertNumber((num / 100).ToString())} {ConvertNumber((num % 100).ToString())}";
            return ConvertNumber(numStr);
        }

        private static string ConvertYear(string yearStr) => ConvertNumberPositive(yearStr);

        private static string ConvertOrdinal(string numStr)
        {
            if (!int.TryParse(numStr, out int num)) return numStr;
            string numWord = ConvertNumber(numStr);
            if (num % 100 >= 11 && num % 100 <= 13) return numWord + "th";
            switch (num % 10)
            {
                case 1: return numWord.EndsWith("one") ? numWord.Substring(0, numWord.Length - 3) + "first" : numWord + "st";
                case 2: return numWord.EndsWith("two") ? numWord.Substring(0, numWord.Length - 3) + "second" : numWord + "nd";
                case 3: return numWord.EndsWith("three") ? numWord.Substring(0, numWord.Length - 5) + "third" : numWord + "rd";
                default:
                    if (numWord.EndsWith("y")) return numWord.Substring(0, numWord.Length - 1) + "ieth";
                    if (numWord.EndsWith("eight")) return numWord + "h";
                    if (numWord.EndsWith("five")) return numWord.Substring(0, numWord.Length - 2) + "fth";
                    if (numWord.EndsWith("nine")) return numWord.Substring(0, numWord.Length - 1) + "th";
                    if (numWord.EndsWith("twelve")) return numWord.Substring(0, numWord.Length - 2) + "fth";
                    return numWord + "th";
            }
        }

        #endregion
    }
}

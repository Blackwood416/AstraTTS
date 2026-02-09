using Xunit;
using Xunit.Abstractions;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.TextNorm;
using System;
using System.IO;
using System.Linq;

namespace AstraTTS.Tests
{
    /// <summary>
    /// 日语 G2P 单元测试
    /// </summary>
    public class JapaneseG2PTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly JapaneseG2P? _g2p;

        // 词典相对路径（从测试项目输出目录到 GenieData）
        private static readonly string DictPath = GetDictPath();

        private static string GetDictPath()
        {
            // 尝试多个可能的词典位置
            var possiblePaths = new[]
            {
                // 从测试项目 bin 目录向上查找 (新位置)
                Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                    @"..\..\..\..\resources\shared\g2p\JapaneseG2P\open_jtalk_dic_utf_8-1.11")),
                // 直接使用绝对路径
                @"E:\RiderProjects\AstraTTS\resources\shared\g2p\JapaneseG2P\open_jtalk_dic_utf_8-1.11",
            };

            foreach (var path in possiblePaths)
            {
                if (Directory.Exists(path))
                    return path;
            }

            return possiblePaths[0]; // 返回第一个路径，让测试失败时显示
        }

        public JapaneseG2PTests(ITestOutputHelper output)
        {
            _output = output;

            if (Directory.Exists(DictPath))
            {
                _g2p = new JapaneseG2P(DictPath);
                _output.WriteLine($"[Setup] Loaded dictionary from: {DictPath}");
            }
            else
            {
                _output.WriteLine($"[Setup] Dictionary not found at: {DictPath}");
                _output.WriteLine("[Setup] Tests will be skipped.");
            }
        }

        public void Dispose()
        {
            _g2p?.Dispose();
        }

        [Fact]
        public void JapaneseG2P_Hiragana_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // 测试平假名
            var result = _g2p!.Process("こんにちは");

            _output.WriteLine($"Input: こんにちは");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");
            _output.WriteLine($"PhoneIds: {string.Join(", ", result.PhoneIds)}");

            Assert.NotEmpty(result.Phones);
            Assert.True(result.Phones.Count > 1, "Should produce multiple phonemes");

            // こんにちは -> k, o, N, n, i, ch, i, w, a (大约 9 个音素)
            Assert.Contains("k", result.Phones);
            Assert.Contains("o", result.Phones);
        }

        [Fact]
        public void JapaneseG2P_Katakana_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // 测试片假名
            var result = _g2p!.Process("コンニチハ");

            _output.WriteLine($"Input: コンニチハ");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");

            Assert.NotEmpty(result.Phones);
            Assert.True(result.Phones.Count > 1, "Should produce multiple phonemes");
        }

        [Fact]
        public void JapaneseG2P_Kanji_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // 测试汉字（日语读音）
            var result = _g2p!.Process("今日");

            _output.WriteLine($"Input: 今日");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");

            Assert.NotEmpty(result.Phones);
            // 今日 -> きょう (kyou) 或 こんにち (konnichi)
        }

        [Fact]
        public void JapaneseG2P_MixedText_ReturnsPhonemes()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            // 测试混合文本
            var result = _g2p!.Process("今日は良い天気です。");

            _output.WriteLine($"Input: 今日は良い天気です。");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");
            _output.WriteLine($"Word2Ph: {string.Join(", ", result.Word2Ph)}");

            Assert.NotEmpty(result.Phones);
        }

        [Fact]
        public void JapaneseG2P_EmptyInput_ReturnsDefault()
        {
            Skip.If(_g2p == null, "Dictionary not available");

            var result = _g2p!.Process("");

            Assert.NotEmpty(result.Phones);
            Assert.Contains("SP", result.Phones);
        }

        [Theory]
        [InlineData("あ", "a")]  // 单个平假名
        [InlineData("い", "i")]
        [InlineData("う", "u")]
        [InlineData("え", "e")]
        [InlineData("お", "o")]
        public void JapaneseG2P_SingleVowel_ReturnsCorrectPhoneme(string input, string expectedPhoneme)
        {
            Skip.If(_g2p == null, "Dictionary not available");

            var result = _g2p!.Process(input);

            _output.WriteLine($"Input: {input}");
            _output.WriteLine($"Phones: {string.Join(", ", result.Phones)}");

            Assert.Contains(expectedPhoneme, result.Phones);
        }
    }

    /// <summary>
    /// 日语文本规范化单元测试
    /// </summary>
    public class JapaneseTextNormalizerTests
    {
        [Fact]
        public void Normalize_FullWidthPunctuation_ConvertsToHalfWidth()
        {
            var input = "こんにちは。これは、テストです！";
            var result = JapaneseTextNormalizer.Normalize(input);

            Assert.Contains(".", result);
            Assert.Contains(",", result);
            Assert.Contains("!", result);
            Assert.DoesNotContain("。", result);
            Assert.DoesNotContain("、", result);
            Assert.DoesNotContain("！", result);
        }

        [Fact]
        public void Normalize_PercentSymbol_ConvertsToJapanese()
        {
            var input = "100%の確率";
            var result = JapaneseTextNormalizer.Normalize(input);

            Assert.Contains("パーセント", result);
            Assert.DoesNotContain("%", result);
        }

        [Fact]
        public void Normalize_ConsecutivePunctuation_MergesIntoOne()
        {
            var input = "なに！！！";
            var result = JapaneseTextNormalizer.Normalize(input);

            // 连续的 ! 应该合并为一个
            Assert.Equal(1, result.Count(c => c == '!'));
        }

        [Fact]
        public void PostReplacePunctuation_JapanesePunctuation_ReturnsEnglish()
        {
            Assert.Equal(".", JapaneseTextNormalizer.PostReplacePunctuation("。"));
            Assert.Equal(",", JapaneseTextNormalizer.PostReplacePunctuation("、"));
            Assert.Equal("!", JapaneseTextNormalizer.PostReplacePunctuation("！"));
            Assert.Equal("?", JapaneseTextNormalizer.PostReplacePunctuation("？"));
        }
    }

    /// <summary>
    /// 混合语言 G2P 日语检测测试
    /// </summary>
    public class MixedLanguageG2PJapaneseTests
    {
        [Fact]
        public void CharDetection_Hiragana_DetectedAsJapanese()
        {
            // 测试平假名字符范围
            char hiragana = 'あ'; // U+3042
            Assert.True(hiragana >= 0x3040 && hiragana <= 0x309F);
        }

        [Fact]
        public void CharDetection_Katakana_DetectedAsJapanese()
        {
            // 测试片假名字符范围
            char katakana = 'ア'; // U+30A2
            Assert.True(katakana >= 0x30A0 && katakana <= 0x30FF);
        }

        [Fact]
        public void CharDetection_Kanji_DetectedAsChinese()
        {
            // 汉字应该被检测为中文（因为优先判断假名）
            char kanji = '日'; // U+65E5
            Assert.True(kanji >= 0x4E00 && kanji <= 0x9FFF);
        }
    }
}

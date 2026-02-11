using System;
using System.IO;
using System.Linq;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.G2P.Common;
using AstraTTS.Core.Frontend.G2P.Chinese;
using AstraTTS.Core.Frontend.G2P.English;
using AstraTTS.Core.Frontend.G2P.Japanese;
using Xunit;

namespace AstraTTS.Tests
{
    public class MixedLanguageG2PTests
    {
        private readonly ChineseG2P _chinese;
        private readonly EnglishG2P _english;
        private readonly MixedLanguageG2P _mixed;

        private static readonly string BaseDir = AppDomain.CurrentDomain.BaseDirectory;
        private static readonly string ProjectDir = Path.GetFullPath(Path.Combine(BaseDir, "..", "..", "..", ".."));
        private static readonly string DictDir = Path.Combine(ProjectDir, "resources", "shared", "g2p", "dicts", "chinese");
        private static readonly string EngDictDir = Path.Combine(ProjectDir, "AstraTTS.Core");
        private static readonly string EngModelDir = Path.Combine(ProjectDir, "AstraTTS.Core", "models");

        public MixedLanguageG2PTests()
        {
            string vocab = Path.Combine(DictDir, "opencpop-strict.txt");
            string dict = Path.Combine(DictDir, "mandarin_pinyin.dict");
            string poly = Path.Combine(DictDir, "polyphonic.json");
            string jieba = Path.Combine(DictDir, "jieba");
            _chinese = new ChineseG2P(vocab, dict, "", poly, jieba);

            string cmu = Path.Combine(ProjectDir, "resources", "shared", "g2p", "dicts", "english", "cmudict.dict");
            string g2pBase = Path.Combine(ProjectDir, "resources", "shared", "g2p", "models", "english");
            string g2pModel = Path.Combine(g2pBase, "checkpoint20.npz");

            string posDir = Path.Combine(g2pBase, "taggers", "averaged_perceptron_tagger_eng");
            string segDir = Path.Combine(g2pBase, "wordsegment");

            _english = new EnglishG2P(cmu, g2pModel, "", posDir, segDir);

            _mixed = new MixedLanguageG2P(_chinese, _english);

            // Verify components are loaded
            Assert.True(_english.HasPosTagger, "EnglishPosTagger failed to load.");
            Assert.True(_english.HasWordSegmenter, "EnglishWordSegmenter failed to load.");
        }

        [Fact]
        public void TestNumericRoutingInheritance()
        {
            // Case 1: Numbers following Chinese
            var res1 = _mixed.Process("你好123");
            Assert.Contains(res1.Segments ?? new System.Collections.Generic.List<LanguageSegment>(),
                s => s.Text.Contains("一百二十三") && s.Language == PhoneLanguage.Chinese);

            // Case 2: Numbers following English
            var res2 = _mixed.Process("Apple 123");
            Assert.Contains(res2.Segments ?? new System.Collections.Generic.List<LanguageSegment>(),
                s => s.Text.ToLower().Contains("one hundred twenty three") && s.Language == PhoneLanguage.English);
        }

        [Fact]
        public void TestLanguageTag()
        {
            var res = _mixed.Process("[lang=en]123[/lang]");
            Assert.Contains(res.Segments ?? new System.Collections.Generic.List<LanguageSegment>(),
                s => s.Text.ToLower().Contains("one hundred twenty three") && s.Language == PhoneLanguage.English);
        }

        [Fact]
        public void TestComplexEnglish()
        {
            // Test Currency
            var res1 = _mixed.Process("$5.20");
            string s1 = string.Join(" | ", res1.Segments.Select(s => $"{s.Text} ({s.Language})"));
            Assert.True(res1.Segments.Any(s => s.Text.ToLower().Contains("five dollars and twenty cents")),
                $"Expected 'five dollars and twenty cents', but got: {s1}");

            // Test Date
            var res2 = _mixed.Process("10/1/2023");
            string s2 = string.Join(" | ", res2.Segments.Select(s => $"{s.Text} ({s.Language})"));
            Assert.True(res2.Segments.Any(s => s.Text.ToLower().Contains("october first, twenty twenty three")),
                $"Expected 'october first, twenty twenty three', but got: {s2}");

            // Test Possessive
            var res3 = _mixed.Process("Apple's price");
            // Apple -> AE1 P AH0 L -> ends with L -> + Z
            Assert.Contains("z", res3.Phones.Select(p => p.ToLower()));

            // Test Mixed Currency
            var res4 = _mixed.Process("价格为$5.20");
            string s4 = string.Join(" | ", res4.Segments.Select(s => $"{s.Text} ({s.Language})"));
            Assert.True(res4.Segments.Any(s => s.Language == PhoneLanguage.Chinese && s.Text.Contains("价格为")),
                $"Expected Chinese '价格为', but got: {s4}");
            Assert.True(res4.Segments.Any(s => s.Language == PhoneLanguage.English && s.Text.ToLower().Contains("five dollars and twenty cents")),
                $"Expected English 'five dollars and twenty cents', but got: {s4}");
        }

        [Fact]
        public void TestHomographs()
        {
            // Test "read"
            // "I will read" -> read (VB) -> /riːd/ (IY1)
            var res1 = _mixed.Process("I will read it");
            Assert.Contains("iy1", res1.Phones.Select(p => p.ToLower()));

            // "I read it yesterday" -> read (VBD) -> /rɛd/ (EH1)
            var res2 = _mixed.Process("I read it yesterday");
            string phones2 = string.Join(" ", res2.Phones);
            Assert.True(res2.Phones.Any(p => p.Equals("eh1", StringComparison.OrdinalIgnoreCase)),
                $"Expected 'eh1' for 'read' (VBD) in 'I read it yesterday', but got: {phones2}");
            Assert.False(res2.Phones.Any(p => p.Equals("iy1", StringComparison.OrdinalIgnoreCase)),
                $"Expected NOT to have 'iy1' for 'read' (VBD) in 'I read it yesterday', but got: {phones2}");

            // "I have read this" -> read (VBN) -> /rɛd/ (EH1)
            var res2b = _mixed.Process("I have read this yesterday");
            string phones2b = string.Join(" ", res2b.Phones);
            Assert.True(res2b.Phones.Any(p => p.Equals("eh1", StringComparison.OrdinalIgnoreCase)),
                $"Expected 'eh1' for 'read' (VBN) in 'I have read this yesterday', but got: {phones2b}");
            Assert.False(res2b.Phones.Any(p => p.Equals("iy1", StringComparison.OrdinalIgnoreCase)),
                $"Expected NOT to have 'iy1' for 'read' (VBN) in 'I have read this yesterday', but got: {phones2b}");

            // Test "lead"
            // "lead pipe" -> lead (NN) -> /lɛd/ (EH1)
            var res3 = _mixed.Process("lead pipe");
            Assert.Contains("eh1", res3.Phones.Select(p => p.ToLower()));
            Assert.DoesNotContain("iy1", res3.Phones.Select(p => p.ToLower()));

            // "they lead the way" -> lead (VB/VBP) -> /liːd/ (IY1)
            var res4 = _mixed.Process("they lead the way");
            Assert.Contains("iy1", res4.Phones.Select(p => p.ToLower()));

            // Test "present"
            // "they present the award" -> present (VB) -> /prɪˈzɛnt/ (IY0 Z EH1 N T)
            var res5 = _mixed.Process("they present the award");
            Assert.Contains("z", res5.Phones.Select(p => p.ToLower()));

            // "a birthday present" -> present (NN) -> /ˈprɛzənt/ (EH1 Z AH0 N T)
            var res6 = _mixed.Process("a birthday present");
            // actually "present" as NN has [P R EH1 Z AH0 N T] as well in some dicts, 
            // but the main difference is stress and phonemes in some cases.
            // My implementation: VB -> IY0 Z EH1 N T, NN -> EH1 Z AH0 N T
            Assert.Contains("iy0", res5.Phones.Select(p => p.ToLower()));
            Assert.DoesNotContain("iy0", res6.Phones.Select(p => p.ToLower()));

            // Test "live"
            // "They live here" -> live (VBP) -> /lɪv/ (IH1)
            var res7 = _mixed.Process("They live here");
            Assert.Contains("ih1", res7.Phones.Select(p => p.ToLower()));
            Assert.DoesNotContain("ay1", res7.Phones.Select(p => p.ToLower()));

            // "a live show" -> live (JJ) -> /laɪv/ (AY1)
            var res8 = _mixed.Process("a live show");
            Assert.Contains("ay1", res8.Phones.Select(p => p.ToLower()));
            Assert.DoesNotContain("ih1", res8.Phones.Select(p => p.ToLower()));

            // Test "record"
            // "record a song" -> record (VB) -> /rɪˈkɔːrd/ (IH0)
            var res9 = _mixed.Process("record a song");
            Assert.Contains("ih0", res9.Phones.Select(p => p.ToLower()));
            Assert.DoesNotContain("eh1", res9.Phones.Select(p => p.ToLower()));

            // "a new record" -> record (NN) -> /ˈrɛkərd/ (EH1)
            var res10 = _mixed.Process("a new record");
            Assert.Contains("eh1", res10.Phones.Select(p => p.ToLower()));
            Assert.DoesNotContain("ih0", res10.Phones.Select(p => p.ToLower()));

            // Test "object"
            // "I object" -> object (VBP) -> /əbˈdʒɛkt/ (AH0)
            var res11 = _mixed.Process("I object");
            Assert.Contains("ah0", res11.Phones.Select(p => p.ToLower()));

            // "a flying object" -> object (NN) -> /ˈɒbdʒɪkt/ (AA1)
            var res12 = _mixed.Process("a flying object");
            Assert.Contains("aa1", res12.Phones.Select(p => p.ToLower()));

            // Test "desert"
            // "do not desert me" -> desert (VB) -> /dɪˈzɜːrt/ (IH0 Z ER1 T)
            var res13 = _mixed.Process("do not desert me");
            Assert.Contains("er1", res13.Phones.Select(p => p.ToLower()));

            // "a hot desert" -> desert (NN) -> /ˈdɛzərt/ (EH1 Z ER0 T)
            var res14 = _mixed.Process("a hot desert");
            Assert.Contains("eh1", res14.Phones.Select(p => p.ToLower()));
        }

        [Fact]
        public void TestCompoundWords()
        {
            // "overload" -> over + load
            var res1 = _mixed.Process("overload");
            // "over" -> OW1 V ER0, "load" -> L OW1 D
            Assert.Contains("v", res1.Phones.Select(p => p.ToLower()));
            Assert.Contains("l", res1.Phones.Select(p => p.ToLower()));

            // "waterfall" -> water + fall
            var res2 = _mixed.Process("waterfall");
            Assert.Contains("w", res2.Phones.Select(p => p.ToLower()));
            Assert.Contains("f", res2.Phones.Select(p => p.ToLower()));
        }

        [Fact]
        public void TestNativeModeTag()
        {
            var res = _mixed.Process("你好 {ni3}{hao3}");
            Assert.Contains("n", res.Phones);
            Assert.Contains("i3", res.Phones);
        }
    }
}

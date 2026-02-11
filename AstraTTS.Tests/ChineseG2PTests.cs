using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.G2P.Common;
using AstraTTS.Core.Frontend.G2P.Chinese;
using Xunit;

namespace AstraTTS.Tests
{
    public class ChineseG2PTests
    {
        private readonly ChineseG2P _g2p;
        private static readonly string BaseDir = AppDomain.CurrentDomain.BaseDirectory;
        private static readonly string ProjectDir = Path.GetFullPath(Path.Combine(BaseDir, "..", "..", "..", ".."));
        private static readonly string DictDir = Path.Combine(ProjectDir, "resources", "shared", "g2p", "dicts", "chinese");

        public ChineseG2PTests()
        {
            string vocab = Path.Combine(DictDir, "opencpop-strict.txt");
            string dict = Path.Combine(DictDir, "mandarin_pinyin.dict");
            string poly = Path.Combine(DictDir, "polyphonic.json");
            string jieba = Path.Combine(DictDir, "jieba");
            _g2p = new ChineseG2P(vocab, dict, "", poly, jieba);
        }

        [Fact]
        public void TestBasicG2P()
        {
            var res = _g2p.Process("你好");
            Assert.NotEmpty(res.Phones);
            // "你好" 触发三声变调: ni3 -> ni2
            Assert.Contains("n", res.Phones);
            Assert.Contains("i2", res.Phones);
            Assert.Contains("h", res.Phones);
            Assert.Contains("ao3", res.Phones);
        }

        [Fact]
        public void TestToneSandhiIntegrate()
        {
            // "一二三": "二" 是四声，所以 "一" 变二声 (yi2)
            var res = _g2p.Process("一二三");
            Assert.Equal("i2", res.Phones[1]); // Opencpop 'yi' -> 'i'
        }

        [Fact]
        public void TestPolyphonicCorrect()
        {
            // "湖泊" -> "hu2 po1" (In dictionary)
            var res = _g2p.Process("湖泊");
            Assert.Equal("hu2", GetPinyin(res, 0));
            Assert.Equal("po1", GetPinyin(res, 1));
        }

        [Fact]
        public void TestErhuaIntegrate()
        {
            // "小院儿" -> "xiao3 yuan4 er5" -> merged: "xiao3 yanr4" or something
            // In ErhuaProcessor: "er2"/"er5" merged with previous final.
            var res = _g2p.Process("小院儿");
            // yuan4 + er5 -> yanr4 (usually)
            Assert.Contains(res.Phones, p => p.Contains("er"));
        }

        [Fact]
        public void TestNativeMode()
        {
            // Input: {ni3}{hao3}
            var res = _g2p.Process("{ni3}{hao3}");
            Assert.Equal(4, res.Phones.Count);
            Assert.Equal("n", res.Phones[0]);
            Assert.Equal("i3", res.Phones[1]);
            Assert.Equal("h", res.Phones[2]);
            Assert.Equal("ao3", res.Phones[3]);
        }

        private string GetPinyin(G2PResult res, int charIdx)
        {
            // Simple helper to reconstruct pinyin from phones based on Word2Ph
            int start = 0;
            for (int i = 0; i < charIdx; i++) start += res.Word2Ph[i];
            int count = res.Word2Ph[charIdx];
            var slice = res.Phones.GetRange(start, count);
            return string.Join("", slice);
        }
    }
}

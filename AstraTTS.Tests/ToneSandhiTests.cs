using System;
using System.Collections.Generic;
using AstraTTS.Core.Frontend.G2P;
using AstraTTS.Core.Frontend.G2P.Common;
using AstraTTS.Core.Frontend.G2P.Chinese;
using Xunit;

namespace AstraTTS.Tests
{
    public class ToneSandhiTests
    {
        private readonly ToneSandhi _sandhi = new ToneSandhi();

        [Theory]
        [InlineData("不怕", "n", "bu4 pa4", "bu2 pa4")]
        [InlineData("看不懂", "v", "kan4 bu4 dong3", "kan4 bu5 dong3")]
        [InlineData("一段", "m", "yi1 duan4", "yi2 duan4")]
        [InlineData("一天", "m", "yi1 tian1", "yi4 tian1")]
        [InlineData("看一看", "v", "kan4 yi1 kan4", "kan4 yi5 kan4")]
        [InlineData("你好", "l", "ni3 hao3", "ni2 hao3")]
        [InlineData("妈妈", "n", "ma1 ma1", "ma1 ma5")]
        [InlineData("什么", "r", "shen2 me5", "shen2 me5")]
        [InlineData("孩子", "n", "hai2 zi3", "hai2 zi5")]
        public void TestModifyTones(string word, string pos, string inputFinals, string expectedFinals)
        {
            var inputList = new List<string>(inputFinals.Split(' '));
            var expectedList = new List<string>(expectedFinals.Split(' '));

            var result = _sandhi.ModifyTones(word, pos, inputList);

            Assert.Equal(expectedList, result);
        }

        [Fact]
        public void TestThreeToneSandhi_ThreeWord_21()
        {
            // 蒙古包 (2+1 structure) -> 2 2 3
            var input = new List<string> { "meng3", "gu3", "bao3" };
            var expected = new List<string> { "meng2", "gu2", "bao3" };
            var result = _sandhi.ModifyTones("蒙古包", "n", input);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void TestPreMerge()
        {
            var segments = new List<(string word, string pos)>
            {
                ("不", "d"), ("怕", "v"),
                ("看", "v"), ("一", "m"), ("看", "v"),
                ("好", "a"), ("好", "a"),
                ("小", "a"), ("院", "n"), ("儿", "n")
            };

            var merged = _sandhi.PreMergeForModify(segments);

            // "不"+"怕" -> "不怕"
            // "看"+"一"+"看" -> "看一看"
            // "好"+"好" -> "好好"
            // "小"+"院"+"儿" -> "小院儿" (MergeEr: "院" + "儿" -> "院儿")

            Assert.Contains(merged, s => s.word == "不怕");
            Assert.Contains(merged, s => s.word == "看一看");
            Assert.Contains(merged, s => s.word == "好好");
            Assert.Contains(merged, s => s.word == "院儿");
        }
    }
}

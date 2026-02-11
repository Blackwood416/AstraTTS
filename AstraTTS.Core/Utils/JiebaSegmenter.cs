using System.Collections.Generic;
using System.Linq;
using JiebaNet.Segmenter;
using JiebaNet.Segmenter.PosSeg;

namespace AstraTTS.Core.Utils
{
    /// <summary>
    /// 基于 jieba.NET 的分词器，支持词性标注（POS Billing）。
    /// </summary>
    public class JiebaSegmenter
    {
        private readonly PosSegmenter _posSeg;

        public JiebaSegmenter(string dictDir)
        {
            if (!string.IsNullOrEmpty(dictDir))
            {
                ConfigManager.ConfigFileBaseDir = dictDir;
            }
            _posSeg = new PosSegmenter();
        }

        /// <summary>
        /// 对文本进行分词并返回词语及其词性。
        /// </summary>
        public List<(string word, string pos)> Cut(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<(string word, string pos)>();

            var result = _posSeg.Cut(text);
            return result.Select(p => (p.Word, p.Flag)).ToList();
        }

        /// <summary>
        /// 甚至可以支持搜索模式的分词（用于 PreMerge 逻辑中的子词判断）。
        /// </summary>
        public List<string> CutForSearch(string text)
        {
            // Note: PosSegmenter doesn't have CutForSearch directly with POS.
            // But we can use the regular segmenter if needed.
            // For now, simple Cut is enough for G2P.
            return _posSeg.Cut(text).Select(p => p.Word).ToList();
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AstraTTS.Core.Utils
{
    public class SimpleMaxMatchSegmenter
    {
        private readonly HashSet<string> _vocab = new HashSet<string>();
        private int _maxLen = 0;

        public SimpleMaxMatchSegmenter(string? dictPath = null)
        {
            // 加载基础词汇 (针对演示)
            var defaults = new[] { "你好", "我是", "阿", "里", "亚", "测试", "语音", "合成", "模型" };
            foreach (var w in defaults) AddWord(w);

            if (!string.IsNullOrEmpty(dictPath) && File.Exists(dictPath))
            {
                foreach (var line in File.ReadAllLines(dictPath))
                {
                    var parts = line.Split(' ');
                    if (parts.Length > 0) AddWord(parts[0]);
                }
            }
        }

        public void AddWord(string word)
        {
            if (string.IsNullOrWhiteSpace(word)) return;
            _vocab.Add(word);
            if (word.Length > _maxLen) _maxLen = word.Length;
        }
        
        public void LoadVocabulary(IEnumerable<string> words)
        {
            foreach (var w in words) AddWord(w);
        }

        public IEnumerable<string> Cut(string text)
        {
            int ptr = 0;
            while (ptr < text.Length)
            {
                string bestWord = text[ptr].ToString();
                // 尝试匹配更长的词
                for (int len = Math.Min(_maxLen, text.Length - ptr); len > 1; len--)
                {
                    string sub = text.Substring(ptr, len);
                    if (_vocab.Contains(sub))
                    {
                        bestWord = sub;
                        break;
                    }
                }
                yield return bestWord;
                ptr += bestWord.Length;
            }
        }
    }
}

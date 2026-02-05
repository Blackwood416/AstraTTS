using System.Collections.Generic;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 音素对应的语言类型
    /// </summary>
    public enum PhoneLanguage : byte
    {
        Chinese,
        English,
        Other  // SP, 标点等
    }

    /// <summary>
    /// 语言片段信息，用于分段 BERT 处理
    /// </summary>
    public struct LanguageSegment
    {
        /// <summary>原始文本片段</summary>
        public string Text;
        /// <summary>语言类型</summary>
        public PhoneLanguage Language;
        /// <summary>在 Phones 列表中的起始索引</summary>
        public int StartPhoneIndex;
        /// <summary>音素数量</summary>
        public int PhoneCount;
        /// <summary>该片段的 Word2Ph 数组</summary>
        public int[] Word2Ph;
    }

    public struct G2PResult
    {
        public string NormalizedText;
        public List<string> Phones;
        public long[] PhoneIds;
        public int[] Word2Ph;
        
        /// <summary>
        /// 每个音素对应的语言标记，用于分段 BERT 处理。
        /// 长度与 Phones 相同。
        /// </summary>
        public PhoneLanguage[]? LanguageTags;
        
        /// <summary>
        /// 语言片段列表，包含每个片段的文本和位置信息。
        /// 用于分段 BERT 处理。
        /// </summary>
        public List<LanguageSegment>? Segments;
    }

    public interface IG2P
    {
        G2PResult Process(string text);
    }
}

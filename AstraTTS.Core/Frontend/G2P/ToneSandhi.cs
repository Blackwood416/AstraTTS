using System;
using System.Collections.Generic;
using System.Linq;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 中文变调处理器。
    /// 移植自 Genie-TTS 的 ToneSandhi.py。
    /// </summary>
    public class ToneSandhi
    {
        /// <summary>
        /// 执行变调修改。
        /// </summary>
        public List<string> ModifyTones(string word, string pos, List<string> finals)
        {
            // 1. 处理“一”和“不”的变调
            if (word == "一") return new List<string> { "yi4" }; // 默认
            if (word == "不") return new List<string> { "bu4" };

            // TODO: 完整的变调逻辑实现
            // 包括：
            // - 三声连读 (3+3 -> 2+3)
            // - “一”在不同声调前的变化
            // - “不”在四声前的变化
            
            return finals;
        }

        /// <summary>
        /// 针对“一”的特殊处理逻辑示例。
        /// </summary>
        private string HandleYi(string nextFinal)
        {
            if (string.IsNullOrEmpty(nextFinal)) return "yi1";
            char tone = nextFinal.Last();
            if (tone == '4') return "yi2";
            if (tone == '1' || tone == '2' || tone == '3') return "yi4";
            return "yi1";
        }
    }
}

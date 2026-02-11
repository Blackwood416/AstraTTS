using System;
using System.Collections.Generic;

namespace AstraTTS.Core.Frontend.G2P.Chinese
{
    /// <summary>
    /// 儿化音处理器。
    /// 移植自 Genie-TTS 的 Erhua.py。
    /// </summary>
    public class ErhuaProcessor
    {
        private static readonly HashSet<string> MustErhua = new HashSet<string>
        {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
        };

        private static readonly HashSet<string> NotErhua = new HashSet<string>
        {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿",
            "妻儿", "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿",
            "脑瘫儿", "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿",
            "侄儿", "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿",
            "猪儿", "猫儿", "狗儿", "少儿",
        };

        public (List<string> initials, List<string> finals) MergeErhua(
            List<string> initials,
            List<string> finals,
            string word,
            string pos)
        {
            // 1. 修正 er1 发音为 er2 (当'儿'在词尾且发音为er1时)
            if (finals.Count > 0 && word.Length == finals.Count && word[^1] == '儿' && finals[^1] == "er1")
            {
                finals[^1] = "er2";
            }

            // 2. 检查是否跳过儿化处理
            if (!MustErhua.Contains(word) && (NotErhua.Contains(word) || pos == "a" || pos == "j" || pos == "nr"))
            {
                return (initials, finals);
            }

            // 3. 长度校验
            if (finals.Count != word.Length)
            {
                return (initials, finals);
            }

            // 4. 执行儿化合并逻辑
            var newInitials = new List<string>();
            var newFinals = new List<string>();

            for (int i = 0; i < finals.Count; i++)
            {
                string phn = finals[i];
                // 判断是否合并儿化音
                if (i == finals.Count - 1 &&
                    word[i] == '儿' &&
                    (phn == "er2" || phn == "er5") &&
                    word.Length >= 2 &&
                    !NotErhua.Contains(word[^2..]) &&
                    newFinals.Count > 0)
                {
                    // 将 'er' 加上前一个字的声调
                    char tone = newFinals[^1][^1];
                    phn = "er" + (char.IsDigit(tone) ? tone : '5');
                }
                newInitials.Add(initials[i]);
                newFinals.Add(phn);
            }

            return (newInitials, newFinals);
        }
    }
}

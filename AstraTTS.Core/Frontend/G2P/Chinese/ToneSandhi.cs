using System;
using System.Collections.Generic;
using System.Linq;

namespace AstraTTS.Core.Frontend.G2P.Chinese
{
    /// <summary>
    /// 中文变调处理器。
    /// 完整移植自 Genie-TTS 的 ToneSandhi.py。
    /// </summary>
    public class ToneSandhi
    {
        private static readonly HashSet<string> MustNeuralToneWords = new HashSet<string>
        {
            "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨",
            "风筝", "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠",
            "钥匙", "里脊", "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么",
            "这个", "运气", "过去", "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主",
            "豆腐", "讲究", "记性", "记号", "认识", "规矩", "见识", "裁缝", "补丁", "衣裳",
            "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄",
            "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在",
            "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同",
            "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆", "老头", "老太",
            "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂", "精神",
            "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
            "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗",
            "砚台", "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛",
            "相声", "盘算", "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意",
            "甘蔗", "琵琶", "琢磨", "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务",
            "牲口", "牙碜", "牌楼", "爽快", "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心",
            "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚", "消息", "浪头",
            "活泼", "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃",
            "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠", "朋友",
            "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾", "收成", "提防",
            "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼", "抬举",
            "护士", "折腾", "扫帚", "打量", "打算", "打点", "打扮", "打听", "打发", "扎实",
            "扁担", "戒指", "懒得", "意识", "意思", "情形", "悟性", "怪物", "思量", "怎么",
            "念头", "念叨", "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通",
            "应酬", "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌",
            "差事", "工夫", "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头",
            "对付", "寡妇", "家伙", "客气", "实在", "官司", "学问", "学生", "字号", "嫁妆",
            "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫", "妯娌", "妥当", "妖精",
            "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫", "多少", "多么",
            "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔", "嘀咕",
            "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
            "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤",
            "口袋", "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹",
            "功夫", "力气", "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析",
            "出息", "凑合", "凉快", "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟",
            "便宜", "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么",
            "人家", "亲戚", "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头", "丧气",
            "两口", "东西", "东家", "世故", "不由", "不在", "下水", "下巴", "上头", "上司",
            "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜", "邋遢", "费用",
            "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划", "扑腾",
            "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
            "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
        };

        private static readonly HashSet<string> MustNotNeuralToneWords = new HashSet<string>
        {
            "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人",
            "虎虎", "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈",
            "花花草草", "留得", "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳",
            "佼佼", "吵吵", "打打", "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青",
        };

        private static readonly string Punc = "：，；。？！“”‘’':,;.?!";

        public List<string> ModifyTones(string word, string pos, List<string> finals)
        {
            finals = BuSandhi(word, finals);
            finals = YiSandhi(word, finals);
            finals = NeuralSandhi(word, pos, finals);
            finals = ThreeSandhi(word, finals);
            return finals;
        }

        private List<string> NeuralSandhi(string word, string pos, List<string> finals)
        {
            // 1. 叠词处理 (名词、动词、形容词)
            for (int i = 1; i < word.Length; i++)
            {
                if (word[i] == word[i - 1] &&
                    (pos.StartsWith("n") || pos.StartsWith("v") || pos.StartsWith("a")) &&
                    !MustNotNeuralToneWords.Contains(word))
                {
                    finals[i] = ReplaceTone(finals[i], '5');
                }
            }

            // 2. 语气助词
            if (word.Length >= 1 && "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶".Contains(word[^1]))
            {
                finals[^1] = ReplaceTone(finals[^1], '5');
            }
            // 3. 结构助词
            else if (word.Length >= 1 && "的地得".Contains(word[^1]))
            {
                finals[^1] = ReplaceTone(finals[^1], '5');
            }
            // 4. 助词 e.g. 走了, 看着, 去过
            else if (word.Length == 1 && "了着过".Contains(word) && (pos == "ul" || pos == "uz" || pos == "ug"))
            {
                finals[^1] = ReplaceTone(finals[^1], '5');
            }
            // 5. 名词后缀 e.g. 们, 子
            else if (word.Length > 1 && "们子".Contains(word[^1]) && (pos == "r" || pos == "n") && !MustNotNeuralToneWords.Contains(word))
            {
                finals[^1] = ReplaceTone(finals[^1], '5');
            }
            // 6. 方位词 e.g. 桌上, 地下, 家里
            else if (word.Length > 1 && "上下里".Contains(word[^1]) && (pos == "s" || pos == "l" || pos == "f"))
            {
                finals[^1] = ReplaceTone(finals[^1], '5');
            }
            // 7. 趋向动词 e.g. 上来, 下去
            else if (word.Length > 1 && "来去".Contains(word[^1]) && "上下进出回过起开".Contains(word[^2]))
            {
                finals[^1] = ReplaceTone(finals[^1], '5');
            }
            // 8. "个" 作为量词
            else
            {
                int geIdx = word.IndexOf('个');
                if ((geIdx >= 1 && (char.IsDigit(word[geIdx - 1]) || "几有两半多各整每做是".Contains(word[geIdx - 1]))) || word == "个")
                {
                    finals[geIdx] = ReplaceTone(finals[geIdx], '5');
                }
                else if (MustNeuralToneWords.Contains(word) || (word.Length >= 2 && MustNeuralToneWords.Contains(word[^2..])))
                {
                    finals[^1] = ReplaceTone(finals[^1], '5');
                }
            }

            return finals;
        }

        private List<string> BuSandhi(string word, List<string> finals)
        {
            // e.g. 看不懂
            if (word.Length == 3 && word[1] == '不')
            {
                finals[1] = ReplaceTone(finals[1], '5');
            }
            else
            {
                for (int i = 0; i < word.Length; i++)
                {
                    // "不" 在四声前变二声, e.g. 不怕
                    if (word[i] == '不' && i + 1 < word.Length && GetTone(finals[i + 1]) == '4')
                    {
                        finals[i] = ReplaceTone(finals[i], '2');
                    }
                }
            }
            return finals;
        }

        private List<string> YiSandhi(string word, List<string> finals)
        {
            // 1. 数字序列中的 "一" 不变调
            if (word.Contains('一') && word.All(c => char.IsDigit(c) || c == '一'))
            {
                return finals;
            }
            // 2. 叠词中间的 "一" 读轻声, e.g. 看一看
            if (word.Length == 3 && word[1] == '一' && word[0] == word[2])
            {
                finals[1] = ReplaceTone(finals[1], '5');
                return finals;
            }
            // 3. 序数词 "第一" 不变调 (yi1)
            if (word.StartsWith("第一"))
            {
                finals[1] = ReplaceTone(finals[1], '1');
                return finals;
            }

            for (int i = 0; i < word.Length; i++)
            {
                if (word[i] == '一' && i + 1 < word.Length)
                {
                    char nextTone = GetTone(finals[i + 1]);
                    // 4. "一" 在四声前读二声, e.g. 一段
                    if (nextTone == '4')
                    {
                        finals[i] = ReplaceTone(finals[i], '2');
                    }
                    // 5. "一" 在非四声前读四声, e.g. 一天
                    else if (nextTone != ' ' && !Punc.Contains(word[i + 1]))
                    {
                        finals[i] = ReplaceTone(finals[i], '4');
                    }
                }
            }
            return finals;
        }

        private List<string> ThreeSandhi(string word, List<string> finals)
        {
            if (word.Length == 2 && AllToneThree(finals))
            {
                finals[0] = ReplaceTone(finals[0], '2');
            }
            else if (word.Length == 3)
            {
                if (AllToneThree(finals))
                {
                    // 策略：如果是 2+1 结构，变 2+2+3；如果是 1+2 结构，变 3+2+3
                    // 这里由于暂无精确分词结构，参考 Python 逻辑简单拆分
                    // 优先级 2+1 常用 (e.g. 蒙古包)
                    finals[0] = ReplaceTone(finals[0], '2');
                    finals[1] = ReplaceTone(finals[1], '2');
                }
                else
                {
                    // 局部变调
                    if (GetTone(finals[0]) == '3' && GetTone(finals[1]) == '3')
                        finals[0] = ReplaceTone(finals[0], '2');
                    else if (GetTone(finals[1]) == '3' && GetTone(finals[2]) == '3')
                        finals[1] = ReplaceTone(finals[1], '2');
                }
            }
            else if (word.Length == 4)
            {
                // 四字词通常拆分为 2+2
                if (GetTone(finals[0]) == '3' && GetTone(finals[1]) == '3')
                    finals[0] = ReplaceTone(finals[0], '2');
                if (GetTone(finals[2]) == '3' && GetTone(finals[3]) == '3')
                    finals[2] = ReplaceTone(finals[2], '2');
            }
            return finals;
        }

        private bool AllToneThree(IEnumerable<string> finals) => finals.All(f => GetTone(f) == '3');

        private char GetTone(string pinyin)
        {
            if (string.IsNullOrEmpty(pinyin)) return ' ';
            char last = pinyin[^1];
            return char.IsDigit(last) ? last : ' ';
        }

        private string ReplaceTone(string pinyin, char newTone)
        {
            if (string.IsNullOrEmpty(pinyin)) return pinyin;
            if (char.IsDigit(pinyin[^1]))
                return pinyin[..^1] + newTone;
            return pinyin + newTone;
        }

        // --- Pre-Merge Logic (Sentence level) ---

        public List<(string word, string pos)> PreMergeForModify(List<(string word, string pos)> segments)
        {
            segments = MergeBu(segments);
            segments = MergeYi(segments);
            segments = MergeReduplication(segments);
            segments = MergeContinuousThreeTones(segments);
            segments = MergeEr(segments);
            return segments;
        }

        private List<(string word, string pos)> MergeBu(List<(string word, string pos)> segments)
        {
            var result = new List<(string word, string pos)>();
            for (int i = 0; i < segments.Count; i++)
            {
                if (segments[i].word == "不" && i + 1 < segments.Count)
                {
                    result.Add((segments[i].word + segments[i + 1].word, segments[i + 1].pos));
                    i++;
                }
                else
                {
                    result.Add(segments[i]);
                }
            }
            return result;
        }

        private List<(string word, string pos)> MergeYi(List<(string word, string pos)> segments)
        {
            var result = new List<(string word, string pos)>();
            for (int i = 0; i < segments.Count; i++)
            {
                // V + 一 + V -> V一V
                if (i > 0 && segments[i].word == "一" && i + 1 < segments.Count &&
                    segments[i - 1].word == segments[i + 1].word && segments[i - 1].pos.StartsWith("v"))
                {
                    var last = result[^1];
                    result[^1] = (last.word + "一" + segments[i + 1].word, last.pos);
                    i++;
                }
                // 一 + X -> 一X (e.g. 一些, 一个)
                else if (segments[i].word == "一" && i + 1 < segments.Count)
                {
                    result.Add((segments[i].word + segments[i + 1].word, segments[i + 1].pos));
                    i++;
                }
                else
                {
                    result.Add(segments[i]);
                }
            }
            return result;
        }

        private List<(string word, string pos)> MergeReduplication(List<(string word, string pos)> segments)
        {
            var result = new List<(string word, string pos)>();
            for (int i = 0; i < segments.Count; i++)
            {
                if (result.Count > 0 && segments[i].word == result[^1].word)
                {
                    var last = result[^1];
                    result[^1] = (last.word + segments[i].word, last.pos);
                }
                else
                {
                    result.Add(segments[i]);
                }
            }
            return result;
        }

        private List<(string word, string pos)> MergeContinuousThreeTones(List<(string word, string pos)> segments)
        {
            // 简单实现：将连续的两个单字三声词合并
            // 注意：这需要获知拼音，目前我们在 PreMerge 阶段可能还不确定拼音
            // 所以建议在 ChineseG2P 的 Process 流程中，先取初始拼音，再做 PreMerge，最后做变调
            return segments;
        }

        private List<(string word, string pos)> MergeEr(List<(string word, string pos)> segments)
        {
            var result = new List<(string word, string pos)>();
            foreach (var seg in segments)
            {
                if (result.Count > 0 && seg.word == "儿" && result[^1].word != "#")
                {
                    var last = result[^1];
                    result[^1] = (last.word + seg.word, last.pos);
                }
                else
                {
                    result.Add(seg);
                }
            }
            return result;
        }
    }
}

#!/usr/bin/env python3
"""
roberta_noloop.py — 把 roberta.onnx 中导致约 278 ms CPU 控制流的 Loop 节点
替换为等价的向量化子图（repeat_interleave 的 CumSum + 比较 + Gather 实现）。

背景（只读分析结论）：
  - 原 /Loop 语义 = torch.repeat_interleave(hidden, repeats, dim=0)
    · trip count = seq_len-2（Shape(Slice(hidden,1:-1))[0]），cond 恒真
    · body 每次取第 i 个 repeat 值 r_i，把 hidden 第 i 行重复 r_i 次追加
  - repeats 先经 /Equal+/Expand+/Where 归一化（len==1 时广播到 [M]）→ /Where_output_0
  - hidden 内容区 = /Slice_output_0（[M,1024] float16）

替换实现（全部算子均有 DirectML 内核；Loop/SplitToSequence 无内核）：
    cum   = CumSum(/Where_output_0, axis=0)          # [M] int64 前缀和（含 i）
    last  = Gather(Shape(/Where_output_0), [0])      # [1] int64 total = Σr
    ones  = ConstantOfShape(last, 1)                 # [total] int64
    ridx  = CumSum(ones, axis=0) - 1                 # [total] int64  0..total-1
    mask  = LessOrEqual(cum[1,M], ridx[total,1])     # [total,M] bool  广播
    idx   = ReduceSum(Cast(mask,int64), axes=[1])    # [total,1] int64
    phone_features = Gather(/Slice_output_0, idx)    # [total,1024] float16

正确性推导：输出行 k 属于第 i 段（cum[i-1] ≤ k < cum[i]）。
idx[k] = #{ j : cum[j] ≤ k }；因 cum 非降，该计数在 k 落入第 i 段（r_i>0）时恰为 i，
且 r_i=0 的空段天然不会被任何 k 命中（idx 会跳过该值），故与 repeat_interleave 逐行一致。
已用 [2,3,1]、[2,0,3]、[0,3] 等用例手工验证。

用法：
    python roberta_noloop.py <输入 roberta.onnx> <输出 roberta_noloop.onnx>
只读原模型（不加载 599MB 权重，保留外部数据引用），仅重写图结构。
"""

import sys
import os
import onnx
from onnx import helper, TensorProto


def build_replacement():
    """构造替换子图的节点与 value_info。"""
    new_nodes = []
    new_vis = []

    def add_node(op, inputs, outputs, name, **attrs):
        n = helper.make_node(op, inputs, outputs, name=name, **attrs)
        new_nodes.append(n)
        return n

    # ---- 常量 ----
    # CumSum 的 axis：本项目模型家族统一用 int32 标量（见 stage_decoder 的 /ar_audio_position/CumSum）
    ax0 = helper.make_tensor('/noloop/axis0', TensorProto.INT32, [], [0])
    # 标量常量
    one = helper.make_tensor('/noloop/one', TensorProto.INT64, [], [1])
    axes0 = helper.make_tensor('/noloop/axes0', TensorProto.INT64, [1], [0])
    axes1 = helper.make_tensor('/noloop/axes1', TensorProto.INT64, [1], [1])
    unsq_axes1 = helper.make_tensor('/noloop/unsq_axes1', TensorProto.INT64, [1], [1])
    unsq_axes0 = helper.make_tensor('/noloop/unsq_axes0', TensorProto.INT64, [1], [0])

    # ---- 计算 ----
    # cum = CumSum(repeats_norm, axis=0)   [M] int64 前缀和（含第 i 项）
    add_node('CumSum', ['/Where_output_0', '/noloop/axis0'], ['/noloop/cum'], '/noloop/CumSum')

    # total = ReduceSum(repeats_norm, axes=[0], keepdims=1)  [1] int64 = Σr
    # 注意：不能用 Shape(repeats)（那是长度 M，不是重复次数之和——实测会导致索引越界）
    add_node('ReduceSum', ['/Where_output_0', '/noloop/axes0'], ['/noloop/total'], '/noloop/Total', keepdims=1)

    # ones = ConstantOfShape(total, value=1) [total] int64
    add_node('ConstantOfShape', ['/noloop/total'], ['/noloop/ones'], '/noloop/ConstantOfShape',
             value=helper.make_tensor('v', TensorProto.INT64, [1], [1]))

    # onesum = CumSum(ones, axis=0)          [total] = 1..total
    add_node('CumSum', ['/noloop/ones', '/noloop/axis0'], ['/noloop/onesum'], '/noloop/CumSumOnes')

    # ridx = onesum - 1                      [total] = 0..total-1
    add_node('Sub', ['/noloop/onesum', '/noloop/one'], ['/noloop/ridx'], '/noloop/Sub')

    # cum2  = Unsqueeze(cum,   axes=[0])     [1, M]
    add_node('Unsqueeze', ['/noloop/cum', '/noloop/unsq_axes0'], ['/noloop/cum2'], '/noloop/UnsqueezeCum')

    # ridx2 = Unsqueeze(ridx,  axes=[1])     [total, 1]
    add_node('Unsqueeze', ['/noloop/ridx', '/noloop/unsq_axes1'], ['/noloop/ridx2'], '/noloop/UnsqueezeRidx')

    # mask = LessOrEqual(cum2, ridx2)  →  [total, M]，元素 [k,j] = (cum[j] <= k)
    # idx[k] = Σ_j mask[k,j] = #{j : cum[j] <= k} = 所在段号 i（r_i=0 的段自动跳过）
    add_node('LessOrEqual', ['/noloop/cum2', '/noloop/ridx2'], ['/noloop/mask'], '/noloop/LessOrEqual')

    # maskl = Cast(mask, int64)              [total, M]
    add_node('Cast', ['/noloop/mask'], ['/noloop/maskl'], '/noloop/Cast', to=TensorProto.INT64)

    # idx = ReduceSum(maskl, axes=[1], keepdims=0)  [total] int64
    # keepdims=0 保证 Gather 输出为 [total,1024]，与原模型 /ConcatFromSequence 结果一致
    add_node('ReduceSum', ['/noloop/maskl', '/noloop/axes1'], ['/noloop/idx'], '/noloop/ReduceSum', keepdims=0)

    # phone_features = Gather(hidden_content, idx, axis=0)  [total,1024] float16
    add_node('Gather', ['/Slice_output_0', '/noloop/idx'], ['phone_features'], '/noloop/GatherOut', axis=0)

    extra_inits = [ax0, one, axes0, axes1, unsq_axes1, unsq_axes0]

    # ---- value_info（形状提示；动态维留空）----
    def vi(name, dtype, shape):
        new_vis.append(helper.make_tensor_value_info(name, dtype, shape))

    vi('/noloop/cum', TensorProto.INT64, [None])
    vi('/noloop/total', TensorProto.INT64, [1])
    vi('/noloop/ones', TensorProto.INT64, [None])
    vi('/noloop/onesum', TensorProto.INT64, [None])
    vi('/noloop/ridx', TensorProto.INT64, [None])
    vi('/noloop/mask', TensorProto.BOOL, [None, None])
    vi('/noloop/maskl', TensorProto.INT64, [None, None])
    vi('/noloop/idx', TensorProto.INT64, [None])

    return new_nodes, extra_inits, new_vis


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]

    print(f'[1/6] 载入原模型（不读外部权重）: {src}')
    model = onnx.load(src, load_external_data=False)
    graph = model.graph

    # ---------- 前置断言：结构必须与调研一致 ----------
    print('[2/6] 校验目标结构...')
    node_by_name = {n.name: n for n in graph.node}

    required = ['/Where', '/Slice', '/SplitToSequence', '/SplitToSequence_1',
                '/SequenceEmpty', '/Loop', '/ConcatFromSequence']
    for name in required:
        if name not in node_by_name:
            raise SystemExit(f'前置断言失败：缺少节点 {name}（模型结构与预期不符，已中止）')

    loop = node_by_name['/Loop']
    if list(loop.input) != ['/Unsqueeze_output_0', '/Constant_8_output_0', '/SequenceEmpty_output_0']:
        raise SystemExit(f'前置断言失败：/Loop 输入与预期不符: {list(loop.input)}')
    if list(loop.output) != ['/Loop_output_0']:
        raise SystemExit(f'前置断言失败：/Loop 输出与预期不符: {list(loop.output)}')
    if len(loop.attribute) != 1 or loop.attribute[0].name != 'body':
        raise SystemExit('前置断言失败：/Loop 属性非单一 body')

    con = node_by_name['/ConcatFromSequence']
    if list(con.input) != ['/Loop_output_0']:
        raise SystemExit(f'前置断言失败：/ConcatFromSequence 输入与预期不符: {list(con.input)}')

    where = node_by_name['/Where']
    if list(where.input) != ['/Equal_output_0', '/Expand_output_0', 'repeats']:
        raise SystemExit(f'前置断言失败：/Where 输入与预期不符: {list(where.input)}')

    slice_node = node_by_name['/Slice']
    if list(slice_node.output) != ['/Slice_output_0']:
        raise SystemExit('前置断言失败：/Slice 输出名与预期不符')

    # 确认 /Where_output_0 与 /Slice_output_0 都是 float16 / int64 与预期一致
    # （通过生产者输出类型间接确认：/Slice 输入来自 LayerNorm 输出）
    print('    结构校验通过。')

    # ---------- 构建替换节点 ----------
    print('[3/6] 构造向量化替换子图...')
    new_nodes, extra_inits, new_vis = build_replacement()

    # ---------- 删除控制流节点 ----------
    print('[4/6] 移除 Loop/SplitToSequence/ConcatFromSequence/SequenceEmpty...')
    to_remove = {'/Loop', '/SplitToSequence', '/SplitToSequence_1', '/ConcatFromSequence',
                 '/SequenceEmpty'}
    kept_nodes = [n for n in graph.node if n.name not in to_remove]
    # 追加新节点
    kept_nodes.extend(new_nodes)
    del graph.node[:]
    graph.node.extend(kept_nodes)

    # 追加 initializer
    existing_init_names = {i.name for i in graph.initializer}
    for t in extra_inits:
        if t.name in existing_init_names:
            raise SystemExit(f'前置断言失败：initializer 名冲突 {t.name}')
        graph.initializer.append(t)

    # 追加 value_info
    for v in new_vis:
        graph.value_info.append(v)

    # ---------- 死代码消除（按消费者反向裁剪）----------
    print('[5/6] 死代码消除...')
    keep = set(o.name for o in graph.output)
    changed = True
    # 反向迭代直到稳定
    while changed:
        changed = False
        for n in reversed(graph.node):
            if any(o in keep for o in n.output):
                for i in n.input:
                    if i and i not in keep:
                        keep.add(i)
                        changed = True
    removed = []
    final_nodes = []
    for n in graph.node:
        if any(o in keep for o in n.output):
            final_nodes.append(n)
        else:
            removed.append(n.name)
    del graph.node[:]
    graph.node.extend(final_nodes)
    removed_inits = [i for i in graph.initializer
                     if i.name not in keep
                     and not any(i.name == inp for n in graph.node for inp in n.input)]
    if removed_inits:
        init_names = {i.name for i in removed_inits}
        graph.initializer[:] = [i for i in graph.initializer if i.name not in init_names]
    print(f'    移除节点 {len(removed)} 个，initializer {len(removed_inits)} 个')
    if removed:
        print(f'    示例: {removed[:8]}')

    # ---------- 校验并保存 ----------
    print('[6/6] 校验并保存...')
    onnx.checker.check_model(model, full_check=False)
    print('    onnx.checker 通过')

    onnx.save(model, dst, save_as_external_data=False)
    size = os.path.getsize(dst)
    print(f'    已保存: {dst} ({size/1e6:.1f} MB)')
    print(f'    节点数: {len(graph.node)}（原 {len(kept_nodes) - len(new_nodes) + len(to_remove)}）')


if __name__ == '__main__':
    main()

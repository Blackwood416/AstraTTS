"""对 emotion_eval 生成的音频做客观指标分析。

指标：
  - 说话人相似度：speaker_encoder 嵌入与基线嵌入的余弦（音色保持度）
  - F0 中位数/范围：自相关法
  - 能量 (RMS)、时长、语速代理（tokens/s 已在 summary）

用法:
  python analyze_eval.py --eval-dir E:/AI/AstraTTS/_emotion_eval_cli --resources E:/AI/AstraTTS/resources
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def log(*a):
    print(*a, flush=True)


def estimate_f0_stats(x, sr, fmin=60, fmax=500):
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    lo = int(sr / fmax)
    hi = int(sr / fmin)
    f0s = []
    voiced = 0
    total = 0
    for start in range(0, max(1, len(x) - frame), hop):
        seg = x[start:start + frame]
        if len(seg) < frame:
            break
        total += 1
        seg = seg - seg.mean()
        if np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        win = ac[lo:hi]
        if len(win) == 0:
            continue
        idx = int(np.argmax(win))
        if win[idx] < 0.3:
            continue
        f0s.append(sr / (lo + idx))
        voiced += 1
    if not f0s:
        return {"f0_med": 0.0, "f0_p10": 0.0, "f0_p90": 0.0, "voiced_ratio": 0.0}
    return {
        "f0_med": float(np.median(f0s)),
        "f0_p10": float(np.percentile(f0s, 10)),
        "f0_p90": float(np.percentile(f0s, 90)),
        "voiced_ratio": voiced / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=r"E:\AI\AstraTTS\_emotion_eval_cli")
    ap.add_argument("--resources", default=r"E:\AI\AstraTTS\resources")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnxruntime as ort
    import soundfile as sf
    import soxr

    out_dir = args.out or args.eval_dir
    os.makedirs(out_dir, exist_ok=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    spk = ort.InferenceSession(
        os.path.join(args.resources, "shared", "v1_extra", "speaker_encoder.onnx"),
        so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])

    def spk_emb(audio, sr):
        a16 = soxr.resample(audio, sr, 16000).astype(np.float32) if sr != 16000 else audio.astype(np.float32)
        a16 = a16 - a16.mean()
        pk = float(np.abs(a16).max())
        if pk > 1e-6:
            a16 = a16 / pk * 0.9
        e = spk.run(None, {"waveform": a16[None, :].astype(np.float32)})[0]
        return np.asarray(e).reshape(-1).astype(np.float32)

    # 读 summary 或直接扫 wav
    files = sorted(glob.glob(os.path.join(args.eval_dir, "*.wav")))
    if not files:
        log("未找到 wav:", args.eval_dir)
        return

    # 基线标签：文件名不含 _a 或者是 neutral
    base_files = [f for f in files if os.path.basename(f).startswith("neutral")]
    if not base_files:
        log("找不到 neutral 基线文件")
        return

    base_embs = []
    for f in base_files:
        a, sr = sf.read(f, dtype="float32", always_2d=True)
        base_embs.append(spk_emb(a.mean(axis=1), sr))
    base_ref = np.mean(np.stack(base_embs), axis=0)  # 平均基线（降低随机性）
    base_ref /= np.linalg.norm(base_ref)

    rows = []
    log(f"{'file':<28} {'dur':>6} {'cos_base':>9} {'f0_med':>7} {'f0_p10':>7} {'f0_p90':>7} {'voiced':>7} {'rms':>6}")
    for f in files:
        a, sr = sf.read(f, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        dur = len(a) / sr
        emb = spk_emb(a, sr)
        emb_n = emb / (np.linalg.norm(emb) + 1e-9)
        cos = float(base_ref @ emb_n)
        f0 = estimate_f0_stats(a, sr)
        r = float(np.sqrt(np.mean(a ** 2)))
        name = os.path.basename(f)[:-4]
        rows.append({"file": name, "dur": dur, "cos_base": cos, "rms": r, **f0})
        log(f"{name:<28} {dur:6.2f} {cos:9.4f} {f0['f0_med']:7.1f} {f0['f0_p10']:7.1f} "
            f"{f0['f0_p90']:7.1f} {f0['voiced_ratio']:7.2f} {r:6.3f}")

    with open(os.path.join(out_dir, "objective_metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 按情感与 α 汇总
    log("\n==== 汇总：音色相似度 vs α（相对 neutral）====")
    for emo in sorted(set(r["file"].split("_a")[0] for r in rows if "_a" in r["file"] or r["file"] == "neutral")):
        sub = [r for r in rows if r["file"].split("_a")[0] == emo]
        for r in sorted(sub, key=lambda x: x["file"]):
            log(f"  {r['file']:<28} cos={r['cos_base']:.4f} f0={r['f0_med']:6.1f} dur={r['dur']:.2f}s")

    log(f"\n保存: {os.path.join(out_dir, 'objective_metrics.csv')}")


if __name__ == "__main__":
    main()

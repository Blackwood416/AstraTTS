"""从 CSEMOTIONS 提取情感方向向量 τ。

关键设计（按"配对差"公式，避免说话人身份混入）：
    τ_e = mean_{s,i} ( x_e^{s,i} - x_n^{s} )
其中 s 是说话人，i 是该说话人的第 i 条样本，x_n^{s} 是同一说话人的中性均值。

数据集样本量：每说话人每情感约 59 条（4160 / 10 / 7）。
默认每说话人每情感取 N 条（默认 20），确保音频可解码。

用法:
    python extract_tau.py --parquet-dir E:\\AI\\_tmp --out E:\\AI\\AstraTTS\\resources\\emotion_vectors
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np


def log(*a):
    print(*a, flush=True)


EMOTIONS = ["neutral", "happy", "angry", "sad", "surprise", "playfulness", "fearful"]
EMO_ALIAS = {
    "neutral": "neutral", "happy": "happy", "angry": "angry", "sad": "sad",
    "surprise": "surprise", "playful": "playfulness", "playfulness": "playfulness",
    "fearful": "fearful", "fear": "fearful",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default=r"E:\AI\_tmp")
    ap.add_argument("--resources", default=r"E:\AI\AstraTTS\resources")
    ap.add_argument("--out", default=r"E:\AI\AstraTTS\resources\emotion_vectors")
    ap.add_argument("--per-speaker-emotion", type=int, default=20)
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--threads", type=int, default=2, help="CPU 线程数（低干扰）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"],
                    help="auto: 优先 DirectML 独显，失败回退 CPU")
    ap.add_argument("--save-wav", action="store_true", help="同时导出裁剪后的 wav（便于人工试听）")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import onnxruntime as ort
    import soundfile as sf
    import soxr

    import faulthandler
    faulthandler.enable()

    t0 = time.time()
    spk_path = os.path.join(args.resources, "shared", "v1_extra", "speaker_encoder.onnx")
    if not os.path.exists(spk_path):
        log("缺少 speaker_encoder.onnx:", spk_path)
        sys.exit(1)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if args.device in ("auto", "dml"):
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif args.device == "dml":
            log("请求 DML 但不可用，回退 CPU")
    log("使用 providers:", providers)
    spk = ort.InferenceSession(spk_path, so, providers=providers)
    in_name = spk.get_inputs()[0].name
    log(f"speaker_encoder 输入: {in_name}, 输出: {spk.get_outputs()[0].name} {spk.get_outputs()[0].shape}")

    files = sorted(glob.glob(os.path.join(args.parquet_dir, "csemo_*.parquet")))
    if not files:
        log("未找到 parquet 分片:", args.parquet_dir)
        sys.exit(1)
    log("分片:", [os.path.basename(f) for f in files])

    def emb_of(audio, sr):
        """音频 -> 16k -> speaker encoder -> 20480 维嵌入"""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = soxr.resample(a, sr, 16000).astype(np.float32)
        a = a - a.mean()
        peak = float(np.abs(a).max())
        if peak > 1e-6:
            a = a / peak * 0.9
        e = spk.run(None, {in_name: a[None, :].astype(np.float32)})[0]
        return np.asarray(e).reshape(-1).astype(np.float32)

    # ---------- 1. 收集样本 ----------
    # per speaker -> per emotion -> list of embeddings
    store = {}          # speaker -> {emotion: [emb,...]}
    meta = []           # 记录来源，便于追溯/试听
    counts = {}

    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        log(f"\n[{fi+1}/{len(files)}] {os.path.basename(f)}: {pf.metadata.num_rows} 行")
        for batch in pf.iter_batches(batch_size=64, columns=["audio", "emotion", "speaker", "text"]):
            d = batch.to_pydict()
            n = len(d["emotion"])
            for i in range(n):
                speaker = str(d["speaker"][i]).strip()
                emo_raw = str(d["emotion"][i]).strip().lower()
                emo = EMO_ALIAS.get(emo_raw)
                if emo is None:
                    continue
                counts.setdefault((speaker, emo), 0)
                if counts[(speaker, emo)] >= args.per_speaker_emotion:
                    continue

                au = d["audio"][i]
                if not au:
                    continue
                # parquet 中 audio 为 struct<bytes: binary, path: string>，从 bytes 解码
                raw = au.get("bytes") if isinstance(au, dict) else None
                if raw is None:
                    continue
                try:
                    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
                except Exception as ex:
                    log(f"   解码失败 {speaker}/{emo}: {ex}")
                    continue
                arr = arr.mean(axis=1)
                sr = int(sr)
                dur = len(arr) / sr
                if dur < args.min_sec:
                    continue
                if args.max_sec > 0 and dur > args.max_sec:
                    # 截取中间段，保持情感表达最充分的区域
                    c = len(arr) // 2
                    half = int(args.max_sec * sr / 2)
                    arr = arr[max(0, c - half): c + half]
                try:
                    e = emb_of(arr, sr)
                except Exception as ex:
                    log(f"   跳过 {speaker}/{emo}: {ex}")
                    continue
                store.setdefault(speaker, {}).setdefault(emo, []).append(e)
                counts[(speaker, emo)] += 1
                meta.append({"speaker": speaker, "emotion": emo, "dur": round(dur, 2),
                             "text": (d["text"][i] or "")[:40], "shard": os.path.basename(f)})

                if args.save_wav:
                    wavdir = os.path.join(args.out, "samples", speaker)
                    os.makedirs(wavdir, exist_ok=True)
                    p = os.path.join(wavdir, f"{emo}_{counts[(speaker,emo)]:03d}.wav")
                    sf.write(p, arr, sr)

        log("  当前样本数:", {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())[:6]}, "...")

    log(f"\n收集完成: {len(meta)} 条样本, {len(store)} 位说话人, 耗时 {time.time()-t0:.1f}s")
    for spk_id in sorted(store):
        row = {e: len(store[spk_id].get(e, [])) for e in EMOTIONS}
        log(f"  {spk_id}: {row}")

    # ---------- 2. 配对差计算 τ ----------
    log("\n==== 配对差 τ 计算 ====")
    taus = {}
    diag = {}
    for emo in EMOTIONS:
        if emo == "neutral":
            continue
        diffs = []
        per_speaker = {}
        for s in sorted(store):
            xs = store[s]
            if "neutral" not in xs or not xs["neutral"]:
                continue
            if emo not in xs or not xs[emo]:
                continue
            mean_n = np.mean(np.stack(xs["neutral"]), axis=0)
            mean_e = np.mean(np.stack(xs[emo]), axis=0)
            diffs.append(mean_e - mean_n)
            per_speaker[s] = {"n_emo": len(xs[emo]), "n_neu": len(xs["neutral"]),
                              "diff_norm": float(np.linalg.norm(mean_e - mean_n))}
        if not diffs:
            log(f"  {emo}: 无配对样本")
            continue
        D = np.stack(diffs)
        tau = D.mean(axis=0)
        # 诊断：跨说话人一致性 = 平均向量范数 / 单个差向量平均范数（越接近 1 越一致）
        consistency = float(np.linalg.norm(tau) / (np.linalg.norm(D, axis=1).mean() + 1e-9))
        taus[emo] = tau
        diag[emo] = {
            "n_speakers": len(diffs),
            "tau_norm": float(np.linalg.norm(tau)),
            "mean_diff_norm": float(np.linalg.norm(D, axis=1).mean()),
            "std_diff_norm": float(np.linalg.norm(D, axis=1).std()),
            "consistency": consistency,
            "per_speaker": per_speaker,
        }
        log(f"  {emo:<12} n_spk={len(diffs)}  |τ|={np.linalg.norm(tau):8.3f}  "
            f"mean|Δ|={diag[emo]['mean_diff_norm']:8.3f}  consistency={consistency:.4f}")

    # ---------- 3. 正交性与归一化 ----------
    log("\n==== τ 之间余弦相似度（应较低，说明方向可区分）====")
    names = [e for e in EMOTIONS if e != "neutral" and e in taus]
    for i in range(len(names)):
        row = []
        for j in range(len(names)):
            a, b = taus[names[i]], taus[names[j]]
            c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            row.append(f"{c:+.3f}")
        log(f"  {names[i]:<12} " + " ".join(row))

    # ---------- 4. 保存 ----------
    os.makedirs(args.out, exist_ok=True)
    # 4a. npz（数值分析用）
    np.savez(os.path.join(args.out, "tau_vectors.npz"), **{k: v for k, v in taus.items()})
    # 4b. 归一化后的二进制 float32（C# 直接 mmap 式读取）+ json 元数据
    manifest = {
        "format": "astra-emotion-tau",
        "version": 1,
        "dim": int(len(next(iter(taus.values())))),
        "source_dataset": "AIDC-AI/CSEMOTIONS (ATH-MaaS/CSEMOTIONS)",
        "method": "paired-difference: tau_e = mean_s(mean_i x_e^{s,i} - mean_j x_n^{s,j})",
        "samples_per_speaker_emotion": args.per_speaker_emotion,
        "n_samples_total": len(meta),
        "emotions": {},
    }
    for e, tau in taus.items():
        n = tau / (np.linalg.norm(tau) + 1e-9)
        with open(os.path.join(args.out, f"tau_{e}.f32"), "wb") as fh:
            fh.write(n.astype("<f4").tobytes())
        manifest["emotions"][e] = {
            "file": f"tau_{e}.f32",
            "dim": int(tau.size),
            "tau_norm": diag[e]["tau_norm"],
            "consistency": diag[e]["consistency"],
            "n_speakers": diag[e]["n_speakers"],
            "normalized": True,
        }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diag, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "samples_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    log(f"\n已保存到 {args.out}")
    log("  tau_vectors.npz / tau_<emotion>.f32 / manifest.json / diagnostics.json")
    log(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""AstraTTS 情感注入原型：验证 sv_emb 空间的情感方向向量是否有效。

链路（与 C# 推理完全一致）：
  wav(16k) --speaker_encoder.onnx--> sv_emb [20480]
  wav(32k) + sv_emb --prompt_encoder.onnx--> ge [1024], ge_advanced [512]
  ge/ge_advanced --> vits.onnx (风格分支)

本脚本：验证 sv_emb 空间的可操作性（不同音频的嵌入差异、注入后的 ge 变化）。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def log(*a):
    print(*a, flush=True)


def main():
    import onnxruntime as ort
    import soundfile as sf

    try:
        import soxr
        def resample(x, sr_in, sr_out):
            return soxr.resample(x, sr_in, sr_out).astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        def resample(x, sr_in, sr_out):
            g = gcd(sr_in, sr_out)
            return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)

    RES = os.environ.get("ASTRA_RES", r"E:\AI\AstraTTS\resources")
    spk_path = os.path.join(RES, "shared", "v1_extra", "speaker_encoder.onnx")
    prm_path = os.path.join(RES, "models_v1", "chenxing", "prompt_encoder.onnx")

    log("== loading models ==")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    spk = ort.InferenceSession(spk_path, so, providers=["CPUExecutionProvider"])
    prm = ort.InferenceSession(prm_path, so, providers=["CPUExecutionProvider"])
    log("speaker_encoder in:", [(i.name, i.shape) for i in spk.get_inputs()],
        "out:", [(o.name, o.shape) for o in spk.get_outputs()])
    log("prompt_encoder  in:", [(i.name, i.shape) for i in prm.get_inputs()],
        "out:", [(o.name, o.shape) for o in prm.get_outputs()])

    def read_audio_16k(path):
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        return resample(a, sr, 16000)

    def read_audio_32k(path):
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        return resample(a, sr, 32000)

    def get_sv_emb(path):
        a16 = read_audio_16k(path)
        return spk.run(None, {"waveform": a16[None, :]})[0][0]

    def get_ge(sv_emb, path):
        a32 = read_audio_32k(path)
        ge, ge_adv = prm.run(None, {"ref_audio": a32[None, :],
                                    "sv_emb": sv_emb[None, :].astype(np.float32)})
        return np.asarray(ge).reshape(-1), np.asarray(ge_adv).reshape(-1)  # [1024], [512]

    av = os.path.join(RES, "avatars", "chenxing", "references")
    refs = [f for f in os.listdir(av) if f.endswith(".wav")]
    log("\n== reference audios ==")
    for f in refs:
        log("  ", f)

    # 1) 同一个 speaker encoder 对不同音频的输出差异
    log("\n== sv_emb across reference audios ==")
    embs, names = [], []
    for f in refs:
        p = os.path.join(av, f)
        e = get_sv_emb(p)
        embs.append(e)
        names.append(f)
        log(f"   {f[:20]:<22} norm={np.linalg.norm(e):8.3f} mean={e.mean():+.4f} std={e.std():.4f}")

    if len(embs) >= 2:
        e0, e1 = embs[0], embs[1]
        n0, n1 = np.linalg.norm(e0), np.linalg.norm(e1)
        cos = float(e0 @ e1 / (n0 * n1))
        log(f"\n   cosine(e0,e1) = {cos:.4f}   |e0-e1| = {np.linalg.norm(e0-e1):.3f} "
            f"(={np.linalg.norm(e0-e1)/n0*100:.1f}% of norm)")

    # 2) 注入链路：ge 对 sv_emb 扰动的敏感度
    log("\n== ge sensitivity to sv_emb perturbation ==")
    base = embs[0]
    ge0, gadv0 = get_ge(base, os.path.join(av, refs[0]))

    rng = np.random.default_rng(0)
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        rand_dir = rng.standard_normal(base.shape).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)
        new = (base + alpha * np.linalg.norm(base) * rand_dir).astype(np.float32)
        ge1, gadv1 = get_ge(new, os.path.join(av, refs[0]))
        dg = np.linalg.norm(ge1 - ge0) / max(np.linalg.norm(ge0), 1e-9)
        dga = np.linalg.norm(gadv1 - gadv0) / max(np.linalg.norm(gadv0), 1e-9)
        cos_g = float(ge1 @ ge0 / (np.linalg.norm(ge1) * np.linalg.norm(ge0)))
        log(f"   alpha={alpha:<5}  rel|dge|={dg:.4f}  rel|dge_adv|={dga:.4f}  cos(ge,ge')={cos_g:.5f}")

    # 3) alpha=0 应该逐位一致（链路无副作用）
    ge_z, gadv_z = get_ge(base, os.path.join(av, refs[0]))
    log(f"\n   bitwise check alpha=0: max|dge|={np.abs(ge_z-ge0).max():.3e} "
        f"max|dge_adv|={np.abs(gadv_z-gadv0).max():.3e}")

    np.save(os.path.join(os.path.dirname(__file__), "emb_ref0.npy"), embs[0])


if __name__ == "__main__":
    main()

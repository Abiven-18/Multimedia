"""
benchmark.py - PhysioSync-AR Quantitative Evaluation
=====================================================
Measures:
  1. Throughput (FPS) - PhysioSync-AR vs dense Farneback optical flow
  2. Compute savings  - arithmetic-operation proxy comparison
  3. F1 / Precision / Recall - keyframe selection vs audio-energy ground truth

Both methods run on video.mp4 + audio.wav (real data).
Ground truth is derived from audio RMS energy peaks
(shared signal observable by both modalities).

Usage:
    python benchmark.py
"""

import time, json, os
import numpy as np
import cv2, librosa
from sklearn.metrics import f1_score, precision_score, recall_score

from audio_analyzer    import AudioSpatialAnalyzer
from visual_analyzer   import VisualPoseAnalyzer
from fusion_engine     import FusionEngine
from keyframe_selector import KeyframeSelector

VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def pr(label, value, unit=""):
    print(f"  {label:<50} {value} {unit}")


def sep(title):
    print(f"\n{'-'*65}\n  {title}\n{'-'*65}")


# ---------------------------------------------------------------------------
# BASELINE  -  Dense Farneback Optical Flow
# ---------------------------------------------------------------------------

def run_dense_of(video_path):
    """Every frame, full resolution, Farneback OF magnitude as importance."""
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores, times, pixel_ops, prev_gray = [], [], 0, None

    t0 = time.perf_counter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        pixel_ops += h * w * 2          # u + v per pixel
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            scores.append(float(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()))
            times.append(ts)
        prev_gray = gray
    cap.release()
    elapsed = time.perf_counter() - t0

    arr   = np.array(scores)
    thr   = arr.mean() + 1.5 * arr.std()
    sel, last = [], -2.0
    for ts, sc in zip(times, scores):
        if sc > thr and (ts - last) >= 2.0:
            sel.append(ts); last = ts
    return sel, elapsed, pixel_ops


# ---------------------------------------------------------------------------
# PROPOSED  -  PhysioSync-AR
# ---------------------------------------------------------------------------

def run_physiosync(video_path, audio_path):
    """Full pipeline: pose landmarks + GCC-PHAT + LSH + adaptive threshold."""
    t0  = time.perf_counter()
    va  = VisualPoseAnalyzer(target_fps=10)
    aa  = AudioSpatialAnalyzer()
    fe  = FusionEngine()
    ks  = KeyframeSelector()

    v_ts, v_res = va.analyze_video(video_path)
    a_ts, a_res = aa.analyze_audio(audio_path)
    aligned     = fe.align_streams(a_ts, a_res, v_ts, v_res)
    hashes      = fe.compute_sais(aligned)
    kfs         = ks.select_keyframes(hashes)

    elapsed      = time.perf_counter() - t0
    landmark_ops = len(v_ts) * 6 * 3       # 6 joints x (x, y, z)
    return [kf["timestamp"] for kf in kfs], elapsed, landmark_ops, len(v_ts)


# ---------------------------------------------------------------------------
# GROUND TRUTH  -  Audio RMS energy peaks
# ---------------------------------------------------------------------------

def audio_gt(audio_path, video_path, k_std=1.0, min_gap=2.0, tol=0.5):
    """
    Derive ground-truth keyframe events from the audio signal.
    Both methods are evaluated against the same audio-energy peaks so
    neither method has an unfair advantage.
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    win   = int(sr * 0.04)
    rms   = librosa.feature.rms(y=y, frame_length=win, hop_length=win)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=win)

    thr   = rms.mean() + k_std * rms.std()
    evts, last = [], -min_gap
    for t, r in zip(times, rms):
        if r > thr and (t - last) >= min_gap:
            evts.append(float(t)); last = t

    fps, n, _, _ = get_video_info(video_path)
    duration = n / fps
    buckets  = np.arange(0, duration, 0.1)
    gt       = np.zeros(len(buckets), dtype=int)
    for ev in evts:
        gt[np.abs(buckets - ev) <= tol] = 1
    return buckets, gt, evts


def to_bin(timestamps, buckets, tol=0.5):
    lbl = np.zeros(len(buckets), dtype=int)
    for ts in timestamps:
        lbl[np.abs(buckets - ts) <= tol] = 1
    return lbl


def metrics(gt, pred):
    if pred.sum() == 0:
        return 0.0, 0.0, 0.0
    return (precision_score(gt, pred, zero_division=0),
            recall_score(gt, pred, zero_division=0),
            f1_score(gt, pred, zero_division=0))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*65)
    print("  PhysioSync-AR x Dense Optical Flow  -  Benchmark Suite")
    print("="*65)

    for f in (VIDEO_PATH, AUDIO_PATH):
        if not os.path.exists(f):
            print(f"\n[ERROR] {f} not found.\n"); return

    fps, n_frames, w, h = get_video_info(VIDEO_PATH)
    duration = n_frames / fps
    print(f"\n  Video : {VIDEO_PATH}  "
          f"({n_frames} frames @ {fps:.0f} FPS, {duration:.1f} s, {w}x{h})")
    print(f"  Audio : {AUDIO_PATH}")

    # ── Derive ground truth ────────────────────────────────────────────────
    print("\n[1/3] Deriving ground-truth events from audio energy ...")
    buckets, gt, gt_evts = audio_gt(AUDIO_PATH, VIDEO_PATH)
    print(f"       {len(gt_evts)} GT event(s) at: "
          f"{', '.join(f'{e:.2f}s' for e in gt_evts) or 'none'}")

    # ── Run dense optical flow ─────────────────────────────────────────────
    print("[2/3] Running Dense Optical Flow ...")
    of_ts, of_elapsed, of_ops = run_dense_of(VIDEO_PATH)
    of_fps = n_frames / of_elapsed

    # ── Run PhysioSync-AR ─────────────────────────────────────────────────
    print("[3/3] Running PhysioSync-AR ...")
    ps_ts, ps_elapsed, ps_ops, ps_sampled = run_physiosync(VIDEO_PATH, AUDIO_PATH)
    ps_fps  = ps_sampled / ps_elapsed
    speedup = of_elapsed / ps_elapsed

    # ── Accuracy ──────────────────────────────────────────────────────────
    of_p, of_r, of_f1 = metrics(gt, to_bin(of_ts, buckets))
    ps_p, ps_r, ps_f1 = metrics(gt, to_bin(ps_ts, buckets))

    # ── Compute savings ───────────────────────────────────────────────────
    of_total_ops = n_frames * h * w * 2        # (u,v) per pixel per frame
    ps_total_ops = ps_ops                      # 6 joints x 3 per sample
    saving_pct   = (1 - ps_total_ops / of_total_ops) * 100

    # ── Print ─────────────────────────────────────────────────────────────
    sep("THROUGHPUT")
    pr("Dense Optical Flow",                  f"{of_fps:.1f} FPS   ({of_elapsed:.1f} s)")
    pr("PhysioSync-AR",                       f"{ps_fps:.1f} FPS   ({ps_elapsed:.1f} s)")
    pr("Wall-clock speedup",                  f"{speedup:.2f}x")

    sep("COMPUTE SAVINGS  (vs dense optical flow)")
    pr("Dense OF op-count proxy",             f"{of_total_ops:,}  ({h}x{w} px x2 x{n_frames} frames)")
    pr("PhysioSync-AR op-count proxy",        f"{ps_total_ops:,}  (6 joints x3 x{ps_sampled} samples)")
    pr("Absolute reduction",                  f"{of_total_ops - ps_total_ops:,}")
    pr("Compute saving",                      f"{saving_pct:.6f} %")
    pr("Op-count ratio  (OF / PhysioSync)",   f"{of_total_ops / max(ps_total_ops,1):,.0f}x fewer")

    sep("KEYFRAME ACCURACY  (audio-energy ground truth)")
    print(f"  Ground-truth events : {', '.join(f'{e:.2f}s' for e in gt_evts) or 'none'}\n")
    print(f"  {'Method':<26} {'Precision':>10} {'Recall':>10} {'F1':>8}  Keyframes selected")
    print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*8}  ------------------")
    of_str = ", ".join(f"{t:.2f}s" for t in of_ts) or "none"
    ps_str = ", ".join(f"{t:.2f}s" for t in ps_ts) or "none"
    print(f"  {'Dense Optical Flow':<26} {of_p:>10.3f} {of_r:>10.3f} {of_f1:>8.3f}  {of_str}")
    print(f"  {'PhysioSync-AR':<26} {ps_p:>10.3f} {ps_r:>10.3f} {ps_f1:>8.3f}  {ps_str}")

    sep("SUMMARY  (on this machine)")
    print(f"""
  Throughput :  PhysioSync-AR {ps_fps:.0f} FPS  vs  Dense OF {of_fps:.0f} FPS  ({speedup:.1f}x faster)
  Compute    :  {saving_pct:.4f}% fewer operations  ({of_total_ops/max(ps_total_ops,1):,.0f}x op-count ratio)
  F1         :  PhysioSync-AR {ps_f1:.3f}  vs  Dense OF {of_f1:.3f}
""")

    os.makedirs("output", exist_ok=True)
    out = {
        "video": VIDEO_PATH, "audio": AUDIO_PATH,
        "frames": n_frames, "fps": fps, "resolution": f"{w}x{h}",
        "ground_truth": {"method": "audio-RMS peaks", "events_s": [round(e,3) for e in gt_evts]},
        "dense_optical_flow": {
            "wall_time_s": round(of_elapsed,3), "analysis_fps": round(of_fps,1),
            "op_proxy": of_total_ops,
            "keyframes": [round(t,3) for t in of_ts],
            "precision": round(of_p,4), "recall": round(of_r,4), "f1": round(of_f1,4)},
        "physiosync_ar": {
            "wall_time_s": round(ps_elapsed,3), "analysis_fps": round(ps_fps,1),
            "op_proxy": ps_total_ops,
            "keyframes": [round(t,3) for t in ps_ts],
            "precision": round(ps_p,4), "recall": round(ps_r,4), "f1": round(ps_f1,4)},
        "comparison": {
            "speedup_x": round(speedup,2),
            "op_ratio_x": round(of_total_ops/max(ps_total_ops,1),0),
            "compute_saving_pct": round(saving_pct,6)}
    }
    with open("output/benchmark_results.json","w") as f:
        json.dump(out, f, indent=4)
    print("  Results saved -> output/benchmark_results.json")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()

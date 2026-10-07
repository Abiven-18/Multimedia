# PhysioSync-AR — Implementation Plan

**Zero-Shot Dynamic Spatial Audio-Visual Keyframe Hashing for Real-Time, Privacy-Preserving AR Video Summarization**
BITE314L Multimedia Systems, Fall 2026–27 | Faculty Guide: Dr. Balasubramani M
Team: Abishek (23BIT0336), LS Sri Aditya (23BIT0024), Raghunandeeswar S (23BIT0328)

---

## 1. Goal of This Stage

Turn the problem statement and literature gap into a working, testable pipeline. The deliverable at the end of implementation is a CPU-only Python system that ingests a stereo AR/lecture video, computes a **Spatial-Acoustic Importance Score (SAIS)** per frame window via **Cross-Modal Spatial Attention Hashing**, and outputs a compact video summary (keyframes / trimmed clip) plus the 64-bit hash log used for retrieval and privacy-preserving storage.

---

## 2. System Architecture

```
Stereo AR Video (.mp4)
        │
        ├──► Audio Branch ─────────────┐
        │     • Stereo demux           │
        │     • ITD / phase-diff calc  │
        │     • Directional energy     │
        │                              ▼
        │                     ┌─────────────────────┐
        ├──► Visual Branch ──►│  Cross-Modal Spatial  │──► SAIS (per window)
        │     • Frame sample  │  Attention Hashing    │
        │     • MediaPipe pose│  (fusion + 64-bit hash)│
        │     • Joint-velocity└─────────────────────┘
        │       vectors                │
        │                              ▼
        │                     Dynamic Thresholding
        │                              │
        │                              ▼
        │                     Keyframe / Clip Selector
        │                              │
        │                              ▼
        └──────────────────►  Summary Video + Hash DB
                              (privacy-preserving, no
                               raw identifiable frames
                               stored beyond summary)
```

**Five modules**, each independently testable:

1. **Audio Spatial Analyzer** — Inter-channel Time/Phase Difference (ITD) extraction.
2. **Visual Pose Analyzer** — Sparse skeletal landmark extraction and motion-vector computation.
3. **Cross-Modal Fusion & Hashing Engine** — Combines both streams into the 64-bit SAIS hash.
4. **Dynamic Keyframe Selector** — Adaptive thresholding over the SAIS time series.
5. **Summarization & Storage Layer** — Assembles output video/clip set and the hash log; enforces the privacy constraint (no raw frame storage for non-selected content).

---

## 3. Tech Stack

| Layer | Tool / Library | Reason |
|---|---|---|
| Language | Python 3.11 | Fast prototyping, rich audio/vision ecosystem |
| Video I/O | OpenCV (`cv2`) | Frame decode, colour histogram fallback checks |
| Audio I/O & DSP | `librosa`, `scipy.signal` | Stereo channel extraction, cross-correlation for ITD |
| Pose Estimation | MediaPipe Pose (BlazePose, CPU) | Sparse 3D landmarks, real-time on CPU, no GPU dependency |
| Hashing | NumPy bit-packing + custom LSH-style projection | Keeps fusion lightweight vs. Transformer-based CTCH |
| Evaluation | `pyAudioAnalysis`, scikit-learn (metrics) | Benchmarking against literature baselines |
| Serving demo | Streamlit or Flask | Matches prior-art benchmark study's deployment style |
| Packaging | `uv` / `pip` + `requirements.txt` | Reproducibility |

No GPU, no deep-learning training loop — this keeps the "zero-shot / real-time / CPU-only" claim in the title literally true.

---

## 4. Module-Wise Implementation Detail

### 4.1 Audio Spatial Analyzer (ITD extraction)
- Demux left/right channels at native sample rate (resample to 16 kHz for speed).
- Slide a short window (e.g., 20–40 ms) across the signal; compute **Generalized Cross-Correlation with Phase Transform (GCC-PHAT)** between L/R channels to estimate the inter-channel time delay → maps to azimuth/direction of the dominant sound source.
- Output per-window: `(direction_estimate, directional_energy, onset_flag)`.
- This directly implements the "directional audio as filtering signal" claim distinguishing the project from US10791412B2 (which visualizes spatial audio rather than using it as a filter).

### 4.2 Visual Pose Analyzer
- Sample frames at a reduced rate (e.g., 5–10 fps for analysis, full-rate for output splicing).
- Run MediaPipe Pose to get ~33 sparse 3D landmarks per sampled frame.
- Compute joint-velocity vectors between consecutive sampled frames (Δposition / Δt) for key joints (wrists, elbows, shoulders, head).
- Aggregate into a single scalar **gesture-velocity magnitude** per window, plus the raw landmark-velocity vector for hashing.
- This replaces dense optical flow (cf. US9881640B2 / KR102145892B1), which is the ~80% compute reduction claim — worth actually benchmarking (Section 6).

### 4.3 Cross-Modal Fusion & Hashing Engine — the novel core
- Normalize both streams' window outputs to a common time base (align audio windows to visual sample timestamps).
- Concatenate: `[direction_estimate, directional_energy, gesture_velocity_vector, landmark_velocity_vector]` into a fixed-length feature vector per window.
- Project through a small fixed (not trained) random hyperplane projection (LSH-style) to produce a 64-bit binary hash — this is the "Spatial-Acoustic Importance Score" hash.
- **Coupling condition** (the patentable differentiator from Intel's US11288506B2, which only checks gesture velocity): a window is only flagged as high-importance if gesture-velocity crosses its adaptive threshold **and** directional audio energy/onset aligns within a small time tolerance (e.g., ±150 ms). This is the actual "Cross-Modal Spatial Attention" logic — implement it as an explicit AND/weighted-OR gate, not just a summed score, so it's demonstrably distinct from simple feature concatenation.
- Store the hash + timestamp; raw pose/audio feature vectors can be discarded after hashing (privacy-preserving: only hashes + selected summary frames persist).

### 4.4 Dynamic Keyframe Selector
- Compute a rolling mean/std of the SAIS-gate output over the video.
- Use adaptive (not fixed-value) thresholding: a window is a keyframe if its score exceeds `mean + k·std` in a local sliding context (handles both "noisy energetic" and "quiet static" videos without hand-tuned constants).
- Enforce a minimum gap between selected keyframes to avoid redundant near-duplicate selections; enforce a maximum gap so long silent/static stretches still get at least a periodic keyframe.

### 4.5 Summarization & Storage Layer
- Assemble either (a) a keyframe image set with timestamps, or (b) a trimmed video (short clip around each keyframe, e.g., ±1–2 s).
- Persist: summary video/images, SAIS hash log (CSV/JSON), and NOT the full raw pose/audio feature stream for discarded segments — this operationalizes the "privacy-preserving" claim for the patent write-up.

---

## 5. Dataset & Experimental Plan

| Need | Plan |
|---|---|
| Stereo AR/lecture footage | Record 5–10 short (3–5 min) pilot clips with a stereo mic + phone/webcam: lecture-style (speaker moves, occasional off-frame questions) and casual AR-style (walking, turning) scenarios |
| Baselines to compare against | (1) Global colour-histogram keyframing (OpenCV), (2) RMS-volume-spike audio summarization, (3) pose-only (no audio gate) summarization |
| Ground truth | Manual annotation of "important" moments (speaker change, gesture + speech, off-frame event) by the team, used as the reference set |
| Metrics | Precision/Recall/F1 of selected keyframes vs. annotated important moments; compression ratio (output length / input length); latency (ms/frame) and FPS on a standard CPU laptop to validate the "30+ FPS" claim |

---

## 6. Milestones & Timeline (suggested 8-week build)

| Week | Milestone |
|---|---|
| 1 | Environment setup, dataset collection (pilot clips), baseline implementations (colour-histogram, RMS) |
| 2 | Audio Spatial Analyzer: GCC-PHAT ITD extraction + validation on known-direction test recordings |
| 3 | Visual Pose Analyzer: MediaPipe integration, joint-velocity computation |
| 4 | Cross-Modal Fusion & Hashing Engine: feature alignment, 64-bit hash projection, coupling-gate logic |
| 5 | Dynamic Keyframe Selector + Summarization/Storage layer; end-to-end pipeline integration |
| 6 | Benchmarking vs. baselines (Section 5 metrics), FPS/latency profiling, ablations (audio-only, pose-only, fused) |
| 7 | Streamlit/Flask demo UI; privacy-claim validation (confirm no raw frame retention for discarded segments) |
| 8 | Report + patent-drafting notes (novelty table vs. Section B patents), conference-paper draft, final presentation |

---

## 7. Suggested Team Split

- **Abishek** — Audio Spatial Analyzer (GCC-PHAT/ITD) + evaluation metrics harness.
- **LS Sri Aditya** — Visual Pose Analyzer (MediaPipe integration, joint-velocity) + baseline implementations for comparison.
- **Raghunandeeswar S** — Cross-Modal Fusion & Hashing Engine (the core novel module) + Keyframe Selector + Streamlit demo.

All three collaborate on integration (Week 5), benchmarking (Week 6), and the report/patent draft (Week 8).

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GCC-PHAT direction estimates noisy with a single stereo mic pair (no true mic array) | Use short, clean pilot recordings first; consider a phone with two rear mics or a simple 2-mic USB array for validation |
| MediaPipe pose loses tracking on partial/occluded AR-style footage | Fall back to last-known-good landmark + confidence gating; exclude low-confidence windows from the hash rather than injecting noise |
| Manual ground-truth annotation is subjective | Have all three team members annotate independently and report inter-annotator agreement, not just a single label set |
| "80% compute reduction vs. dense optical flow" is a specific, checkable claim | Actually benchmark sparse pose vs. Farneback/dense optical flow on the same clips before quoting a number in the report/patent |
| Novelty overlap with reviewed patents | Keep the AND/weighted-OR cross-modal coupling gate as the explicit, documented differentiator (Section 4.3) — this is what a patent examiner or reviewer will look for |

---

## 9. Deliverables at End of This Stage

1. Working end-to-end Python pipeline (modules 1–5) runnable on a sample video.
2. Benchmark report: precision/recall/F1 vs. baselines, FPS/latency numbers.
3. Hash log format specification (for the privacy-preserving storage claim).
4. Updated novelty table (Section B style) showing exactly how the built system differs from each cited patent, backed by actual measured results rather than projected claims.
5. Draft demo (Streamlit/Flask) for the presentation.

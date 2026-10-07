# PhysioSync-AR: Complete Project Documentation & Presentation Guide

## Project Overview
**Title:** Zero-Shot Dynamic Spatial Audio-Visual Keyframe Hashing for Real-Time, Privacy-Preserving AR Video Summarization (Codename: PhysioSync-AR).

PhysioSync-AR is a novel video summarization pipeline designed specifically for Augmented Reality (AR) and wearable camera streams. It moves away from computationally heavy deep-learning models (like Transformers) and single-modality heuristics (like color histograms), instead introducing a lightweight, CPU-friendly method that fuses human movement with directional sound.

---

## How Data is Processed (The Pipeline)

The core innovation of the project is its **Cross-Modal Spatial Attention Hashing**. The pipeline operates in four distinct stages:

### 1. Visual Pose Extraction (MediaPipe)
- **Input:** Video frames.
- **Process:** Instead of dense optical flow or full-frame CNNs, the system uses MediaPipe to extract sparse 3D skeletal landmarks. It focuses only on key joints: shoulders, elbows, and wrists.
- **Output:** It calculates a "gesture magnitude" representing the velocity of these physical movements across frames.

### 2. Audio Spatial Extraction (GCC-PHAT)
- **Input:** Stereo audio channels.
- **Process:** Instead of just measuring loudness (RMS volume), the system uses Generalized Cross-Correlation with Phase Transform (GCC-PHAT) to calculate the Interaural Time Difference (ITD). 
- **Output:** This calculates the spatial direction of the sound (localization) and directional energy.

### 3. Cross-Modal Fusion & Hashing
- **Process:** The system aligns the visual timestamps with the audio timestamps. It fuses the physical gesture magnitude and the spatial audio phase shifts into a single metric.
- **Output:** Generates a unified **64-bit Spatial-Acoustic Importance Score (SAIS)** for each time window.

### 4. Keyframe Selection
- **Process:** The system applies dynamic thresholding over the 64-bit hashes to pinpoint the exact moments where an acoustically significant event and a physically significant gesture occur simultaneously.

---

## Outputs Generated

When the pipeline runs on real media, it generates three outputs in the `output/` directory:

1. **`output/keyframes/` (Directory):** Contains `.jpg` images of the exact video frames the algorithm deemed most important. This serves as the visual "summary" of the video.
2. **`hash_log.json`:** A log of the 64-bit binary hashes mapped to their timestamps. This is the **privacy-preserving** output; in a real AR edge deployment, only these hashes are saved or transmitted—no raw, identifiable video data is stored.
3. **`summary_timestamps.txt`:** A quick, human-readable list of the exact seconds the keyframes were triggered.

---

## Presentation Strategy: What to Tell the Professor (Dr. Balasubramani M)

When presenting this project, follow this narrative to highlight its academic novelty and patent potential:

### 1. Start with the Problem (The Gap)
*   Explain that traditional summarizers rely on *scene changes* (color histograms) or *heavy deep learning* (Transformers). 
*   Point out that in AR, the background rarely changes, but the *interaction* (where sound comes from + human gestures) is what matters. Existing models are too heavy for edge deployment.

### 2. Demonstrate the Pipeline (The Solution)
*   Show the code running live (or the terminal logs).
*   **Key Talking Point:** Emphasize the speed (e.g., processing a video in ~60 seconds). Point out that this is running entirely on a standard CPU without GPUs, proving it is a lightweight, edge-ready architecture.

### 3. Highlight Privacy-Preservation
*   Open the `hash_log.json` file.
*   **Key Talking Point:** Show the binary strings and explain, *"Because this is for AR wearables, privacy is critical. Our system doesn't store raw video; it generates these compact 64-bit hashes based on skeletal and audio math. This prevents sensitive imagery from being saved unnecessarily."*

### 4. Prove Cross-Modal Summarization Works
*   Open the `output/keyframes/` folder and show the extracted images.
*   **Key Talking Point:** Explain that these frames weren't selected randomly or just because the scene changed. They were selected because a person was physically gesturing *while* a spatial sound event occurred.

### 5. The Pitch for Patentability (IPO Filing)
*   Bring it back to the Literature/Patent survey.
*   **Key Talking Point:** Conclude by stating: *"Unlike prior patents from Tencent or Samsung that use heavy global image vectors or dense optical flow, PhysioSync-AR successfully couples sparse 3D skeletal landmarks with Inter-Channel Time Difference audio. This creates a uniquely efficient, privacy-first summarization tool highly suitable for an IPO patent filing."*

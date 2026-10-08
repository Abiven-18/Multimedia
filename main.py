from audio_analyzer import AudioSpatialAnalyzer
from visual_analyzer import VisualPoseAnalyzer
from fusion_engine import FusionEngine
from keyframe_selector import KeyframeSelector
from summarizer import Summarizer
import time

def main(video_path, audio_path):
    print(f"Starting PhysioSync-AR pipeline for {video_path}")
    start_time = time.time()
    
    # Initialize modules
    audio_analyzer = AudioSpatialAnalyzer()
    visual_analyzer = VisualPoseAnalyzer()
    fusion_engine = FusionEngine()
    keyframe_selector = KeyframeSelector()
    summarizer = Summarizer()
    
    # 1 & 2. Analyze Streams
    print("1. Extracting visual pose features...")
    visual_timestamps, visual_results = visual_analyzer.analyze_video(video_path)
    
    print("2. Extracting audio spatial features...")
    audio_timestamps, audio_results = audio_analyzer.analyze_audio(audio_path)
    
    # 3. Cross-Modal Fusion
    print("3. Fusing streams and computing SAIS hashes...")
    aligned_features = fusion_engine.align_streams(
        audio_timestamps, audio_results, 
        visual_timestamps, visual_results
    )
    sais_hashes = fusion_engine.compute_sais(aligned_features)
    
    # 4. Keyframe Selection
    print("4. Selecting keyframes via dynamic thresholding...")
    selected_keyframes = keyframe_selector.select_keyframes(sais_hashes)
    print(f"   -> Selected {len(selected_keyframes)} keyframes.")
    
    # 5. Summarization & Storage
    print("5. Generating summary and saving hash log (Privacy-Preserving)...")
    hash_log_path = summarizer.save_hash_log(selected_keyframes)
    summary_path = summarizer.assemble_summary(video_path, selected_keyframes)
    
    end_time = time.time()
    print("Pipeline Complete.")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    print(f"Hash log saved to: {hash_log_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    import os
    video = "video.mp4"
    audio = "audio.wav"

    if os.path.exists(video) and os.path.exists(audio):
        main(video, audio)
    else:
        print(f"Input files not found. Ensure {video} and {audio} are in the project directory.")

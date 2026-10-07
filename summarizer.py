import json
import csv
import os

class Summarizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_hash_log(self, keyframes, filename="hash_log.json"):
        """
        Save the 64-bit hash logs and timestamps to a JSON file.
        Enforces privacy by ONLY saving hashes and timestamps, not raw features.
        """
        log_path = os.path.join(self.output_dir, filename)
        
        log_data = []
        for kf in keyframes:
            # Convert binary array hash to string for JSON serialization
            hash_str = "".join(map(str, kf['hash']))
            log_data.append({
                'timestamp': kf['timestamp'],
                'hash_64bit': hash_str
            })
            
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
            
        return log_path

    def assemble_summary(self, video_path, keyframes, output_filename="summary.mp4"):
        """
        Extract the actual keyframes as images from the video and save them.
        """
        import cv2
        # Save the summary timestamps
        summary_txt = os.path.join(self.output_dir, "summary_timestamps.txt")
        with open(summary_txt, 'w') as f:
            f.write(f"Source video: {video_path}\n")
            f.write("Selected Keyframe Timestamps:\n")
            for kf in keyframes:
                f.write(f"- {kf['timestamp']:.2f}s\n")
        
        # Extract images
        keyframes_dir = os.path.join(self.output_dir, "keyframes")
        if not os.path.exists(keyframes_dir):
            os.makedirs(keyframes_dir)
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
            
        for idx, kf in enumerate(keyframes):
            ts = kf['timestamp']
            frame_number = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                img_path = os.path.join(keyframes_dir, f"keyframe_{idx:03d}_{ts:.2f}s.jpg")
                cv2.imwrite(img_path, frame)
                
        cap.release()
        return summary_txt

if __name__ == "__main__":
    summarizer = Summarizer()
    mock_kfs = [{'timestamp': 1.0, 'hash': [1]*64}, {'timestamp': 5.0, 'hash': [0]*64}]
    summarizer.save_hash_log(mock_kfs)
    print("Mock hash log saved.")

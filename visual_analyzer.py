import cv2
import mediapipe as mp
import numpy as np

class VisualPoseAnalyzer:
    def __init__(self, target_fps=10):
        self.target_fps = target_fps
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        # Key joints: wrists, elbows, shoulders (indices in MediaPipe Pose)
        self.key_joints = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value
        ]

    def process_frame(self, frame_rgb, prev_landmarks=None, dt=0.1):
        """
        Process a single RGB frame to extract pose and compute velocity.
        """
        results = self.pose.process(frame_rgb)
        
        current_landmarks = None
        velocity_vector = np.zeros((len(self.key_joints), 3)) # x, y, z for each key joint
        gesture_magnitude = 0.0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract key joints
            current_landmarks = np.array([
                [landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in self.key_joints
            ])
            
            if prev_landmarks is not None:
                # Calculate velocity (Δposition / Δt)
                velocity_vector = (current_landmarks - prev_landmarks) / dt
                
                # Aggregate into a single scalar gesture-velocity magnitude
                gesture_magnitude = np.sum(np.linalg.norm(velocity_vector, axis=1))
                
        return current_landmarks, velocity_vector.flatten(), gesture_magnitude

    def analyze_video(self, video_path):
        """
        Load video, sample frames, return list of window features.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30 # fallback
            
        frame_interval = max(1, int(fps / self.target_fps))
        dt = 1.0 / self.target_fps
        
        timestamps = []
        results = [] # list of (gesture_magnitude, velocity_vector)
        
        frame_count = 0
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = frame_count / fps
                
                current_landmarks, velocity_vector, gesture_magnitude = self.process_frame(frame_rgb, prev_landmarks, dt)
                
                timestamps.append(ts)
                results.append((gesture_magnitude, velocity_vector))
                
                prev_landmarks = current_landmarks
                
            frame_count += 1
            
        cap.release()
        return timestamps, results

if __name__ == "__main__":
    # Mock testing
    analyzer = VisualPoseAnalyzer()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    curr, vel, mag = analyzer.process_frame(mock_frame)
    print(f"Mock frame result: Gesture Mag={mag}")

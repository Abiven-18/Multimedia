import cv2
import numpy as np
import scipy.io.wavfile as wav
import os

def create_mock_video(filename="mock_video.mp4", fps=30, duration_sec=10):
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    num_frames = fps * duration_sec
    for i in range(num_frames):
        # Create a simple moving object to simulate some "gesture"
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circle
        x = int(width/2 + 100 * np.sin(2 * np.pi * i / fps))
        y = int(height/2 + 50 * np.cos(2 * np.pi * i / fps))
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        
        # Add some occasional "high activity" (e.g. noise)
        if 3 * fps < i < 4 * fps: # Second 3 to 4
            noise = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.5, noise, 0.5, 0)
            
        out.write(frame)
        
    out.release()
    print(f"Mock video created: {filename}")

def create_mock_audio(filename="mock_audio.wav", sr=16000, duration_sec=10):
    t = np.linspace(0, duration_sec, sr * duration_sec)
    
    # Base tone
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 440 * t)
    
    # Add a loud "event" at second 3 to 4 with a phase shift (direction change)
    event_start = 3 * sr
    event_end = 4 * sr
    
    event_sig = 2.0 * np.sin(2 * np.pi * 880 * t[event_start:event_end])
    left[event_start:event_end] += event_sig
    # Delay right channel by 10 samples to simulate ITD
    right_event = np.roll(event_sig, 10)
    right[event_start:event_end] += right_event
    
    # Normalize
    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
    left = left / max_val
    right = right / max_val
    
    # Save as 16-bit PCM stereo
    stereo = np.vstack((left, right)).T
    stereo_int16 = np.int16(stereo * 32767)
    
    wav.write(filename, sr, stereo_int16)
    print(f"Mock audio created: {filename}")

if __name__ == "__main__":
    print("Generating mock data...")
    create_mock_video()
    create_mock_audio()
    print("Done.")

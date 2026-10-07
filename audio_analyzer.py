import numpy as np
import librosa
from scipy.signal import correlate

class AudioSpatialAnalyzer:
    def __init__(self, target_sr=16000, window_ms=40):
        self.target_sr = target_sr
        self.window_size = int(self.target_sr * (window_ms / 1000.0))

    def gcc_phat(self, sig1, sig2):
        """
        Compute Generalized Cross-Correlation with Phase Transform.
        """
        # FFT of both signals
        X1 = np.fft.rfft(sig1)
        X2 = np.fft.rfft(sig2)
        
        # Cross-power spectrum
        cross_power = X1 * np.conj(X2)
        
        # Phase transform (normalize by magnitude)
        eps = 1e-15
        phat = cross_power / (np.abs(cross_power) + eps)
        
        # Inverse FFT to get correlation function
        cc = np.fft.irfft(phat)
        
        # Shift the peak to the center
        cc = np.fft.fftshift(cc)
        
        # Find the peak which corresponds to the time delay
        shift = np.argmax(cc) - len(cc) // 2
        return shift

    def process_window(self, left_window, right_window):
        """
        Process a single stereo window.
        Returns (direction_estimate, directional_energy, onset_flag)
        """
        # Estimate direction via ITD (using GCC-PHAT)
        direction_estimate = self.gcc_phat(left_window, right_window)
        
        # Directional energy (simple RMS of the window for now)
        energy = np.sqrt(np.mean(left_window**2) + np.mean(right_window**2))
        
        # Onset flag could be dynamically determined; keeping false as default and will handle outside or with a threshold
        onset_flag = False 
        
        return direction_estimate, energy, onset_flag

    def analyze_audio(self, audio_path):
        """
        Load audio, process in windows, return list of window features.
        """
        # Load stereo audio
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=False)
        
        if y.ndim == 1:
            # If mono, duplicate to make it "stereo" for testing
            y = np.vstack([y, y])
            
        left_channel = y[0]
        right_channel = y[1]
        
        results = []
        timestamps = []
        
        num_windows = len(left_channel) // self.window_size
        
        for i in range(num_windows):
            start = i * self.window_size
            end = start + self.window_size
            
            l_win = left_channel[start:end]
            r_win = right_channel[start:end]
            
            direction, energy, onset = self.process_window(l_win, r_win)
            results.append((direction, energy, onset))
            
            # Timestamp at the middle of the window
            ts = (start + self.window_size / 2) / self.target_sr
            timestamps.append(ts)
            
        return timestamps, results

if __name__ == "__main__":
    # Mock testing
    analyzer = AudioSpatialAnalyzer()
    # Mock stereo signal
    t = np.linspace(0, 1, 16000)
    l = np.sin(2 * np.pi * 440 * t)
    r = np.roll(l, 5) # delay by 5 samples
    direction, energy, onset = analyzer.process_window(l[:640], r[:640])
    print(f"Mock window result: Direction Shift={direction}, Energy={energy}")

import numpy as np

class KeyframeSelector:
    def __init__(self, k_std=1.5, min_gap_sec=2.0, max_gap_sec=10.0):
        self.k_std = k_std
        self.min_gap_sec = min_gap_sec
        self.max_gap_sec = max_gap_sec

    def select_keyframes(self, sais_hashes):
        """
        Select keyframes using adaptive thresholding.
        sais_hashes is a list of dicts: {'timestamp': float, 'hash': array, 'score': float}
        """
        if not sais_hashes:
            return []
            
        scores = [item['score'] for item in sais_hashes]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        threshold = mean_score + self.k_std * std_score
        
        selected_keyframes = []
        last_selected_ts = -self.min_gap_sec
        
        for item in sais_hashes:
            ts = item['timestamp']
            score = item['score']
            
            # Condition 1: High importance score, and minimum gap respected
            if score > threshold and (ts - last_selected_ts) >= self.min_gap_sec:
                selected_keyframes.append(item)
                last_selected_ts = ts
            # Condition 2: Maximum gap exceeded, force a keyframe
            elif (ts - last_selected_ts) >= self.max_gap_sec:
                selected_keyframes.append(item)
                last_selected_ts = ts
                
        return selected_keyframes

if __name__ == "__main__":
    selector = KeyframeSelector()
    # Mock data
    mock_data = [{'timestamp': i, 'hash': [], 'score': np.random.rand()} for i in range(20)]
    kf = selector.select_keyframes(mock_data)
    print(f"Selected {len(kf)} keyframes from 20 mock inputs.")

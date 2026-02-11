import numpy as np

class InferenceTimer:  
    def __init__(self, name=""):
        self.name = name
        self.times = []
        self.total_frames = 0
        self.total_time = 0
        
    def add_time(self, inference_ms): #inference time in milliseconds
        self.times.append(inference_ms)
        self.total_frames += 1
        self.total_time += inference_ms
    
    def get_statistics(self):
        """Get complete statistics"""
        if len(self.times) == 0:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'p95': 0,
                'p99': 0,
                'max_fps': 0
            }
        
        times_array = np.array(self.times)
        mean_time = np.mean(times_array)
        
        return {
            'count': len(self.times),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'mean': mean_time,
            'median': np.median(times_array),
            'std': np.std(times_array),
            'p95': np.percentile(times_array, 95),
            'p99': np.percentile(times_array, 99),
            'max_fps': 1000 / mean_time if mean_time > 0 else 0
        }
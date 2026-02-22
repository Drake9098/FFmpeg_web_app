"""
Video Quality Metrics Module
Calculates SSIM and PSNR between original and converted videos
"""

import cv2
import numpy as np

class VideoQualityAnalyzer:
    """Analyzes quality differences between original and converted videos"""
    
    def __init__(self):
        pass
    
    def extract_frames(self, video_path: str, num_frames: int = 10) -> list:
        """
        Extract evenly distributed frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract (evenly distributed)
        if total_frames < num_frames:
            frame_indices = range(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
        
        cap.release()
        
        return frames
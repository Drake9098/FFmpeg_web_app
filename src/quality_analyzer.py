"""
Video Quality Metrics Module
Calculates SSIM and PSNR between original and converted videos
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict

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
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            PSNR value in dB (higher is better, typically 30-50 dB is good)
        """
        # Ensure images have the same dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        if mse == 0:
            return float('inf')  # Images are identical
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            SSIM value between -1 and 1 (higher is better)
        """
        # Ensure images have the same dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # SSIM
        ssim_value = ssim(img1_gray, img2_gray)
        
        return ssim_value
    
    def compare_videos(self, original_path: str, converted_path: str, 
                      num_frames: int = 10) -> Dict:
        """
        Compare quality between original and converted videos
        
        Args:
            original_path: Path to original video
            converted_path: Path to converted video
            num_frames: Number of frames to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        print(f"\nAnalyzing video quality...")
        print(f"Extracting {num_frames} frames from each video...")
        
        # Extract frames from both videos
        original_frames = self.extract_frames(original_path, num_frames)
        converted_frames = self.extract_frames(converted_path, num_frames)
        
        # Ensure we have the same number of frames
        min_frames = min(len(original_frames), len(converted_frames))
        original_frames = original_frames[:min_frames]
        converted_frames = converted_frames[:min_frames]
        
        psnr_values = []
        ssim_values = []
        
        print(f"Calculating quality metrics for {min_frames} frames...")
        
        # Calculate metrics for each frame pair
        for i, (orig_frame, conv_frame) in enumerate(zip(original_frames, converted_frames)):
            psnr = self.calculate_psnr(orig_frame, conv_frame)
            ssim_val = self.calculate_ssim(orig_frame, conv_frame)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
            
            print(f"  Frame {i+1}/{min_frames}: PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}")
        
        # Calculate statistics
        results = {
            'num_frames_analyzed': min_frames,
            'psnr_mean': round(np.mean(psnr_values), 2),
            'psnr_min': round(np.min(psnr_values), 2),
            'psnr_max': round(np.max(psnr_values), 2),
            'psnr_std': round(np.std(psnr_values), 2),
            'ssim_mean': round(np.mean(ssim_values), 4),
            'ssim_min': round(np.min(ssim_values), 4),
            'ssim_max': round(np.max(ssim_values), 4),
            'ssim_std': round(np.std(ssim_values), 4),
        }
        
        # Add quality assessment
        if results['ssim_mean'] > 0.95:
            quality_assessment = "Excellent"
        elif results['ssim_mean'] > 0.90:
            quality_assessment = "Very Good"
        elif results['ssim_mean'] > 0.80:
            quality_assessment = "Good"
        elif results['ssim_mean'] > 0.70:
            quality_assessment = "Fine"
        else:
            quality_assessment = "Bad"
        
        results['quality_assessment'] = quality_assessment
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Quality Analysis Results")
        print(f"{'='*60}")
        print(f"Frames analyzed: {results['num_frames_analyzed']}")

        print(f"\nPSNR (Peak Signal-to-Noise Ratio):")
        print(f"  Mean: {results['psnr_mean']} dB")
        print(f"  Range: {results['psnr_min']} - {results['psnr_max']} dB")

        print(f"\nSSIM (Structural Similarity Index):")
        print(f"  Mean: {results['ssim_mean']}")
        print(f"  Range: {results['ssim_min']} - {results['ssim_max']}")
        
        print(f"\nOverall Quality: {results['quality_assessment']}")
        print(f"{'='*60}\n")
        
        return results


if __name__ == "__main__":
    # Example usage
    analyzer = VideoQualityAnalyzer()
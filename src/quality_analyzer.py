"""
Video Quality Metrics Module
Calculates SSIM and PSNR between original and converted videos
"""

from typing import Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class VideoQualityAnalyzer:
    """Analyzes quality differences between original and converted videos"""

    def __init__(self):
        pass

    # extract_frames helpers

    def _open_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """Open a video file and return the capture object, raising on failure."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        return cap

    def _calculate_frame_indices(self, total_frames: int, num_frames: int):
        """Return evenly-distributed frame indices within [0, total_frames)."""
        if total_frames < num_frames:
            return range(total_frames)
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    def _read_frames_at_indices(self, cap: cv2.VideoCapture, frame_indices) -> list:
        """Seek to each index and collect successfully decoded frames."""
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        return frames

    def extract_frames(self, video_path: str, num_frames: int = 10) -> list:
        """
        Extract evenly distributed frames from a video file.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to extract.

        Returns:
            List of frames as numpy arrays.
        """
        cap = self._open_video_capture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._calculate_frame_indices(total_frames, num_frames)
        frames = self._read_frames_at_indices(cap, frame_indices)
        cap.release()
        return frames

    # calculate psnr helpers

    def _align_dimensions(self, img1: np.ndarray, img2: np.ndarray):
        """Resize img2 to match img1's dimensions if they differ."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2

    def _calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Return the Mean Squared Error between two images."""
        return float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))

    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            PSNR value in dB (higher is better; typically 30-50 dB is good).
        """
        img1, img2 = self._align_dimensions(img1, img2)
        mse = self._calculate_mse(img1, img2)
        if mse == 0:
            return float("inf")  # Images are identical
        max_pixel = 255.0
        return float(20 * np.log10(max_pixel / np.sqrt(mse)))

    # calculate ssim

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale; return as-is if already single-channel."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two images.

        Args:
            img1: First image.
            img2: Second image.

        Returns:
            SSIM value between -1 and 1 (higher is better).
        """
        img1, img2 = self._align_dimensions(img1, img2)
        return float(ssim(self._to_grayscale(img1), self._to_grayscale(img2)))

    # compare videos

    def _align_frame_lists(self, frames_a: list, frames_b: list):
        """Truncate both lists to the length of the shorter one."""
        min_len = min(len(frames_a), len(frames_b))
        return frames_a[:min_len], frames_b[:min_len]

    def _calculate_frame_metrics(self, original_frames: list, converted_frames: list):
        """
        Compute per-frame PSNR and SSIM values, printing progress as it goes.

        Returns:
            Tuple of (psnr_values, ssim_values) lists.
        """
        psnr_values, ssim_values = [], []
        total = len(original_frames)
        for i, (orig, conv) in enumerate(zip(original_frames, converted_frames)):
            psnr = self.calculate_psnr(orig, conv)
            ssim_val = self.calculate_ssim(orig, conv)
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
            print(f"  Frame {i+1}/{total}: PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}")
        return psnr_values, ssim_values

    def _build_statistics(self, psnr_values: list, ssim_values: list) -> Dict:
        """Aggregate per-frame metric lists into a statistics dictionary."""
        return {
            "num_frames_analyzed": len(psnr_values),
            "psnr_mean": round(np.mean(psnr_values), 2),
            "psnr_min": round(np.min(psnr_values), 2),
            "psnr_max": round(np.max(psnr_values), 2),
            "psnr_std": round(np.std(psnr_values), 2),
            "ssim_mean": round(np.mean(ssim_values), 4),
            "ssim_min": round(np.min(ssim_values), 4),
            "ssim_max": round(np.max(ssim_values), 4),
            "ssim_std": round(np.std(ssim_values), 4),
        }

    def _assess_quality(self, ssim_mean: float) -> str:
        """Map a mean SSIM value to a human-readable quality label."""
        if ssim_mean > 0.95:
            return "Excellent"
        if ssim_mean > 0.90:
            return "Very Good"
        if ssim_mean > 0.80:
            return "Good"
        if ssim_mean > 0.70:
            return "Fine"
        return "Bad"

    def _print_quality_report(self, results: Dict) -> None:
        """Print a formatted quality-analysis summary to stdout."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("Quality Analysis Results")
        print(sep)
        print(f"Frames analyzed: {results['num_frames_analyzed']}")
        print("\nPSNR (Peak Signal-to-Noise Ratio):")
        print(f"  Mean: {results['psnr_mean']} dB")
        print(f"  Range: {results['psnr_min']} - {results['psnr_max']} dB")
        print("\nSSIM (Structural Similarity Index):")
        print(f"  Mean: {results['ssim_mean']}")
        print(f"  Range: {results['ssim_min']} - {results['ssim_max']}")
        print(f"\nOverall Quality: {results['quality_assessment']}")
        print(f"{sep}\n")

    def compare_videos(
        self, original_path: str, converted_path: str, num_frames: int = 10
    ) -> Dict:
        """
        Compare quality between original and converted videos.

        Args:
            original_path: Path to the original video.
            converted_path: Path to the converted video.
            num_frames: Number of frames to analyze.

        Returns:
            Dictionary with quality metrics.
        """
        print("\nAnalyzing video quality...")
        print(f"Extracting {num_frames} frames from each video...")

        original_frames = self.extract_frames(original_path, num_frames)
        converted_frames = self.extract_frames(converted_path, num_frames)
        original_frames, converted_frames = self._align_frame_lists(
            original_frames, converted_frames
        )

        print(f"Calculating quality metrics for {len(original_frames)} frames...")
        psnr_values, ssim_values = self._calculate_frame_metrics(
            original_frames, converted_frames
        )

        results = self._build_statistics(psnr_values, ssim_values)
        results["quality_assessment"] = self._assess_quality(results["ssim_mean"])

        self._print_quality_report(results)
        return results


if __name__ == "__main__":
    # Example usage
    analyzer = VideoQualityAnalyzer()

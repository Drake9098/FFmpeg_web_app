"""
Tests for VideoQualityAnalyzer class.
calculate_psnr / calculate_ssim use real numpy arrays (no I/O needed).
extract_frames and compare_videos mock cv2.VideoCapture / instance methods.
"""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quality_analyzer import VideoQualityAnalyzer


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_frame(h: int = 64, w: int = 64, value: int = 128, channels: int = 3) -> np.ndarray:
    """Return a solid-colour BGR frame as uint8."""
    return np.full((h, w, channels), value, dtype=np.uint8)


def make_cap_mock(total_frames: int, read_side_effect: list) -> MagicMock:
    """
    Build a MagicMock mimicking cv2.VideoCapture.
    `read_side_effect`: list of (ret, frame) tuples returned by successive read() calls.
    """
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = float(total_frames)
    cap.read.side_effect = read_side_effect
    return cap


# ── TestCalculatePSNR ──────────────────────────────────────────────────────────

class TestCalculatePSNR:
    def setup_method(self):
        self.analyzer = VideoQualityAnalyzer()

    def test_identical_images_returns_inf(self):
        img = make_frame(value=100)
        assert self.analyzer.calculate_psnr(img, img.copy()) == float('inf')

    def test_uniform_images_known_psnr(self):
        # Solid colour: all pixels differ by 50 → MSE=2500 → PSNR≈14.15 dB
        img1 = make_frame(value=100)
        img2 = make_frame(value=150)
        result = self.analyzer.calculate_psnr(img1, img2)
        assert isinstance(result, float)
        assert result == pytest.approx(14.15, abs=0.1)

    def test_higher_difference_gives_lower_psnr(self):
        base = make_frame(value=128)
        small_diff = make_frame(value=130)
        large_diff = make_frame(value=200)
        assert self.analyzer.calculate_psnr(base, small_diff) > \
               self.analyzer.calculate_psnr(base, large_diff)

    def test_different_sizes_are_resized(self):
        # After resize, img2 is still uniform value=150 → same PSNR as same-size case ≈14.15 dB
        img1 = make_frame(h=64, w=64, value=100)
        img2 = make_frame(h=32, w=32, value=150)
        result = self.analyzer.calculate_psnr(img1, img2)
        assert isinstance(result, float)
        assert result == pytest.approx(14.15, abs=0.1)

    def test_psnr_is_positive_for_noisy_image(self):
        rng = np.random.default_rng(0)
        img1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        noise = rng.integers(0, 10, (64, 64, 3), dtype=np.uint8)
        img2 = np.clip(img1.astype(int) + noise, 0, 255).astype(np.uint8)
        assert self.analyzer.calculate_psnr(img1, img2) > 0


# ── TestCalculateSSIM ──────────────────────────────────────────────────────────

class TestCalculateSSIM:
    def setup_method(self):
        self.analyzer = VideoQualityAnalyzer()

    def test_identical_images_returns_one(self):
        img = make_frame(value=128)
        assert self.analyzer.calculate_ssim(img, img.copy()) == pytest.approx(1.0, abs=1e-6)

    def test_returns_float_in_valid_range(self):
        img1 = make_frame(value=50)
        img2 = make_frame(value=200)
        result = self.analyzer.calculate_ssim(img1, img2)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_different_sizes_are_resized(self):
        img1 = make_frame(h=64, w=64, value=100)
        img2 = make_frame(h=32, w=32, value=100)
        result = self.analyzer.calculate_ssim(img1, img2)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_grayscale_input_skips_conversion(self):
        """2-D (grayscale) images should not trigger cvtColor."""
        img1 = np.full((64, 64), 128, dtype=np.uint8)
        img2 = np.full((64, 64), 128, dtype=np.uint8)
        assert self.analyzer.calculate_ssim(img1, img2) == pytest.approx(1.0, abs=1e-6)

    def test_dissimilar_images_score_below_one(self):
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        assert self.analyzer.calculate_ssim(img1, img2) < 1.0


# ── TestExtractFrames ──────────────────────────────────────────────────────────

class TestExtractFrames:
    def setup_method(self):
        self.analyzer = VideoQualityAnalyzer()

    def test_raises_if_video_cannot_open(self):
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch('cv2.VideoCapture', return_value=cap):
            with pytest.raises(ValueError, match="Cannot open video"):
                self.analyzer.extract_frames("nonexistent.mp4")

    def test_returns_correct_number_of_frames(self):
        frame = make_frame()
        cap = make_cap_mock(total_frames=20, read_side_effect=[(True, frame)] * 10)
        with patch('cv2.VideoCapture', return_value=cap):
            result = self.analyzer.extract_frames("fake.mp4", num_frames=10)
        assert len(result) == 10

    def test_fewer_total_frames_than_requested_uses_all(self):
        """When total_frames < num_frames, every available frame is returned."""
        frame = make_frame()
        cap = make_cap_mock(total_frames=3, read_side_effect=[(True, frame)] * 3)
        with patch('cv2.VideoCapture', return_value=cap):
            result = self.analyzer.extract_frames("fake.mp4", num_frames=10)
        assert len(result) == 3

    def test_skips_failed_reads(self):
        """Frames whose ret=False are not appended to the result."""
        frame = make_frame()
        cap = make_cap_mock(total_frames=4, read_side_effect=[
            (True, frame), (False, None), (True, frame), (True, frame),
        ])
        with patch('cv2.VideoCapture', return_value=cap):
            result = self.analyzer.extract_frames("fake.mp4", num_frames=4)
        assert len(result) == 3

    def test_release_is_called(self):
        frame = make_frame()
        cap = make_cap_mock(total_frames=2, read_side_effect=[(True, frame)] * 2)
        with patch('cv2.VideoCapture', return_value=cap):
            self.analyzer.extract_frames("fake.mp4", num_frames=2)
        cap.release.assert_called_once()

    def test_frames_are_numpy_arrays(self):
        frame = make_frame()
        cap = make_cap_mock(total_frames=3, read_side_effect=[(True, frame)] * 3)
        with patch('cv2.VideoCapture', return_value=cap):
            result = self.analyzer.extract_frames("fake.mp4", num_frames=3)
        for f in result:
            assert isinstance(f, np.ndarray)


# ── TestCompareVideos ──────────────────────────────────────────────────────────

class TestCompareVideos:
    def setup_method(self):
        self.analyzer = VideoQualityAnalyzer()

    def _mock_extract(self, frames_orig, frames_conv):
        """Patch extract_frames to return preset frame lists for orig and conv."""
        return patch.object(
            self.analyzer, 'extract_frames',
            side_effect=[frames_orig, frames_conv]
        )

    def _identical_frames(self, n: int = 3) -> list:
        return [make_frame(value=128).copy() for _ in range(n)]

    # ── result structure

    def test_all_expected_keys_present(self):
        frames = self._identical_frames()
        with self._mock_extract(frames, frames):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        for key in [
            'num_frames_analyzed',
            'psnr_mean', 'psnr_min', 'psnr_max', 'psnr_std',
            'ssim_mean', 'ssim_min', 'ssim_max', 'ssim_std',
            'quality_assessment',
        ]:
            assert key in result, f"Missing key: {key}"

    def test_num_frames_analyzed_value(self):
        frames = self._identical_frames(n=5)
        with self._mock_extract(frames, frames):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4", num_frames=5)
        assert result['num_frames_analyzed'] == 5

    # ── metric values for identical videos

    def test_identical_videos_ssim_mean_is_one(self):
        frames = self._identical_frames(n=3)
        with self._mock_extract(frames, frames):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        assert result['ssim_mean'] == pytest.approx(1.0, abs=1e-4)

    def test_identical_videos_psnr_mean_is_inf(self):
        frames = self._identical_frames(n=3)
        with self._mock_extract(frames, frames):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        assert result['psnr_mean'] == float('inf')

    # ── unequal frame counts → uses min

    def test_unequal_frame_lists_uses_minimum(self):
        orig_frames = self._identical_frames(n=5)
        conv_frames = self._identical_frames(n=3)
        with self._mock_extract(orig_frames, conv_frames):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        assert result['num_frames_analyzed'] == 3

    # ── statistics sanity checks

    def test_ssim_min_lte_mean_lte_max(self):
        frames = self._identical_frames(n=4)
        with self._mock_extract(frames, frames), \
             patch.object(self.analyzer, 'calculate_ssim',
                          side_effect=[0.80, 0.90, 0.85, 0.95]), \
             patch.object(self.analyzer, 'calculate_psnr', return_value=35.0):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        assert result['ssim_min'] <= result['ssim_mean'] <= result['ssim_max']

    def test_psnr_min_lte_mean_lte_max(self):
        frames = self._identical_frames(n=3)
        with self._mock_extract(frames, frames), \
             patch.object(self.analyzer, 'calculate_psnr',
                          side_effect=[30.0, 40.0, 35.0]), \
             patch.object(self.analyzer, 'calculate_ssim', return_value=0.95):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        assert result['psnr_min'] <= result['psnr_mean'] <= result['psnr_max']

    # ── quality assessment thresholds (all 5 branches)

    @pytest.mark.parametrize("ssim_val,expected", [
        (0.97, "Excellent"),
        (0.92, "Very Good"),
        (0.85, "Good"),
        (0.75, "Fine"),
        (0.50, "Bad"),
    ])
    def test_quality_assessment_thresholds(self, ssim_val, expected):
        frames = self._identical_frames(n=2)
        with self._mock_extract(frames, frames), \
             patch.object(self.analyzer, 'calculate_ssim', return_value=ssim_val), \
             patch.object(self.analyzer, 'calculate_psnr', return_value=40.0):
            result = self.analyzer.compare_videos("orig.mp4", "conv.mp4")
        assert result['quality_assessment'] == expected

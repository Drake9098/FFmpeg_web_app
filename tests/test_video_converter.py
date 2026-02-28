"""
Tests for VideoConverter class
Uses mocking to avoid requiring real video files or ffmpeg installation.
"""

import sys
import ffmpeg
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.video_converter import VideoConverter


# Shared fake data

FAKE_PROBE = {
    'format': {
        'format_name': 'mov,mp4,m4a,3gp,3g2,mj2',
        'duration': '120.0',
        'size': '10485760',   # 10 MB
        'bit_rate': '699050',
    },
    'streams': [
        {
            'codec_type': 'video',
            'codec_name': 'h264',
            'width': 1920,
            'height': 1080,
            'r_frame_rate': '30/1',
            'bit_rate': '600000',
        },
        {
            'codec_type': 'audio',
            'codec_name': 'aac',
            'bit_rate': '128000',
            'sample_rate': '44100',
        },
    ],
}

FAKE_OUTPUT_PROBE = {
    'format': {
        'format_name': 'mov,mp4,m4a,3gp,3g2,mj2',
        'duration': '120.0',
        'size': '5242880',   # 5 MB (half the input)
        'bit_rate': '349525',
    },
    'streams': [
        {
            'codec_type': 'video',
            'codec_name': 'h264',
            'width': 1920,
            'height': 1080,
            'r_frame_rate': '30/1',
            'bit_rate': '300000',
        },
        {
            'codec_type': 'audio',
            'codec_name': 'aac',
            'bit_rate': '128000',
            'sample_rate': '44100',
        },
    ],
}

# Fixtures

@pytest.fixture
def converter(tmp_path):
    """Return a VideoConverter that writes to a temporary directory."""
    return VideoConverter(output_dir=str(tmp_path))


@pytest.fixture
def mock_probe():
    """Patch ffmpeg.probe to return FAKE_PROBE."""
    with patch('ffmpeg.probe', return_value=FAKE_PROBE) as m:
        yield m


@pytest.fixture
def mock_subprocess_ok():
    """
    Patch subprocess.Popen to simulate a successful ffmpeg process.
    The fake stderr emits one progress line then EOF.
    """
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.poll.return_value = 0   # always signals process finished
    mock_process.stderr.readline.side_effect = [
        'frame=  60 fps= 30 q=28.0 size=    1024kB time=00:00:02.00 bitrate= 512.0kbits/s speed=1.00x\n',
        '',
    ]
    with patch('subprocess.Popen', return_value=mock_process) as m:
        yield m


# __init__ tests

class TestInit:
    def test_creates_output_directory(self, tmp_path):
        out = tmp_path / "new_output"
        assert not out.exists()
        VideoConverter(output_dir=str(out))
        assert out.exists()

    def test_output_dir_stored_as_path(self, tmp_path):
        vc = VideoConverter(output_dir=str(tmp_path))
        assert isinstance(vc.output_dir, Path)
        assert vc.output_dir == tmp_path

    def test_default_output_dir_is_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vc = VideoConverter()
        assert vc.output_dir == Path("output")


# get_video_info tests

class TestGetVideoInfo:
    def test_returns_dict(self, converter, mock_probe):
        info = converter.get_video_info("fake.mp4")
        assert isinstance(info, dict)

    def test_basic_format_fields(self, converter, mock_probe):
        info = converter.get_video_info("fake.mp4")
        assert info['filename'] == 'fake.mp4'
        assert info['format'] == 'mov,mp4,m4a,3gp,3g2,mj2'
        assert info['duration'] == 120.0
        assert info['size_bytes'] == 10485760
        assert info['size_mb'] == 10.0
        assert info['bitrate'] == 699050

    def test_video_stream_fields(self, converter, mock_probe):
        info = converter.get_video_info("fake.mp4")
        assert info['video_codec'] == 'h264'
        assert info['width'] == 1920
        assert info['height'] == 1080
        assert info['fps'] == 30.0
        assert info['video_bitrate'] == 600000

    def test_audio_stream_fields(self, converter, mock_probe):
        info = converter.get_video_info("fake.mp4")
        assert info['audio_codec'] == 'aac'
        assert info['audio_bitrate'] == 128000
        assert info['sample_rate'] == 44100

    def test_no_audio_stream(self, converter):
        probe_no_audio = {
            'format': FAKE_PROBE['format'],
            'streams': [FAKE_PROBE['streams'][0]],  # video only
        }
        with patch('ffmpeg.probe', return_value=probe_no_audio):
            info = converter.get_video_info("fake.mp4")
        assert 'audio_codec' not in info
        assert 'audio_bitrate' not in info
        assert 'sample_rate' not in info

    def test_no_video_stream(self, converter):
        probe_no_video = {
            'format': FAKE_PROBE['format'],
            'streams': [FAKE_PROBE['streams'][1]],  # audio only
        }
        with patch('ffmpeg.probe', return_value=probe_no_video):
            info = converter.get_video_info("fake.mp4")
        assert 'video_codec' not in info
        assert 'width' not in info

    def test_raises_on_ffmpeg_error(self, converter):
        err = ffmpeg.Error('ffprobe', None, b'ffprobe error')
        with patch('ffmpeg.probe', side_effect=err):
            with pytest.raises(ffmpeg.Error):
                converter.get_video_info("nonexistent.mp4")

    def test_video_bitrate_none_when_missing(self, converter):
        probe = {
            'format': FAKE_PROBE['format'],
            'streams': [
                {
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1280,
                    'height': 720,
                    'r_frame_rate': '25/1',
                    # no bit_rate key
                },
            ],
        }
        with patch('ffmpeg.probe', return_value=probe):
            info = converter.get_video_info("fake.mp4")
        assert info['video_bitrate'] is None


# convert_video tests

class TestConvertVideo:
    """Tests for the convert_video method, with ffmpeg and subprocess mocked."""

    def _patch_all(self, converter, probe_side_effect=None, returncode=0):
        """
        Context-manager helper that patches the external calls used by
        convert_video:  ffmpeg.probe, ffmpeg.input/output/compile/filter
        and subprocess.Popen.
        """
        import contextlib

        @contextlib.contextmanager
        def cm():
            probes = probe_side_effect or [FAKE_PROBE, FAKE_OUTPUT_PROBE]

            mock_process = MagicMock()
            mock_process.returncode = returncode
            mock_process.poll.return_value = returncode  # constant – loop exits on first empty read
            mock_process.stderr.readline.side_effect = [
                'time=00:00:02.00 bitrate= 512.0kbits/s\n',
                '',
            ]
            mock_process.wait.return_value = None

            with patch('ffmpeg.probe', side_effect=probes), \
                 patch('ffmpeg.input', return_value=MagicMock()), \
                 patch('ffmpeg.output', return_value=MagicMock()), \
                 patch('ffmpeg.compile', return_value=['ffmpeg', '-i', 'fake']), \
                 patch('ffmpeg.filter', return_value=MagicMock()), \
                 patch('subprocess.Popen', return_value=mock_process):
                yield mock_process

        return cm()

    # Stats structure

    def test_stats_keys_present(self, converter):
        with self._patch_all(converter):
            stats = converter.convert_video("input/fake.mp4")
        expected_keys = [
            'input_file', 'output_file',
            'input_size_mb', 'output_size_mb',
            'space_saved_mb', 'space_saved_percent',
            'compression_ratio', 'execution_time_seconds',
            'input_resolution', 'output_resolution',
            'input_codec', 'output_codec',
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_space_saved_values(self, converter):
        with self._patch_all(converter):
            stats = converter.convert_video("input/fake.mp4")
        assert stats['input_size_mb'] == 10.0
        assert stats['output_size_mb'] == 5.0
        assert stats['space_saved_mb'] == 5.0
        assert stats['space_saved_percent'] == 50.0
        assert stats['compression_ratio'] == 2.0

    def test_output_filename_format(self, converter):
        with self._patch_all(converter):
            stats = converter.convert_video("input/myvideo.mp4", output_format='mkv')
        assert stats['output_file'] == 'myvideo_converted.mkv'

    def test_execution_time_is_positive(self, converter):
        with self._patch_all(converter):
            stats = converter.convert_video("input/fake.mp4")
        assert stats['execution_time_seconds'] >= 0

    def test_resolutions_in_stats(self, converter):
        with self._patch_all(converter):
            stats = converter.convert_video("input/fake.mp4")
        assert stats['input_resolution'] == '1920x1080'
        assert stats['output_resolution'] == '1920x1080'

    def test_codecs_in_stats(self, converter):
        with self._patch_all(converter):
            stats = converter.convert_video("input/fake.mp4")
        assert stats['input_codec'] == 'h264'
        assert stats['output_codec'] == 'h264'

    # FFmpeg process failure

    def test_raises_on_process_failure(self, converter):
        with self._patch_all(converter, returncode=1):
            with pytest.raises(Exception, match="FFmpeg conversion failed"):
                converter.convert_video("input/fake.mp4")

    # Codec / format options

    def test_default_codec_is_libx264(self, converter):
        with self._patch_all(converter), \
             patch('ffmpeg.output') as mock_output:
            mock_output.return_value = MagicMock()
            converter.convert_video("input/fake.mp4")
            mock_output.assert_called_once()
            call_kwargs = mock_output.call_args[1]
            assert call_kwargs.get('c:v') == 'libx264'

    def test_crf_passed_when_no_bitrate(self, converter):
        with self._patch_all(converter), \
             patch('ffmpeg.output') as mock_output:
            mock_output.return_value = MagicMock()
            converter.convert_video("input/fake.mp4", crf=18)
            mock_output.assert_called_once()
            call_kwargs = mock_output.call_args[1]
            assert call_kwargs.get('crf') == 18

    def test_video_bitrate_overrides_crf(self, converter):
        with self._patch_all(converter), \
             patch('ffmpeg.output') as mock_output:
            mock_output.return_value = MagicMock()
            converter.convert_video("input/fake.mp4", video_bitrate='2M')
            mock_output.assert_called_once()
            call_kwargs = mock_output.call_args[1]
            assert 'crf' not in call_kwargs
            assert call_kwargs.get('b:v') == '2M'

    def test_resolution_triggers_scale_filter(self, converter):
        with self._patch_all(converter), \
             patch('ffmpeg.filter') as mock_filter:
            mock_filter.return_value = MagicMock()
            converter.convert_video("input/fake.mp4", resolution=(1280, 720))
            mock_filter.assert_called_once()
            args = mock_filter.call_args[0]
            assert 'scale' in args
            assert 1280 in args
            assert 720 in args

    def test_no_scale_filter_without_resolution(self, converter):
        with self._patch_all(converter), \
             patch('ffmpeg.filter') as mock_filter:
            mock_filter.return_value = MagicMock()
            converter.convert_video("input/fake.mp4", resolution=None)
            mock_filter.assert_not_called()

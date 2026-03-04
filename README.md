# FFmpeg Video Converter WebApp
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![FFmpeg](https://img.shields.io/badge/FFmpeg-v7.0-blue?logo=ffmpeg&logoColor=white)
![StreamLit](https://img.shields.io/badge/-Streamlit-blue?style=flat&logo=streamlit&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-94%25-blue)

A Streamlit web app that wraps the FFmpeg Python library, and provides feedback on space saved, and output video quality.

## Current Features
- ☑️ Convert video to different formats
- ☑️ Analyze original video properties (resolution, bitrate, fps, codecs, etc.)
- ☑️ Provide options for output settings (resolution, bitrate, codec, CRF, preset)
- ☑️ Provide feedback of space saved and output video quality (SSIM and PSNR)
- ☑️ Streamlit GUI

### Usage
1. Clone the repository: `git clone https://github.com/Drake9098/FFmpeg_web_app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app via streamlit: `streamlit run main.py`

### Video Codecs
- `libx264` - H.264 (default, widely compatible)
- `libx265` - H.265/HEVC (better compression, slower)
- `libvpx-vp9` - VP9 (for WebM)
- `libaom-av1` - AV1 (best compression, slowest)

### Audio Codecs
- `aac` - AAC (default)
- `mp3` - MP3
- `opus` - Opus
- `vorbis` - Vorbis

### Presets
- `ultrafast`, `superfast`, `veryfast`, `faster`, `fast` - Faster encoding, larger files
- `medium` - Default balanced option
- `slow`, `slower`, `veryslow` - Slower encoding, smaller files

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in dB (higher is better)
- \>40 dB:  🟢 Excellent quality
- 30-40 dB: 🟡 Good quality
- <30 dB:   🔴 Bad quality

### SSIM (Structural Similarity Index)
- Range: -1 to 1 (1 is identical)
- \>0.95: 🟢 Excellent
- \>0.90: 🟢 Very Good
- \>0.80: 🟡 Good
- \>0.70: 🟠 Fine
- <0.70:  🔴 Bad

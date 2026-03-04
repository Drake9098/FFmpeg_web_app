import json
from pathlib import Path

import streamlit as st

from src.quality_analyzer import VideoQualityAnalyzer


def run_conversion(
    converter, input_path: Path, settings: dict, analyze_quality: bool, save_stats
):
    """
    Render the 'Converti Video' button and, when clicked, run the conversion
    pipeline. Results are stored in st.session_state.conversion_result.
    """
    codec_mapping = {
        "h264": "libx264",
        "h265": "libx265",
        "vp9": "libvpx-vp9",
        "av1": "libaom-av1",
    }
    audio_codec_mapping = {
        "aac": "aac",
        "mp3": "libmp3lame",
        "opus": "libopus",
        "vorbis": "libvorbis",
    }

    if st.button("Converti Video", type="primary", use_container_width=True):
        st.session_state.conversion_result = None

        video_codec = codec_mapping.get(settings["codec"], settings["codec"])
        audio_codec = audio_codec_mapping.get(
            settings["audio_codec"], settings["audio_codec"]
        )

        resolution = None
        if settings["width"] != "Originale" and settings["height"] != "Originale":
            resolution = (int(settings["width"]), int(settings["height"]))

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Conversione in corso...")

            stats = converter.convert_video(
                input_path=str(input_path),
                output_format=settings["output_format"],
                video_codec=video_codec,
                audio_codec=audio_codec,
                crf=settings["crf"],
                preset=settings["preset"],
                video_bitrate=settings["video_bitrate"],
                audio_bitrate=settings["audio_bitrate"],
                resolution=resolution,
                progress_bar=progress_bar,
            )

            output_path = Path("output") / stats["output_file"]

            if analyze_quality:
                status_text.text("Analisi qualità...")
                analyzer = VideoQualityAnalyzer()
                quality_results = analyzer.compare_videos(
                    str(input_path), str(output_path)
                )
                stats["quality_metrics"] = quality_results

            progress_bar.progress(100)
            status_text.text("✓ Completato!")

            stats_path = None
            if save_stats and "quality_metrics" in stats:
                stats_path = output_path.with_suffix(".json")
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=4)

            st.session_state.conversion_result = {
                "stats": stats,
                "output_path": str(output_path),
                "stats_path": str(stats_path) if stats_path else None,
                "output_format": settings["output_format"],
                "analyze_quality": analyze_quality,
                "save_stats": save_stats,
            }

        except Exception as e:
            st.error(f"Errore durante la conversione: {e}")
            progress_bar.empty()
            status_text.empty()

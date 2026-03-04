import json
from pathlib import Path
import streamlit as st


def page_config():
    """
    Setup the Streamlit page configuration and title.
    """
    st.set_page_config(
        page_title="Video Converter",
        page_icon="🎬",
        layout="centered"
    )

    st.title("Video Converter")
    st.markdown("Conversione video con FFmpeg")


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save the uploaded file to the input/ directory and return its path.
    """
    input_path = Path("input") / uploaded_file.name
    input_path.parent.mkdir(exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    return input_path


def show_video_info(converter, input_path: Path):
    """
    Display an expander with metadata about the uploaded video.
    """
    with st.expander("📊 Info Video", expanded=True):
        try:
            info = converter.get_video_info(str(input_path))
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dimensione", f"{info['size_mb']} MB")
                st.metric("Durata", f"{info['duration']:.1f}s")

            with col2:
                st.metric("Risoluzione", f"{info['width']}x{info['height']}")
                st.metric("FPS", f"{info['fps']:.1f}")

            with col3:
                st.metric("Codec Video", info['video_codec'])
                st.metric("Codec Audio", info.get('audio_codec', 'N/A'))
        except Exception as e:
            st.error(f"Errore nell'analisi: {e}")


def show_conversion_settings() -> dict:
    """
    Render all conversion settings widgets and return the selected values as a dict.

    Returns a dict with keys:
        width, height, codec, audio_codec, output_format,
        crf, video_bitrate, audio_bitrate, preset
    """
    st.subheader("⚙️ Impostazioni")

    st.info(
        "ℹ️ **Nota:** Non tutte le combinazioni di formato, codec video e codec audio sono compatibili tra loro. "
        "Scegliere parametri incompatibili (es. H.264 in WebM, o AAC in MKV con VP9) può causare errori di conversione. "
        "Si consiglia di usare combinazioni note: \n\n"
        "**MP4 → H.264/AAC**, **WebM → VP9/Opus**, **MKV → H.265/AAC**."
    )

    col1, col2 = st.columns(2)

    with col1:
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            width = st.selectbox(
                "Larghezza",
                ["Originale", "1920", "1280", "854", "640"],
                index=0
            )
        with res_col2:
            height = st.selectbox(
                "Altezza",
                ["Originale", "1080", "720", "480", "360"],
                index=0
            )

        codec = st.selectbox(
            "Codec Video",
            ["h264", "h265", "vp9", "av1"],
            index=0
        )

        audio_codec = st.selectbox(
            "Codec Audio",
            ["aac", "mp3", "opus", "vorbis"],
            index=0,
            help="Codec per l'audio del video"
        )

        output_format = st.selectbox(
            "Formato Output",
            ["mp4", "mkv", "webm", "avi", "mov"],
            index=0
        )

    with col2:
        bitrate_mode = st.radio(
            "Modalità Bitrate Video",
            ["CRF (Qualità Costante)", "CBR (Bitrate Costante)"],
            index=0,
            help="CRF mantiene qualità costante, CBR mantiene bitrate costante"
        )

        if bitrate_mode == "CRF (Qualità Costante)":
            crf = st.slider(
                "Qualità (CRF)",
                min_value=18,
                max_value=32,
                value=23,
                help="Valori più bassi = qualità migliore"
            )
            video_bitrate = None
        else:
            crf = None
            video_bitrate = st.text_input(
                "Bitrate Video",
                value="2M",
                help="Es: 1M, 2M, 5M, 2000k"
            )

        audio_bitrate = st.text_input(
            "Bitrate Audio",
            value="192k",
            help="Es: 128k, 192k, 256k, 320k"
        )

        preset = st.selectbox(
            "Velocità Conversione",
            ['ultrafast', 'superfast', 'veryfast', 'faster',
             'fast', 'medium', 'slow', 'slower', 'veryslow'],
            index=5
        )

    return {
        "width": width,
        "height": height,
        "codec": codec,
        "audio_codec": audio_codec,
        "output_format": output_format,
        "crf": crf,
        "video_bitrate": video_bitrate,
        "audio_bitrate": audio_bitrate,
        "preset": preset,
    }


def show_quality_options(codec: str) -> tuple:
    """
    Render quality-analysis checkboxes.

    Returns (analyze_quality, save_stats).
    save_stats is None when quality analysis is unavailable.
    """
    if codec not in ("vp9", "av1"):
        analyze_quality = st.checkbox("Analizza la qualità della conversione")
        save_stats = st.checkbox("Salva l'analisi e le metriche dettagliate in JSON")
    else:
        st.info(
            "ℹ️ L'analisi della qualità non è disponibile per i codec VP9 e AV1 "
            "a causa di limitazioni tecniche di OpenCV."
        )
        analyze_quality = False
        save_stats = None

    # If the user wants to save stats, quality analysis must run regardless
    if save_stats is not None and not analyze_quality:
        analyze_quality = True

    return analyze_quality, save_stats


def run_conversion(converter, input_path: Path, settings: dict,
                   analyze_quality: bool, save_stats):
    """
    Render the 'Converti Video' button and, when clicked, run the conversion
    pipeline. Results are stored in st.session_state.conversion_result.
    """
    from src.quality_analyzer import VideoQualityAnalyzer

    codec_mapping = {
        'h264': 'libx264',
        'h265': 'libx265',
        'vp9': 'libvpx-vp9',
        'av1': 'libaom-av1',
    }
    audio_codec_mapping = {
        'aac': 'aac',
        'mp3': 'libmp3lame',
        'opus': 'libopus',
        'vorbis': 'libvorbis',
    }

    if st.button("Converti Video", type="primary", use_container_width=True):
        st.session_state.conversion_result = None

        video_codec = codec_mapping.get(settings['codec'], settings['codec'])
        audio_codec = audio_codec_mapping.get(settings['audio_codec'], settings['audio_codec'])

        resolution = None
        if settings['width'] != "Originale" and settings['height'] != "Originale":
            resolution = (int(settings['width']), int(settings['height']))

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Conversione in corso...")

            stats = converter.convert_video(
                input_path=str(input_path),
                output_format=settings['output_format'],
                video_codec=video_codec,
                audio_codec=audio_codec,
                crf=settings['crf'],
                preset=settings['preset'],
                video_bitrate=settings['video_bitrate'],
                audio_bitrate=settings['audio_bitrate'],
                resolution=resolution,
                progress_bar=progress_bar
            )

            output_path = Path("output") / stats['output_file']

            if analyze_quality:
                status_text.text("Analisi qualità...")
                analyzer = VideoQualityAnalyzer()
                quality_results = analyzer.compare_videos(
                    str(input_path),
                    str(output_path)
                )
                stats['quality_metrics'] = quality_results

            progress_bar.progress(100)
            status_text.text("✓ Completato!")

            stats_path = None
            if save_stats and 'quality_metrics' in stats:
                stats_path = output_path.with_suffix('.json')
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=4)

            st.session_state.conversion_result = {
                'stats': stats,
                'output_path': str(output_path),
                'stats_path': str(stats_path) if stats_path else None,
                'output_format': settings['output_format'],
                'analyze_quality': analyze_quality,
                'save_stats': save_stats,
            }

        except Exception as e:
            st.error(f"Errore durante la conversione: {e}")
            progress_bar.empty()
            status_text.empty()


def show_results():
    """
    Display conversion results stored in st.session_state.conversion_result.
    """
    if not st.session_state.get('conversion_result'):
        return

    r = st.session_state.conversion_result
    stats = r['stats']
    output_path = Path(r['output_path'])
    stats_path = Path(r['stats_path']) if r['stats_path'] else None

    st.success("Conversione completata con successo!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dimensione Originale", f"{stats['input_size_mb']:.1f} MB")
    with col2:
        st.metric("Dimensione Finale", f"{stats['output_size_mb']:.1f} MB")
    with col3:
        st.metric("Riduzione", f"{stats['compression_ratio']:.1%}")

    if r['analyze_quality'] and 'quality_metrics' in stats:
        st.subheader("📈 Metriche Qualità")
        qm = stats['quality_metrics']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PSNR", f"{qm['psnr_mean']:.2f} dB")
            st.metric("PSNR Min", f"{qm['psnr_min']:.2f} dB")
            st.metric("PSNR Max", f"{qm['psnr_max']:.2f} dB")
        with col2:
            st.metric("SSIM", f"{qm['ssim_mean']:.4f}")
            st.metric("SSIM Min", f"{qm['ssim_min']:.4f}")
            st.metric("SSIM Max", f"{qm['ssim_max']:.4f}")

    with open(output_path, "rb") as file:
        st.download_button(
            label="📼 Scarica Video",
            data=file,
            file_name=stats['output_file'],
            mime=f"video/{r['output_format']}",
            use_container_width=True
        )

    if r['save_stats'] and stats_path and stats_path.exists():
        with open(stats_path, "rb") as file:
            st.download_button(
                label="📄 Scarica JSON",
                data=file,
                file_name=stats_path.name,
                mime="application/json",
                use_container_width=True
            )
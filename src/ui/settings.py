import streamlit as st


def show_conversion_settings() -> dict:
    """
    Render all conversion settings widgets and return the selected values as a dict.

    Returns a dict with keys:
        width, height, codec, audio_codec, output_format,
        crf, video_bitrate, audio_bitrate, preset
    """
    st.subheader("⚙️ Impostazioni")

    st.info(
        "ℹ️ **Nota:** Non tutte le combinazioni di formato, codec video e codec audio "
        "sono compatibili tra loro. "
        "Scegliere parametri incompatibili (es. H.264 in WebM, o AAC in MKV con VP9) "
        "può causare errori di conversione. "
        "Si consiglia di usare combinazioni note: \n\n"
        "**MP4 → H.264/AAC**, **WebM → VP9/Opus**, **MKV → H.265/AAC**."
    )

    col1, col2 = st.columns(2)

    with col1:
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            width = st.selectbox(
                "Larghezza", ["Originale", "1920", "1280", "854", "640"], index=0
            )
        with res_col2:
            height = st.selectbox(
                "Altezza", ["Originale", "1080", "720", "480", "360"], index=0
            )

        codec = st.selectbox("Codec Video", ["h264", "h265", "vp9", "av1"], index=0)

        audio_codec = st.selectbox(
            "Codec Audio",
            ["aac", "mp3", "opus", "vorbis"],
            index=0,
            help="Codec per l'audio del video",
        )

        output_format = st.selectbox(
            "Formato Output", ["mp4", "mkv", "webm", "avi", "mov"], index=0
        )

    with col2:
        bitrate_mode = st.radio(
            "Modalità Bitrate Video",
            ["CRF (Qualità Costante)", "CBR (Bitrate Costante)"],
            index=0,
            help="CRF mantiene qualità costante, CBR mantiene bitrate costante",
        )

        if bitrate_mode == "CRF (Qualità Costante)":
            crf = st.slider(
                "Qualità (CRF)",
                min_value=18,
                max_value=32,
                value=23,
                help="Valori più bassi = qualità migliore",
            )
            video_bitrate = None
        else:
            crf = None
            video_bitrate = st.text_input(
                "Bitrate Video", value="2M", help="Es: 1M, 2M, 5M, 2000k"
            )

        audio_bitrate = st.text_input(
            "Bitrate Audio", value="192k", help="Es: 128k, 192k, 256k, 320k"
        )

        preset = st.selectbox(
            "Velocità Conversione",
            [
                "ultrafast",
                "superfast",
                "veryfast",
                "faster",
                "fast",
                "medium",
                "slow",
                "slower",
                "veryslow",
            ],
            index=5,
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

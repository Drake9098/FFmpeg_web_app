from pathlib import Path

import streamlit as st


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save the uploaded file to the input/ directory and return its path.
    """
    input_path: Path = Path("input") / str(uploaded_file.name)
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
                st.metric("Codec Video", info["video_codec"])
                st.metric("Codec Audio", info.get("audio_codec", "N/A"))
        except Exception as e:
            st.error(f"Errore nell'analisi: {e}")

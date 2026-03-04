import streamlit as st


def page_config():
    """
    Setup the Streamlit page configuration and title.
    """
    st.set_page_config(page_title="Video Converter", page_icon="🎬", layout="centered")

    st.title("Video Converter")
    st.markdown("Conversione video con FFmpeg")

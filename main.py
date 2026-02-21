from pathlib import Path
import json
import streamlit as st

# Page config
st.set_page_config(
    page_title="Video Converter",
    page_icon="🎬",
    layout="centered"
)

st.title("Video Converter")
st.markdown("Conversione video con FFmpeg")

# Initialize converter Object
### converter = VideoConverter()

# File upload
uploaded_file = st.file_uploader(
    "Carica un video",
    type=['mp4', 'avi', 'mov', 'mkv', 'webm'] # Supported formats
)

if uploaded_file:
    # Save uploaded file temporarily
    input_path = Path("input") / uploaded_file.name
    input_path.parent.mkdir(exist_ok=True)
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    ### Show video info

    with st.expander("📊 Info Video", expanded=True):
        try:
            ### Get video info using converter object
            st.markdown("#### Placeholder dettagli del video") 
        except Exception as e:
            st.error(f"Errore nell'analisi: {e}")

    # Conversion settings
    st.subheader("⚙️ Impostazioni")
    st.markdown("#### Placeholder per le impostazioni di conversione")
    ### Implement conversion settings UI (format, codec, crf, preset, resolution, bitrate, etc.)

    # Convert button
    if st.button("Converti Video", type="primary", use_container_width=True):
        ### Implement conversion logic when button is clicked
        ### Show Conversion Results
        ### Show quality analysis results if enabled 
        pass ### Placeholder
        
     
    # Save stats to JSON
    ### Convert stats to JSON

    # Download Video Button
    ### Download converted video

    # Download JSON Button
    ### Download stats as JSON
else:
    st.info("Carica un video per iniziare")

# Footer
st.markdown("---")
st.markdown("FFmpeg Video Converter Web App - Progetto di UAF | AA 2025/2026")

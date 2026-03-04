from src.video_converter import VideoConverter
from src.utils import *

page_config()

converter = VideoConverter()

uploaded_file = st.file_uploader(
    "Carica un video",
    type=['mp4', 'avi', 'mov', 'mkv', 'webm']
)

if uploaded_file:
    input_path = save_uploaded_file(uploaded_file)
    show_video_info(converter, input_path)
    settings = show_conversion_settings()
    analyze_quality, save_stats = show_quality_options(settings['codec'])
    run_conversion(converter, input_path, settings, analyze_quality, save_stats)
    show_results()
else:
    st.info("Carica un video per iniziare")

st.markdown("---")
st.markdown("FFmpeg Video Converter Web App - Progetto di UAF | AA 2025/2026")

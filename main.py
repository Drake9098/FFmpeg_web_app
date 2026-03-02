from pathlib import Path
import json
import streamlit as st

from src.video_converter import VideoConverter
from src.quality_analyzer import VideoQualityAnalyzer

# Page config
st.set_page_config(
    page_title="Video Converter",
    page_icon="🎬",
    layout="centered"
)

st.title("Video Converter")
st.markdown("Conversione video con FFmpeg")

# Initialize converter Object
converter = VideoConverter()

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

    # Show video info

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

    # Conversion settings
    st.subheader("⚙️ Impostazioni")

    st.info(
        "ℹ️ **Nota:** Non tutte le combinazioni di formato, codec video e codec audio sono compatibili tra loro. "
        "Scegliere parametri incompatibili (es. H.264 in WebM, o AAC in MKV con VP9) può causare errori di conversione. "
        "Si consiglia di usare combinazioni note: \n\n"
        "**MP4 → H.264/AAC**, **WebM → VP9/Opus**, **MKV → H.265/AAC**."
    )

    col1, col2 = st.columns(2)

    with col1:
        # Risoluzione divisa in larghezza e altezza
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
        # Video bitrate mode
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
        
        # Audio bitrate
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
    
    save_stats = None

    if codec != "vp9" and codec != "av1":
        analyze_quality = st.checkbox("Analizza la qualità della conversione")
        save_stats = st.checkbox("Salva l'analisi e le metriche dettagliate in JSON")
    else:
        st.info("ℹ️ L'analisi della qualità non è disponibile per i codec VP9 e AV1 a causa di limitazioni tecniche di OpenCV.")
        analyze_quality = False
    
    
    # If saving stats, we need to analyze quality to get the metrics
    if save_stats is not None and not analyze_quality: 
        analyze_quality = True

    # Convert button
    if st.button("Converti Video", type="primary", use_container_width=True):
        # Reset previous results
        st.session_state.conversion_result = None
        
        # Codec mapping
        codec_mapping = {
            'h264': 'libx264',
            'h265': 'libx265',
            'vp9': 'libvpx-vp9',
            'av1': 'libaom-av1'
        }
        
        audio_codec_mapping = {
            'aac': 'aac',
            'mp3': 'libmp3lame',
            'opus': 'libopus',
            'vorbis': 'libvorbis'
        }
        
        video_codec = codec_mapping.get(codec, codec)
        audio_codec = audio_codec_mapping.get(audio_codec, audio_codec)
        
        # Prepare resolution tuple
        if width != "Originale" and height != "Originale":
            resolution = (int(width), int(height))
        else:
            resolution = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Conversione in corso...")
            
            # Convert video
            stats = converter.convert_video(
                input_path=str(input_path),
                output_format=output_format,
                video_codec=video_codec,
                audio_codec=audio_codec,
                crf=crf,
                preset=preset,
                video_bitrate=video_bitrate,
                audio_bitrate=audio_bitrate,
                resolution=resolution,
                progress_bar=progress_bar
            )
            
            output_path = Path("output") / stats['output_file']
            
            # Quality analysis if requested
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

            # Save stats to JSON
            stats_path = None
            if save_stats and 'quality_metrics' in stats:
                stats_path = output_path.with_suffix('.json')
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=4)
            
            # Persist results in session state
            st.session_state.conversion_result = {
                'stats': stats,
                'output_path': str(output_path),
                'stats_path': str(stats_path) if stats_path else None,
                'output_format': output_format,
                'analyze_quality': analyze_quality,
                'save_stats': save_stats,
            }
        
        except Exception as e:
            st.error(f"Errore durante la conversione: {e}")
            progress_bar.empty()
            status_text.empty()

    # Show results from session state (persists across reruns caused by download buttons)
    if st.session_state.get('conversion_result'):
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
        
        # Quality metrics
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

        # Download Video Button
        with open(output_path, "rb") as file:
            st.download_button(
                label="📼 Scarica Video",
                data=file,
                file_name=stats['output_file'],
                mime=f"video/{r['output_format']}",
                use_container_width=True
            )

        # Download JSON Button
        if r['save_stats'] and stats_path and stats_path.exists():
            with open(stats_path, "rb") as file:
                st.download_button(
                    label="📄 Scarica JSON",
                    data=file,
                    file_name=stats_path.name,
                    mime="application/json",
                    use_container_width=True
                )
else:
    st.info("Carica un video per iniziare")

# Footer
st.markdown("---")
st.markdown("FFmpeg Video Converter Web App - Progetto di UAF | AA 2025/2026")

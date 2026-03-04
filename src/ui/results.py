from pathlib import Path

import streamlit as st


def show_results():
    """
    Display conversion results stored in st.session_state.conversion_result.
    """
    if not st.session_state.get("conversion_result"):
        return

    r = st.session_state.conversion_result
    stats = r["stats"]
    output_path = Path(r["output_path"])
    stats_path = Path(r["stats_path"]) if r["stats_path"] else None

    st.success("Conversione completata con successo!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dimensione Originale", f"{stats['input_size_mb']:.1f} MB")
    with col2:
        st.metric("Dimensione Finale", f"{stats['output_size_mb']:.1f} MB")
    with col3:
        st.metric("Riduzione", f"{stats['compression_ratio']:.1%}")

    if r["analyze_quality"] and "quality_metrics" in stats:
        st.subheader("📈 Metriche Qualità")
        qm = stats["quality_metrics"]
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
            file_name=stats["output_file"],
            mime=f"video/{r['output_format']}",
            use_container_width=True,
        )

    if r["save_stats"] and stats_path and stats_path.exists():
        with open(stats_path, "rb") as file:
            st.download_button(
                label="📄 Scarica JSON",
                data=file,
                file_name=stats_path.name,
                mime="application/json",
                use_container_width=True,
            )

import streamlit as st
import requests

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian",
    page_icon="🛡️",
    layout="centered"
)

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.big-font { font-size:20px !important; }
.highlight {
    background-color: #ffcccc;
    padding: 6px 10px;
    border-radius: 8px;
    font-weight: bold;
    color: #990000;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# UI
# =====================================================
st.title("🛡️ Hate Speech Detector")
st.markdown("Masukkan teks untuk mendeteksi ujaran kebencian.")

API_URL = "http://127.0.0.1:8000/predict"

user_input = st.text_area(
    "Masukkan Teks:",
    height=120,
    placeholder="Contoh: Dasar lu bodoh banget!"
)

if st.button("🔍 Analisis", type="primary"):

    if not user_input.strip():
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Menganalisis teks..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": user_input},
                    timeout=30
                )

                if response.status_code != 200:
                    st.error(f"Server error: {response.status_code}")
                    st.stop()

                result = response.json()

                label = result["prediction"]
                confidence = result["original_confidence"]
                is_hate = result["is_hate_speech"]
                triggers = result["trigger_analysis"]
                process_time = result["process_time_seconds"]

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Hasil Prediksi")
                    if is_hate:
                        st.error(f"🚨 **{label}**")
                    else:
                        st.success(f"✅ **{label}**")

                with col2:
                    st.markdown("### Confidence")
                    st.metric(
                        label="Hate Speech Probability",
                        value=f"{confidence * 100:.1f}%"
                    )

                st.caption(f"⏱️ Waktu proses: {process_time} detik")

                # Trigger analysis
                if triggers:
                    st.markdown("### 🔥 Kata Pemicu")
                    for item in triggers:
                        st.markdown(
                            f"""
                            🔴 **{item['word']}**  
                            - Confidence tanpa kata: `{item['confidence_without_word']}`
                            - Impact drop: `{item['impact_drop']}`
                            """
                        )

                    badges = ""
                    for item in triggers:
                        badges += f'<span class="highlight">{item["word"]}</span>'
                    st.markdown(badges, unsafe_allow_html=True)

                elif is_hate:
                    st.info(
                        "Terdeteksi sebagai ujaran kebencian secara kontekstual, "
                        "tanpa satu kata dominan."
                    )
                else:
                    st.info("Teks aman dan tidak mengandung ujaran kebencian.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Tidak dapat terhubung ke API FastAPI.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
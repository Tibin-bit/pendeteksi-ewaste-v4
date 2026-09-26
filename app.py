import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & STATE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global AI E-Waste Detector Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# ---------------------------------------------------------
# 2. FUNGSI AMBIL DAFTAR MODEL AKTIF DARI GOOGLE API
# ---------------------------------------------------------
def get_available_gemini_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                models.append(name)
        return models, None
    except Exception as e:
        return [], str(e)

# ---------------------------------------------------------
# 3. SIDEBAR (KONFIGURASI API KEY & PILIHAN MODEL)
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ AI Core Settings")
    st.caption("Universal E-Waste Detection System")
    
    api_key = st.text_input("Masukkan Gemini API Key:", type="password", help="Dapatkan API Key dari Google AI Studio")
    
    selected_model = None
    if api_key:
        with st.spinner("Memeriksa model aktif di akun kamu..."):
            avail_models, fetch_err = get_available_gemini_models(api_key)
            if avail_models:
                st.success(f"Ditemukan {len(avail_models)} model aktif!")
                selected_model = st.selectbox("Model Gemini Terdeteksi:", avail_models, index=0)
            else:
                st.warning("⚠️ Tidak dapat mengambil daftar model. Memakai fallback standar.")
                selected_model = "gemini-3.8-flash"
    else:
        st.info("💡 Tempelkan API Key kamu di atas untuk mengaktifkan sistem.")
        selected_model = "gemini-3.8-flash"
        
    st.markdown("---")
    st.markdown("### 📋 Standar Klasifikasi")
    st.info("Menggunakan pedoman **UN Global E-Waste Monitor** untuk identifikasi bahaya dan daur ulang sampah elektronik.")
    st.markdown("---")
    st.caption("v3.1 Pro — Gemini 3.8 Ready")

# ---------------------------------------------------------
# 4. FUNGSI ANALISIS GAMBAR (GEMINI 3.8 FLASH)
# ---------------------------------------------------------
def analyze_ewaste_smart(image, key, model_name):
    genai.configure(api_key=key)
    
    prompt = """
    Bertindaklah sebagai Ahli Pengolahan Sampah Elektronik (E-Waste Specialist) berstandar Internasional.
    Analisis gambar ini dengan teliti dan berikan output STRICTLY dalam format JSON murni tanpa markdown formatting di luar JSON.

    Struktur JSON yang wajib dihasilkan:
    {
        "nama_objek": "Nama spesifik perangkat/komponen elektronik pada gambar",
        "kategori_un": "Salah satu dari 6 Kategori UN E-Waste (1. Temperature Exchange Equipment, 2. Screens & Monitors, 3. Lamps, 4. Large Equipment, 5. Small Equipment, 6. Small IT & Telecommunication)",
        "deskripsi": "Deskripsi mendalam mengenai objek yang teridentifikasi",
        "tingkat_bahaya": "Tinggi / Sedang / Rendah",
        "skor_bahaya": 8,
        "bahan_berbahaya": ["Contoh: Timbal", "Raksa", "Kadmium", "CFC/Freon"],
        "potensi_logam_mulia": {
            "Emas (Au)": "Ada / Tidak ada / Tinggi",
            "Perak (Ag)": "Ada / Tidak ada / Sedang",
            "Tembaga (Cu)": "Ada / Tinggi"
        },
        "instruksi_penanganan": [
            "Langkah 1 penanganan aman",
            "Langkah 2 pemisahan komponen",
            "Langkah 3 opsi pembuangan/daur ulang"
        ],
        "dapat_didaur_ulang_persen": 75
    }
    """
    
    # Urutan percobaan model dengan mengutamakan versi 3.8 terbaru
    models_to_try = [model_name]
    fallback_options = ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for fb in fallback_options:
        if fb not in models_to_try:
            models_to_try.append(fb)
            
    response = None
    last_error = ""
    used_model = ""

    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(m_name)
            res = model.generate_content([prompt, image])
            if res and res.text:
                response = res
                used_model = m_name
                break
        except Exception as e:
            last_error = str(e)
            continue

    if response is None:
        return None, None, f"Gagal memproses gambar. Detail error: {last_error}"

    try:
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(clean_text)
        return parsed_data, used_model, None
    except Exception as e:
        return None, used_model, f"Gagal membaca format JSON dari AI: {str(e)}"

# ---------------------------------------------------------
# 5. TAMPILAN UTAMA APLIKASI
# ---------------------------------------------------------
st.title("⚡ Global AI E-Waste Detector Pro")
st.markdown("Sistem Pengenal & Analisis Bahaya Sampah Elektronik Berbasis Vision AI")

tab1, tab2 = st.tabs(["🔍 Analisis E-Waste", "📊 Dashboard & Riwayat"])

# --- TAB 1: ANALISIS ---
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="medium")
    
    with col_input:
        st.subheader("1. Pilih Sumber Gambar")
        source = st.radio("Metode Input:", ["Kamera Langsung 📷", "Unggah Berkas 📁"], horizontal=True)
        
        input_image = None
        if "Kamera" in source:
            cam_file = st.camera_input("Ambil Foto Perangkat E-Waste")
            if cam_file:
                input_image = Image.open(cam_file)
        else:
            uploaded_file = st.file_uploader("Pilih gambar perangkat (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                input_image = Image.open(uploaded_file)
                
        if input_image:
            st.image(input_image, caption="Gambar Siap Dianalisis", use_container_width=True)
            analyze_btn = st.button("🚀 Jalankan Analisis AI Universal", type="primary", use_container_width=True)

    with col_output:
        st.subheader("2. Hasil Deteksi & Analisis Mendalam")
        
        if 'analyze_btn' in locals() and analyze_btn:
            if not api_key:
                st.error("⚠️ **API Key Belum Diisi!** Masukkan Gemini API Key pada menu sidebar di sebelah kiri.")
            elif input_image is None:
                st.warning("⚠️ **Gambar Belum Ada!** Ambil foto atau unggah gambar terlebih dahulu.")
            else:
                target_model = selected_model if selected_model else "gemini-3.8-flash"
                with st.spinner(f"🧠 Menganalisis gambar menggunakan model `{target_model}`..."):
                    data, active_model, err = analyze_ewaste_smart(input_image, api_key, target_model)
                    
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state.detection_history.append({
                            "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "nama": data.get("nama_objek", "Tidak diketahui"),
                            "kategori": data.get("kategori_un", "Umum"),
                            "bahaya": data.get("tingkat_bahaya", "Sedang"),
                            "daur_ulang": data.get("dapat_didaur_ulang_persen", 0),
                            "model": active_model
                        })
                        
                        st.success(f"✅ **Berhasil Dianalisis** (Model Digunakan: `{active_model}`)")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Perangkat Terdeteksi", data.get("nama_objek", "-"))
                        m2.metric("Tingkat Bahaya", data.get("tingkat_bahaya", "-"), delta=f"Skor {data.get('skor_bahaya', 0)}/10", delta_color="inverse")
                        m3.metric("Potensi Daur Ulang", f"{data.get('dapat_didaur_ulang_persen', 0)}%")
                        
                        st.markdown("---")
                        st.markdown(f"**📂 Kategori UN E-Waste:** `{data.get('kategori_un', '-')}`")
                        st.markdown(f"**📝 Deskripsi Objek:** {data.get('deskripsi', '-')}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("🚨 **Bahan / Zat Berbahaya:**")
                            for bahan in data.get("bahan_berbahaya", []):
                                st.write(f"- {bahan}")
                                
                        with col_b:
                            st.markdown("💎 **Potensi Logam Mulia:**")
                            lm = data.get("potensi_logam_mulia", {})
                            for k, v in lm.items():
                                st.write(f"- **{k}:** {v}")
                                
                        st.markdown("---")
                        st.markdown("🛠️ **Instruksi Penanganan & Daur Ulang Aman:**")
                        for idx, step in enumerate(data.get("instruksi_penanganan", []), 1):
                            st.write(f"**{idx}.** {step}")

# --- TAB 2: DASHBOARD ---
with tab2:
    st.subheader("📊 Rekapitulasi Deteksi E-Waste")
    
    if len(st.session_state.detection_history) > 0:
        df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df, names="kategori", title="Distribusi Kategori UN E-Waste", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            fig_bar = px.bar(df, x="nama", y="daur_ulang", color="bahaya", title="Persentase Daur Ulang per Objek")
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Belum ada riwayat deteksi pada sesi ini. Lakukan deteksi di Tab 1 untuk melihat dashboard.")


 

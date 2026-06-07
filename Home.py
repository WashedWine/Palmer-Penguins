import streamlit as st

st.set_page_config(
    page_title="Penguin Data Mining",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐧 Klasifikasi Spesies dan Segmentasi Karakteristik Fisik Penguin")
st.markdown("### Menggunakan Random Forest dan K-Means Berbasis Streamlit")

st.markdown("""
<div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-bottom: 20px;">
    <h4>Tujuan Penelitian</h4>
    <p>Aplikasi dashboard ini dibangun untuk memenuhi project <b>UAS Data Mining</b>. Proyek ini mengimplementasikan analisis komprehensif terhadap dataset fisik penguin. Analisis meliputi proses klasifikasi spesies menggunakan algoritma <b>Random Forest</b> dan segmentasi karakteristik fisik penguin menggunakan <b>K-Means Clustering</b>.</p>
</div>
""", unsafe_allow_html=True)

st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔍 Framework CRISP-DM")
    st.markdown("""
    Aplikasi ini merangkum siklus Data Mining dengan standar CRISP-DM:
    1. **Business Understanding**: Memahami tujuan klasifikasi dan segmentasi habitat/fisik penguin.
    2. **Data Understanding**: Eksplorasi atribut fisik melalui visualisasi statistik.
    3. **Data Preparation**: Cleansing, penanganan *missing values*, *scaling*, dan augmentasi (600 records).
    4. **Modeling**: Membangun algoritma Random Forest dan K-Means.
    5. **Evaluation**: Mengukur *Accuracy*, *Precision*, *Recall*, dan *Silhouette Score*.
    6. **Deployment**: Membangun web interaktif ini menggunakan Streamlit.
    """)

with col2:
    st.markdown("#### 📊 Informasi Model & Dataset")
    st.markdown("""
    - **Dataset**: `dataset_penguin_augmentasi.csv` (Total 600 records)
    - **Fitur Utama**:
        - `bill_length_mm` (Panjang paruh)
        - `bill_depth_mm` (Kedalaman paruh)
        - `flipper_length_mm` (Panjang sirip)
        - `body_mass_g` (Berat badan)
    - **Target Label**: `species` (Adelie, Chinstrap, Gentoo)
    - **Algoritma Machine Learning**: 
        - Classification: **Random Forest Classifier**
        - Clustering: **K-Means Clustering**
    """)

st.write("---")
st.markdown("#### 👥 Anggota Kelompok")
st.info("""
1. Yesy Graceseyla Pakpahan (24051214143) 
2. Neilsen Iyanaka Santoso (24051214169) 
""")
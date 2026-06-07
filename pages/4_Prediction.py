import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Prediction", page_icon="🚀", layout="wide")
st.title("🚀 Prediksi Interaktif")

@st.cache_resource
def load_all_models():
    df = pd.read_csv("dataset/dataset_penguin_augmentasi.csv")
    rf = joblib.load("model/random_forest.pkl")
    scaler = joblib.load("model/scaler.pkl")
    kmeans = joblib.load("model/kmeans.pkl")
    return df, rf, scaler, kmeans

df, rf, scaler, kmeans = load_all_models()

st.markdown("Masukkan karakteristik fisik penguin di *sidebar* untuk melihat hasil prediksi spesies (RF) dan segmentasi kelompoknya (K-Means).")

st.sidebar.header("🎛️ Input Karakteristik Fisik")

bill_length = st.sidebar.slider("Bill Length (mm)", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Bill Depth (mm)", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Flipper Length (mm)", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass = st.sidebar.number_input("Body Mass (g)", min_value=float(df['body_mass_g'].min()), max_value=float(df['body_mass_g'].max()), value=float(df['body_mass_g'].mean()))

bg_color = '#0E1117'
font_color = '#FAFAFA'

if st.sidebar.button("Prediksi Sekarang 🚀"):
    input_data = pd.DataFrame({
        'bill_length_mm': [bill_length],
        'bill_depth_mm': [bill_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass]
    })
    
    # --- PROSES CLASSIFICATION ---
    pred_class_encoded = rf.predict(input_data)[0]
    pred_proba = rf.predict_proba(input_data)[0]
    species_mapping = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
    pred_species = species_mapping[pred_class_encoded]
    confidence = np.max(pred_proba) * 100
    
    # --- PROSES CLUSTERING ---
    input_scaled = scaler.transform(input_data)
    pred_cluster = kmeans.predict(input_scaled)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌲 Prediksi Random Forest")
        st.markdown(f"#### Spesies Terprediksi: <span style='color:#FF4B4B'>{pred_species}</span>", unsafe_allow_html=True)
        st.progress(int(confidence))
        st.write(f"Confidence Score: **{confidence:.2f}%**")
        
        prob_df = pd.DataFrame({'Species': ['Adelie', 'Chinstrap', 'Gentoo'], 'Probability': pred_proba})
        fig_prob = px.bar(prob_df, x='Species', y='Probability', text_auto='.2%', title="Probability Distribution")
        fig_prob.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color, height=300)
        fig_prob.update_traces(marker_color='#FF4B4B')
        st.plotly_chart(fig_prob, use_container_width=True)

    with col2:
        st.subheader("🧩 Segmentasi K-Means")
        st.markdown(f"#### Masuk ke Segmentasi: <span style='color:#60B4FF'>Cluster {pred_cluster}</span>", unsafe_allow_html=True)
        
        cluster_desc = {
            0: "Grup penguin dengan ukuran **terbesar** dan sirip panjang (Karakteristik anatomi dominan Gentoo).",
            1: "Grup penguin berukuran kecil dengan **paruh pendek dan dalam** (Karakteristik anatomi dominan Adelie).",
            2: "Grup penguin berukuran kecil dengan **paruh panjang dan dalam** (Karakteristik anatomi dominan Chinstrap)."
        }
        st.info(f"📝 **Deskripsi Cluster:** {cluster_desc[pred_cluster]}")
        
    st.write("---")
    
    # --- VISUALISASI POSISI INPUT ---
    st.subheader("📍 Posisi Data Anda Dibandingkan Seluruh Dataset")
    
    X_all = df.drop('species', axis=1)
    X_all_scaled = scaler.transform(X_all)
    pca = PCA(n_components=2)
    pca.fit(X_all_scaled)
    
    X_pca = pca.transform(X_all_scaled)
    input_pca = pca.transform(input_scaled)
    
    df_pca = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Data Type': 'Dataset Asli'})
    df_input = pd.DataFrame({'PCA1': input_pca[:, 0], 'PCA2': input_pca[:, 1], 'Data Type': 'Input Anda'})
    df_plot = pd.concat([df_pca, df_input])
    
    fig_scatter = px.scatter(df_plot, x='PCA1', y='PCA2', color='Data Type', 
                             color_discrete_map={'Dataset Asli': '#60B4FF', 'Input Anda': '#FF4B4B'})
    
    for trace in fig_scatter.data:
        if trace.name == 'Input Anda':
            trace.marker.size = 20
            trace.marker.symbol = 'star'
        else:
            trace.marker.size = 6
            trace.marker.opacity = 0.5
            
    fig_scatter.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color, title="PCA Scatter Plot 2D")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("👈 Silakan sesuaikan parameter di sidebar dan tekan tombol **Prediksi Sekarang 🚀** untuk memulai simulasi.")
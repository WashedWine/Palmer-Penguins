import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Clustering", page_icon="🧩", layout="wide")
st.title("🧩 Clustering: K-Means Segmentasi")

@st.cache_resource
def load_clustering_assets():
    df = pd.read_csv("dataset/dataset_penguin_augmentasi.csv")
    scaler = joblib.load("model/scaler.pkl")
    kmeans = joblib.load("model/kmeans.pkl")
    return df, scaler, kmeans

df, scaler, kmeans_model = load_clustering_assets()
X = df.drop('species', axis=1)
X_scaled = scaler.transform(X)

bg_color = '#0E1117'
font_color = '#FAFAFA'

st.header("1. Pemilihan Jumlah Cluster (K)")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Elbow Method Visualization")
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        
    fig_elbow = px.line(x=list(K_range), y=wcss, markers=True, 
                        labels={'x':'Jumlah Cluster (K)', 'y':'WCSS (Inertia)'},
                        title="Titik Siku (Elbow) untuk Menentukan K Optimal")
    fig_elbow.update_traces(line_color='#FF4B4B', marker=dict(size=8, color='white'))
    fig_elbow.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    st.subheader("Evaluasi Cluster")
    best_k = kmeans_model.n_clusters
    labels = kmeans_model.predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    
    st.metric("Jumlah Cluster Terpilih (K)", best_k)
    st.metric("Silhouette Score", f"{sil_score:.4f}", help="Nilai mendekati 1 menandakan pemisahan cluster yang sangat baik.")

st.write("---")
st.header("2. PCA Visualization (2D)")
st.write("Memproyeksikan 4 dimensi data fisik menjadi 2 dimensi menggunakan Principal Component Analysis (PCA).")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': labels.astype(str), 'Species Asli': df['species']})

fig_pca = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', symbol='Species Asli',
                     title="Sebaran Cluster vs Spesies Asli",
                     color_discrete_sequence=['#FF4B4B', '#60B4FF', '#29B09D'])
fig_pca.update_traces(marker=dict(size=8, opacity=0.8))
fig_pca.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
st.plotly_chart(fig_pca, use_container_width=True)

st.write("---")
st.header("3. Cluster Profiling (Rata-rata Fisik)")

df_clustered = df.copy()
df_clustered['Cluster'] = labels
profile = df_clustered.groupby('Cluster')[X.columns].mean().reset_index()

st.dataframe(profile.style.highlight_max(axis=0, color='#4C4F5A'), use_container_width=True)

st.markdown("### 💡 Interpretasi Karakteristik per Cluster")
col_c0, col_c1, col_c2 = st.columns(3)

with col_c0:
    st.success("**Cluster 0**")
    st.write("Mewakili kelompok penguin besar. **Karakteristik:** Flipper paling panjang, body mass paling berat, dan bill depth dangkal. Secara alami selaras dengan anatomi spesies **Gentoo**.")

with col_c1:
    st.info("**Cluster 1**")
    st.write("Mewakili kelompok penguin kompak. **Karakteristik:** Bill length sangat pendek, bill depth dalam, flipper pendek, dan body mass ringan. Selaras dengan anatomi spesies **Adelie**.")

with col_c2:
    st.warning("**Cluster 2**")
    st.write("Mewakili kelompok paruh panjang. **Karakteristik:** Bill length panjang, bill depth dalam, flipper sedang, dan body mass menengah. Selaras dengan anatomi spesies **Chinstrap**.")
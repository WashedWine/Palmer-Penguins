import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")
st.title("📊 Dataset Overview")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/dataset_penguin_augmentasi.csv")

df = load_data()

st.header("1. Informasi Dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Record", f"{df.shape[0]} Data")
col2.metric("Jumlah Fitur", f"{df.shape[1] - 1} Fitur")
col3.metric("Missing Value", f"{df.isnull().sum().sum()} Null")
col4.metric("Jumlah Spesies", df['species'].nunique())

st.subheader("Dataframe Interaktif")
st.dataframe(df, use_container_width=True)

st.header("2. Statistik Deskriptif")
st.dataframe(df.describe(), use_container_width=True)

st.header("3. Visualisasi Data")
tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Species", "Histogram Fitur", "Correlation Heatmap", "Boxplot Spesies"])

colors = ['#FF4B4B', '#60B4FF', '#29B09D']
bg_color = '#0E1117'
font_color = '#FAFAFA'

with tab1:
    fig_pie = px.pie(df, names='species', title='Distribusi Spesies Penguin', color_discrete_sequence=colors)
    fig_pie.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    feature = st.selectbox("Pilih Fitur:", df.columns[:-1])
    fig_hist = px.histogram(df, x=feature, color="species", marginal="box", 
                       title=f"Distribusi Histogram: {feature}", color_discrete_sequence=colors)
    fig_hist.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.markdown("### Correlation Heatmap Antar Fitur")
    corr = df.drop('species', axis=1).corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                    color_continuous_scale='Blues', title="Korelasi Antar Karakteristik Fisik")
    fig_corr.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    feature_box = st.selectbox("Pilih Fitur untuk Boxplot:", df.columns[:-1], key='box')
    fig_box = px.box(df, x="species", y=feature_box, color="species", 
                 title=f"Boxplot {feature_box} berdasarkan Spesies", color_discrete_sequence=colors)
    fig_box.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
    st.plotly_chart(fig_box, use_container_width=True)
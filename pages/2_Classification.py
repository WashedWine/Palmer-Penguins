import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Classification", page_icon="🌲", layout="wide")
st.title("🌲 Classification: Random Forest")

@st.cache_resource
def load_classification_assets():
    df = pd.read_csv("dataset/dataset_penguin_augmentasi.csv")
    rf = joblib.load("model/random_forest.pkl")
    return df, rf

df, rf = load_classification_assets()

le = LabelEncoder()
y = le.fit_transform(df['species'])
X = df.drop('species', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

st.header("1. Evaluasi Model Random Forest")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Accuracy", f"{acc:.4f}")
col2.metric("📊 Precision", f"{prec:.4f}")
col3.metric("🔄 Recall", f"{rec:.4f}")
col4.metric("⭐ F1 Score", f"{f1:.4f}")

st.write("---")
col_cm, col_cr = st.columns(2)

bg_color = '#0E1117'
font_color = '#FAFAFA'

with col_cm:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, 
                       x=le.classes_, y=le.classes_, 
                       color_continuous_scale='Blues',
                       labels=dict(x="Predicted Label", y="True Label", color="Count"))
    fig_cm.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cr:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.background_gradient(cmap='Blues'), use_container_width=True)

st.write("---")
st.header("2. Feature Importance")

importances = rf.feature_importances_
df_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=True)

fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                 title="Pengaruh Fitur terhadap Prediksi Spesies (Random Forest)",
                 color='Importance', color_continuous_scale='Blues')
fig_imp.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, font_color=font_color)
st.plotly_chart(fig_imp, use_container_width=True)

top_feature = df_imp.iloc[-1]['Feature']
second_feature = df_imp.iloc[-2]['Feature']
st.info(f"💡 **Interpretasi:** Fitur **{top_feature}** memiliki peranan paling krusial dalam membedakan spesies penguin pada dataset ini, disusul oleh fitur **{second_feature}** di urutan kedua.")
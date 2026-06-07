import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About This Project")

st.markdown("""
### 📂 Dataset
Dataset yang digunakan adalah **Penguin Size Dataset** yang telah mengalami *data preparation* dan augmentasi hingga mencapai **600 records** demi keperluan stabilitas Machine Learning. Dataset ini memuat karakteristik fisik penguin dari tiga spesies berbeda (Adelie, Chinstrap, Gentoo) yang diamati di Kepulauan Palmer, Antartika.

### 🧠 Metode Data Mining
1. **Random Forest (Classification)**
   Algoritma *ensemble learning* yang membangun puluhan hingga ratusan *decision tree* untuk menghasilkan prediksi yang lebih stabil, akurat, dan dapat meminimalisir *overfitting*.
   
2. **K-Means (Clustering)**
   Algoritma *Unsupervised Learning* terpopuler yang mengelompokkan data tak berlabel ke dalam jumlah cluster *K* berdasarkan kedekatan/kemiripan *Euclidean distance*.

### 🏗️ Framework (CRISP-DM)
Proyek ini mengadopsi standar industri **Cross-Industry Standard Process for Data Mining (CRISP-DM)** yang terdiri dari 6 fase komprehensif:
* **Business Understanding**
* **Data Understanding**
* **Data Preparation**
* **Modeling**
* **Evaluation**
* **Deployment** (Fase saat ini)

### 🛠️ Tools & Libraries
* **Python**: Bahasa Pemrograman.
* **Streamlit**: *Framework* UI untuk Data Science web apps.
* **Scikit-Learn**: *Library Core* Machine Learning.
* **Pandas & NumPy**: Data structure dan komputasi numerik.
* **Plotly & Seaborn**: *Library* Visualisasi interaktif.

### 📚 Referensi Akademik
1. Gorman KB, Williams TD, Fraser WR (2014). *Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis)*. PLoS ONE 9(3): e90081.
2. [Scikit-Learn Documentation](https://scikit-learn.org/)
3. [Streamlit Documentation](https://streamlit.io/)
""")
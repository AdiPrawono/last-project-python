import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import streamlit as st
import pandas as pd
from component.nav import navbar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pickle

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Pendat", layout="wide")
navbar()

# Daftar fitur
fitur = ['V', 'H', 'S']

# Tab Naive Bayes dan KNN
tabs = st.tabs(["Stacking Predict"])

with tabs[0]:
    st.header('Prediksi Menggunakan Metode Stacking')

    # Mengambil input dari pengguna
    feature1 = st.number_input('Masukkan Voltage: ')
    feature2 = st.number_input('Masukkan High: ')
    feature3 = st.number_input('Masukkan Soil Type: ')

    # Menyimpan fitur input dalam list
    final_feature = [feature1, feature2, feature3]

    # Tombol untuk memulai prediksi
    if st.button('Predict'):
        # Membaca model dari file pickle
        with open('model-pickle.pkl', 'rb') as file:
            model = pickle.load(file)

        # Melakukan prediksi
        input_data = np.array(final_feature).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Menampilkan hasil prediksi
        if prediction == 1:
            st.write("Prediction for new data: Tidak Ada Land Mines")
        elif prediction == 2:
            st.write("Prediction for new data: Anti tank")
        elif prediction == 3:
            st.write("Prediction for new data: Anti Personnel")
        elif prediction == 4:
            st.write("Prediction for new data: Bobby Trapped Anti Personnel")
        elif prediction == 5:
            st.write("Prediction for new data: M14 Anti-personnel")

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

    # Inisialisasi list di session state jika belum ada
    if 'knn_data' not in st.session_state:
        st.session_state.knn_data = [0.0] * len(fitur)
    if 'knn_current_index' not in st.session_state:
        st.session_state.knn_current_index = 0

    # Menggunakan form untuk mengelola input secara berkala
    if st.session_state.knn_current_index < len(fitur):
        with st.form(key='knn_input_form'):
            feature = fitur[st.session_state.knn_current_index]
            angka = st.number_input(f'Masukkan {feature}: ', key=f'knn_{feature}', value=st.session_state.knn_data[st.session_state.knn_current_index])
            submit_button = st.form_submit_button(label='Tambah ke Data')

            if submit_button:
                st.session_state.knn_data[st.session_state.knn_current_index] = angka
                st.session_state.knn_current_index += 1
                st.experimental_rerun()

    # Menampilkan list yang telah diisi
    st.write('Data:', st.session_state.knn_data)

    # Memastikan semua fitur telah diinput
    if st.session_state.knn_current_index == len(fitur):
        # Membaca model dari file pickle
        with open('model-pickle.pkl', 'rb') as file:
            model = pickle.load(file)

        # Melakukan prediksi
        input_data = np.array(st.session_state.knn_data).reshape(1, -1)
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

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

st.set_page_config(page_title="Pendat", layout="wide")
navbar()

# Daftar fitur
fitur = ['V','H','S']

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
        with open('model-pickle.pkl','rb') as file:
            model = pickle.load(file)
        prediction = model.predict

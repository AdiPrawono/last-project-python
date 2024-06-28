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
# from imblearn.over_sampling import SMOTE

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
        df = pd.read_csv('data-cleaned.csv')

        # Menghapus kolom yang tidak relevan jika ada
        if 'Unnamed: 0' in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        if 'Outlier' in df.columns:
            df.drop("Outlier", axis=1, inplace=True)

        # Memisahkan fitur dan target
        X = df[fitur]
        y = df['M']

        # Split dataset into training and testing data with random_state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        # Standardisasi fitur
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Metode SMOTE
        # smote = SMOTE(k_neighbors=2, random_state=10)
        # X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Inisialisasi model-model yang akan digunakan
        base_models = [
            ('knn3', KNeighborsClassifier(n_neighbors=3)),
            ('knn5', KNeighborsClassifier(n_neighbors=5)),
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ]

        # Inisialisasi meta-classifier
        meta_classifier = GaussianNB()

        # Inisialisasi stacking classifier
        stack_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            stack_method='predict',  # stack_method='predict' untuk klasifikasi
            cv=5  # cross-validation folds for training meta-classifier
        )

        # Melatih stacking classifier dengan data training
        stack_clf.fit(X_train, y_train)

        # Prediksi pada data uji
        y_test_pred = stack_clf.predict(X_test)
        
        # Prediksi untuk data baru
        X_new = scaler.transform([st.session_state.knn_data])
        y_pred_new = stack_clf.predict(X_new)

        # Evaluasi akurasi
        accuracy = accuracy_score(y_test, y_test_pred)

        # Menampilkan hasil prediksi dan akurasi
        if y_new_pred[0] == 1:
            print(f"Prediction for new data: {y_new_pred[0]} (Tidak Ada Land Mines)")
        elif y_new_pred[0] == 2:
            print(f"Prediction for new data: {y_new_pred[0]} (Anti tank)")
        elif y_new_pred[0] == 3:
            print(f"Prediction for new data: {y_new_pred[0]} (Anti Presonnel)")
        elif y_new_pred[0] == 4:
            print(f"Prediction for new data: {y_new_pred[0]} (Bobby Trapped Anti Presonnel)")
        elif y_new_pred[0] == 5:
            print(f"Prediction for new data: {y_new_pred[0]} (M14 Anti-personnel)")
        print(f'Accuracy: {accuracy * 100:.2f}%')

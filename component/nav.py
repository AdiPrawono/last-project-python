import streamlit as st

def navbar():
    # Membuat 4 kolom
    col1, col2, col3, col4 = st.columns(4, gap="small")

    # Menambahkan konten di dalam kolom pertama
    with col1:
        st.page_link("app.py", label="Home")

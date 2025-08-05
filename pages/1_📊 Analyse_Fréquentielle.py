import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import os
import glob
from utils import analyse_dataframe, lda_model, visualize_lda

file_list = glob.glob('data/*.parquet')

# Extract just the filenames (without the full path)
file_names = [os.path.basename(file) for file in file_list]

# Show files in a selectbox
selected_file = st.selectbox("📁 Choisir un fichier", file_names, index=None)
num_topics = st.slider("Nombre de topics (LDA)", min_value=2, max_value=20, value=3)

if selected_file:
    df=pd.read_parquet(os.path.join('data', selected_file))
    st.write(f"{selected_file} has {df.shape[0]} lines")

if st.button("📊 Lancer l'analyse descriptive"):
    if selected_file : 
        selected_path = os.path.join('data', selected_file)
        analyse_dataframe(selected_path)
    else: 
        st.write("Veuillez séléctionner un fichier.")
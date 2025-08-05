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
selected_file = st.selectbox("Select a file", file_names, index=None)
num_topics = st.slider("Nombre de topics (LDA)", min_value=2, max_value=20, value=3)

if selected_file:
    df=pd.read_parquet(os.path.join('data', selected_file))
    st.write(f"{selected_file} has {df.shape[0]} lines")

if st.button("Lancer l'analysee descriptive"):
    selected_path = os.path.join('data', selected_file)
    analyse_dataframe(selected_path)

if st.button("Lancer le LDA"):
    filename=selected_file.split(".")[0]
    selected_path = os.path.join('intermediate_data', f"{filename}_processed_texts.parquet")
    df_hs = pd.read_parquet(selected_path)
    processed_texts_hs = [list(doc) for doc in df_hs["processed_texts_hs"]]
    model_hs, corpus_hs, dictionary_hs = lda_model(processed_texts_hs, num_topics=num_topics)
    visualize_lda(model_hs, corpus_hs, dictionary_hs)
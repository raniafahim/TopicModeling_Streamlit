import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import os
import glob
from utils import  lda_model, visualize_lda


num_topics = st.slider("Nombre de topics (LDA)", min_value=2, max_value=20, value=3)

if st.button("🔍 Lancer le LDA"):
    st.info("📑 Traitement en cours ...")
    df_hs = pd.read_parquet("intermediate_data_sommaire/processed_titles_hs.parquet")
    all_chunks_hs = [list(title) for summary in df_hs["lda_documents"] for title in summary]
    model_hs, corpus_hs, dictionary_hs = lda_model(all_chunks_hs)
    st.success("✅ Modèle entraîné. Affichage de la visualisation...")
    visualize_lda(model_hs, corpus_hs, dictionary_hs)




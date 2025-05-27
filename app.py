import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Chargement des données
df = pd.read_csv('/content/drive/MyDrive/just-the-basics-the-after-party/train.csv')

# Chargement du modèle
model = joblib.load('/content/modele_final.pkl')

st.title("📊 Tableau de bord IA – Projet de régression")

# Onglets
tab1, tab2, tab3, tab4 = st.tabs(["Statistiques", "Corrélation", "Visualisations", "Prédictions"])

# ---- Statistiques
with tab1:
    st.header("Statistiques descriptives")
    st.write("Voici les statistiques (min, max, moyenne) des données :")
    st.dataframe(df.describe().loc[['min', 'max', 'mean']])

# ---- Matrice de corrélation
with tab2:
    st.header("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ---- Visualisations
with tab3:
    st.header("Visualisation")
    colonne = st.selectbox("Choisir une colonne à visualiser", df.columns)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[colonne], kde=True, ax=ax2)
    st.pyplot(fig2)

# ---- Prédiction
with tab4:
    st.header("Faire une prédiction")
    st.write("Entrer les valeurs pour chaque caractéristique :")

    colonnes = df.columns.tolist()[:-1]  # suppose que la dernière colonne est la cible
    valeurs = []

    for col in colonnes:
        valeur = st.number_input(f"{col}", value=float(df[col].mean()))
        valeurs.append(valeur)

    if st.button("Prédire"):
        input_df = pd.DataFrame([valeurs], columns=colonnes)
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Prédiction du modèle : {prediction:.2f}")

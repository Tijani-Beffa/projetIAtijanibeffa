import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Tableau de bord IA", layout="wide")
st.title("📊 Tableau de bord IA – Projet de régression")

# === Upload du fichier CSV
uploaded_file = st.file_uploader("📁 Importer le fichier `train.csv`", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Fichier chargé avec succès.")

    # Chargement du modèle (doit être dans le même dossier que app.py)
    try:
        model = joblib.load('modele_final.pkl')
    except FileNotFoundError:
        st.error("❌ Fichier 'modele_final.pkl' introuvable. Veuillez l'ajouter au projet.")
    else:
        # Onglets
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistiques", "🧩 Corrélation", "📊 Visualisations", "🤖 Prédictions"])

        # ---- Statistiques
        with tab1:
            st.header("📈 Statistiques descriptives")
            st.write("Voici les statistiques (min, max, moyenne) des données :")
            stats = df.describe().loc[['min', 'max', 'mean']]
            st.dataframe(stats.style.format("{:.2f}"))

        # ---- Matrice de corrélation
        with tab2:
            st.header("🧩 Matrice de corrélation")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # ---- Visualisations
        with tab3:
            st.header("📊 Visualisation d'une variable")
            colonnes_numeriques = df.select_dtypes(include=np.number).columns
            colonne = st.selectbox("Choisir une colonne à visualiser", colonnes_numeriques)
            fig2, ax2 = plt.subplots()
            sns.histplot(df[colonne], kde=True, ax=ax2)
            ax2.set_title(f'Distribution de {colonne}')
            st.pyplot(fig2)

        # ---- Prédiction
        with tab4:
            st.header("🤖 Faire une prédiction")
            st.write("Entrer les valeurs pour chaque caractéristique :")

            colonnes = df.columns.tolist()[:-1]  # on suppose que la dernière colonne est la cible
            valeurs = []

            for col in colonnes:
                valeur = st.number_input(f"{col}", value=float(df[col].mean()))
                valeurs.append(valeur)

            if st.button("Prédire"):
                input_df = pd.DataFrame([valeurs], columns=colonnes)
                prediction = model.predict(input_df)[0]
                st.success(f"✅ Prédiction du modèle : {prediction:.2f}")

else:
    st.warning("📂 Veuillez importer un fichier CSV pour continuer.")

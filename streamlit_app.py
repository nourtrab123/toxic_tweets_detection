import streamlit as st
import pandas as pd
from datetime import datetime
from bert_model import predict_toxicity
import os

# --- Configuration de la page ---
st.set_page_config(
    page_title="Saisie des commentaires",
    layout="wide",
)

st.title("💬 Saisie de commentaire pour détection de toxicité")

# --- Zone de texte pour saisir le commentaire ---
comment = st.text_area("Écris ton commentaire ici :")

# --- Seuil pour considérer un commentaire toxique ---
TOXICITY_THRESHOLD = 50.0  # % de toxicité

# --- Bouton pour lancer l'analyse ---
if st.button("Analyser la toxicité"):
    if comment.strip() == "":
        st.warning("Veuillez saisir un commentaire.")
    else:
        # --- Prédiction ---
        score = predict_toxicity(comment)

        # --- Déterminer si le commentaire est toxique ---
        is_toxic = score >= TOXICITY_THRESHOLD
        label = "Toxique" if is_toxic else "Non toxique"

        st.success(f"Toxicité détectée : {score:.2f}% → {label}")

        # --- Chemin du fichier log ---
        LOG_FILE = "data/predictions_log.csv"

        # --- Lecture du fichier existant ou création d'un DataFrame vide ---
        if os.path.exists(LOG_FILE):
            df_log = pd.read_csv(LOG_FILE)
        else:
            df_log = pd.DataFrame(columns=["timestamp", "comment", "toxicity_score", "is_toxic"])

        # --- Ajout du nouveau commentaire ---
        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comment": comment,
            "toxicity_score": score,
            "is_toxic": is_toxic
        }
        df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)

        # --- Sauvegarde du fichier CSV ---
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        df_log.to_csv(LOG_FILE, index=False)

        st.info("Commentaire enregistré avec succès !")

# --- Optionnel : afficher les derniers commentaires ---
st.markdown("---")
st.subheader("📊 Derniers commentaires enregistrés")
if os.path.exists("data/predictions_log.csv"):
    df_display = pd.read_csv("data/predictions_log.csv")
    if not df_display.empty:
        st.dataframe(df_display.tail(10))  # affiche les 10 derniers
    else:
        st.info("Aucun commentaire enregistré pour le moment.")
else:
    st.info("Aucun commentaire enregistré pour le moment.")

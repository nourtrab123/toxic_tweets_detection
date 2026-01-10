# dashboard.py
import streamlit as st

st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)
st.title("📊 Dashboard des commentaires")
# … ton dashboard complet …

import pandas as pd
import altair as alt
from collections import Counter
import re

# --- Configuration de la page ---
st.set_page_config(
    page_title="Dashboard Toxicité des Commentaires",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard des Commentaires (Amélioré)")

# --- Charger les données ---
DATA_FILE = "comments_log.csv"

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.warning(f"Le fichier {DATA_FILE} n'a pas été trouvé.")
    st.stop()

# --- Vérifier colonne timestamp pour filtre temporel ---
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    min_date, max_date = df["timestamp"].min(), df["timestamp"].max()
else:
    min_date, max_date = None, None

# --- Sidebar : filtres utilisateurs ---
st.sidebar.header("🔍 Filtrer les commentaires")
min_score = st.sidebar.slider("Score toxicité minimum", 0, 100, 0)
max_score = st.sidebar.slider("Score toxicité maximum", 0, 100, 100)
category_filter = st.sidebar.multiselect(
    "Catégorie", options=df["category"].unique(), default=df["category"].unique()
)
keyword = st.sidebar.text_input("Filtrer par mot-clé")
if min_date:
    date_range = st.sidebar.date_input("Filtrer par date",
                                       [min_date.date(), max_date.date()],
                                       min_value=min_date.date(),
                                       max_value=max_date.date())

# --- Appliquer les filtres ---
filtered_df = df[(df["toxicity_score"] >= min_score) &
                 (df["toxicity_score"] <= max_score) &
                 (df["category"].isin(category_filter))]

if keyword.strip() != "":
    filtered_df = filtered_df[filtered_df["comment"].str.contains(keyword, case=False)]

if min_date:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df["timestamp"] >= start) & (filtered_df["timestamp"] <= end)]

# --- Calcul des statistiques avancées ---
total_comments = len(filtered_df)
num_toxic = len(filtered_df[filtered_df["category"]=="Toxique"])
num_non_toxic = len(filtered_df[filtered_df["category"]=="Non toxique"])
avg_score = filtered_df["toxicity_score"].mean() if total_comments>0 else 0
median_score = filtered_df["toxicity_score"].median() if total_comments>0 else 0
min_score_val = filtered_df["toxicity_score"].min() if total_comments>0 else 0
max_score_val = filtered_df["toxicity_score"].max() if total_comments>0 else 0

st.header("📈 Statistiques Globales")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total", total_comments)
col2.metric("Toxiques", f"{num_toxic} ({num_toxic/total_comments*100:.1f}%)" if total_comments>0 else "0")
col3.metric("Non-toxiques", f"{num_non_toxic} ({num_non_toxic/total_comments*100:.1f}%)" if total_comments>0 else "0")
col4.metric("Moyenne", f"{avg_score:.2f}%")
col5.metric("Médiane", f"{median_score:.2f}%")
col6.metric("Min / Max", f"{min_score_val:.0f}% / {max_score_val:.0f}%")

# --- Graphiques ---
st.header("📊 Visualisations")
# Histogramme score de toxicité
hist = alt.Chart(filtered_df).mark_bar().encode(
    alt.X("toxicity_score", bin=alt.Bin(maxbins=20), title="Score de toxicité"),
    y='count()',
    tooltip=["count()"]
).properties(width=600, height=300)

# Pie chart toxicité vs non-toxicité
pie = alt.Chart(filtered_df).mark_arc().encode(
    theta="count():Q",
    color="category:N",
    tooltip=["category:N", "count():Q"]
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribution des scores de toxicité")
    st.altair_chart(hist, use_container_width=True)

with col2:
    st.subheader("Répartition Toxique / Non toxique")
    st.altair_chart(pie, use_container_width=True)

# --- Top mots les plus fréquents dans les commentaires toxiques ---
st.header("🔥 Top mots dans les commentaires toxiques")
toxic_comments = filtered_df[filtered_df["category"]=="Toxique"]["comment"]
all_words = ' '.join(toxic_comments).lower()
words = re.findall(r'\b\w+\b', all_words)
most_common = Counter(words).most_common(10)
if most_common:
    top_words_df = pd.DataFrame(most_common, columns=["Mot", "Fréquence"])
    st.bar_chart(top_words_df.set_index("Mot"))
else:
    st.info("Aucun commentaire toxique filtré pour afficher les mots les plus fréquents.")

# --- Tableau des commentaires filtrés avec coloration ---
st.header("📝 Commentaires filtrés")
if total_comments == 0:
    st.info("Aucun commentaire ne correspond aux filtres.")
else:
    def color_row(row):
        if row["category"] == "Toxique":
            return ["background-color: #ff9999"]*len(row)
        else:
            return ["background-color: #99ff99"]*len(row)
    st.dataframe(filtered_df.style.apply(color_row, axis=1))

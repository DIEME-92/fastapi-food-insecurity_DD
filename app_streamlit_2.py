import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

###########################################################
# ✅ Chargement des modèles sauvegardés
###########################################################
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("modele_food_insecurity_D.pkl")
        return rf_model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

rf_model = load_models()

###########################################################
# ✅ Chargement des données
###########################################################
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_encoded_3.csv")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

df = load_data()
df_sample = df.sample(100) if not df.empty else pd.DataFrame()

if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données 'data_encoded_3.csv' : Echantillon de 100 observateurs")
    st.write(df_sample)

st.title("📊 Analyse exploratoire du dataset")
if not df.empty:
    st.subheader("📌 Statistiques descriptives")
    st.dataframe(df.describe().round(2))

variables = [
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q603_sauter_un_repas",
    "q601_ne_pas_manger_nourriture_saine_nutritive"
]

###########################################################
# 🔹 Matrice de corrélation
###########################################################
if not df.empty:
    st.subheader("📈 Matrice de corrélation des variables")
    fig, ax = plt.subplots(figsize=(20, 10))
    corr = df[variables].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

###########################################################
# 🔹 Histogrammes des variables
###########################################################
st.sidebar.subheader("📊 Sélection des variables à afficher")
vars_selectionnees = st.sidebar.multiselect("Choisissez les variables :", variables)
couleurs = sns.color_palette("husl", len(vars_selectionnees))

if vars_selectionnees and not df.empty:
    cols = st.columns(2)
    for index, (var, couleur) in enumerate(zip(vars_selectionnees, couleurs)):
        with cols[index % 2]:
            st.subheader(f"Histogramme : {var}")
            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=10, kde=True, color=couleur, ax=ax)
            ax.set_title(f"Distribution de : {var}")
            st.pyplot(fig)

###########################################################
# 🔹 Performances des modèles
###########################################################
rf_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.973172, 0.968635, 0.937269],
    "Test": [0.981092, 0.977833, 0.955665]
})

xgb_perf = rf_perf.copy()

st.sidebar.subheader("⚙️ Choix du modèle à afficher")
modele = st.sidebar.selectbox("Sélectionnez un modèle :", ["Random Forest", "XGBoost"])

perf = rf_perf if modele == "Random Forest" else xgb_perf
st.subheader(f"📋 Performance - {modele}")
st.dataframe(perf)

fig, ax = plt.subplots()
perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax,
    color=["#4CAF50", "#2196F3"] if modele=="Random Forest" else ["#FF9800", "#9C27B0"])
ax.set_title(f"{modele} - Performance")
st.pyplot(fig)

###########################################################
# 🔹 Formulaire de prédiction
###########################################################
st.title("🧠 Prédiction d'insécurité alimentaire")
q606 = st.number_input("Combien de fois avez-vous eu faim sans manger ?", 0, 10, 0)
q605 = st.number_input("Combien de fois avez-vous manqué de nourriture par manque d'argent ?", 0, 10, 0)
q604 = st.number_input("Combien de fois avez-vous mangé moins que nécessaire ?", 0, 10, 0)
q603 = st.number_input("Combien de repas avez-vous sauté aujourd'hui ?", 0, 10, 0)
q601 = st.number_input("Combien de fois avez-vous mangé une nourriture peu nutritive ?", 0, 10, 0)

if st.button("🔍 Lancer la prédiction"):
    payload = {
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q603_sauter_un_repas": q603,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601,
        "modele": modele
    }

    try:
        response = requests.post("https://ton-api.onrender.com/predict", json=payload)
        st.write("Status code:", response.status_code)
        st.write("Raw response:", response.text[:200])  # debug

        result = {}
        try:
            result = response.json()
        except ValueError:
            st.error("La réponse n'est pas du JSON valide.")
            st.stop()

        niveau = result.get("niveau", "inconnu")
        score = result.get("score", 0.00)
        profil = result.get("profil", "inconnu")
        probabilites = result.get("probabilités", {})

        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        else:
            st.success("🟢 Aucun signe d'insécurité alimentaire")

        st.write("### 🔎 Score de risque")
        st.progress(score)

        st.write(f"Profil détecté : **{profil.capitalize()}**")

        if probabilites:
            st.write("### 📊 Répartition des probabilités")
            fig, ax = plt.subplots()
            labels = list(probabilites.keys())
            sizes = list(probabilites.values())
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=['#4CAF50', '#FF9800'])
            ax.axis('equal')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")

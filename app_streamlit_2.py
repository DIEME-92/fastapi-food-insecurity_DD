import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

###########################################################"""chargement des modeles################################################################"

# ✅ Charger les modèles sauvegardés
@st.cache_resource
def load_models():
    rf_model = joblib.load("modele_food_insecurity_D.pkl")
    return rf_model

rf_model = load_models()

###################################################################################"chargement des donnees##########################################

# ✅ Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("data_encoded_4.csv")   # ⚠️ utilise ton dataset avec insecurité alimentaire
    return df

df = load_data()
df_sample = df.sample(100)

if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données 'data_encoded_4.csv' : Echantillon de 100 observateurs")
    st.write(df_sample)

st.title("📊 Analyse exploratoire du dataset")
st.subheader("📌 Statistiques descriptives")
st.dataframe(df.describe().round(2))

variables = [
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q603_sauter_un_repas",
    "q601_ne_pas_manger_nourriture_saine_nutritive"
]

############################################################################################################################################
# 🔹 Matrice de corrélation
st.subheader("📈 Matrice de corrélation des variables")
fig, ax = plt.subplots(figsize=(20, 10))
corr = df[variables].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

########################################
# 🔹 Histogrammes des variables
########################################
st.sidebar.subheader("📊 Sélection des variables à afficher")

vars_selectionnees = st.sidebar.multiselect("Choisissez les variables :", variables)
couleurs = sns.color_palette("husl", len(vars_selectionnees))

if vars_selectionnees:
    cols = st.columns(2)
    index = 0
    for var, couleur in zip(vars_selectionnees, couleurs):
        with cols[index % 2]:
            st.subheader(f"Histogramme : {var}")
            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=10, kde=True, color=couleur, ax=ax)
            ax.set_title(f"Distribution de : {var}")
            st.pyplot(fig)
        index += 1

########################################
# 🔹 Performances des modèles avec sélecteur
########################################

rf_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.973172, 0.968635, 0.937269],
    "Test": [0.981092, 0.977833, 0.955665]
})

xgb_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.973172, 0.968635, 0.937269],
    "Test": [0.981092, 0.977833, 0.955665]
})

st.sidebar.subheader("⚙️ Choix du modèle à afficher")
modele = st.sidebar.selectbox("Sélectionnez un modèle :", ["Random Forest", "XGBoost"])

if modele == "Random Forest":
    st.subheader("📋 Performance - Random Forest")
    st.dataframe(rf_perf)
    fig, ax = plt.subplots()
    rf_perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax, color=["#4CAF50", "#2196F3"])
    ax.set_title("Random Forest - Performance")
    st.pyplot(fig)

elif modele == "XGBoost":
    st.subheader("📋 Performance - XGBoost")
    st.dataframe(xgb_perf)
    fig, ax = plt.subplots()
    xgb_perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax, color=["#FF9800", "#9C27B0"])
    ax.set_title("XGBoost - Performance")
    st.pyplot(fig)

##########################################################################################################################################
# 🔹 Formulaire de prédiction individuelle
##########################################################################################################################################

st.title("🧠 Prédiction d'insécurité alimentaire")
q606 = st.number_input("Faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Manque de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Repas sautés aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

if st.button("🔍 Lancer la prédiction"):
    payload = {
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q603_sauter_un_repas": q603,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601,
        "modele": "rf_model"
    }
    try:
        response = requests.post("https://fastapi-food-insecurity-dd-1.onrender.com/predict", json=payload)
        response.raise_for_status()
        result = response.json()

        niveau = result.get("niveau", "inconnu")
        score = result.get("score", 0.00)
        profil = result.get("profil", "inconnu")
        probabilites = result.get("probabilités", {})

        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        elif niveau == "aucune":
            st.success("🟢 Aucun signe d'insécurité alimentaire")
        else:
            st.info("ℹ️ Niveau inconnu")

        st.write("### 🔎 Score de risque")
        st.progress(score)
        st.write(f"Profil détecté : **{profil.capitalize()}**")

        if probabilites:
            st.write("### 📊 Répartition des probabilités")
            fig, ax = plt.subplots()
            labels = ["Modérée", "Sévère"]
            sizes = [probabilites.get("classe_0", 0.0), probabilites.get("classe_1", 0.0)]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF9800'])
            ax.axis('equal')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")
        if 'response' in locals():
            st.text(f"Réponse brute : {response.text}")

##########################################################################################################################################
# 🔹 Prédiction agrégée par région
##########################################################################################################################################

st.sidebar.subheader("📊 Analyse par région")

if st.sidebar.button("Lancer la prédiction par région"):
    try:
        response = requests.post("https://fastapi-food-insecurity-dd-1.onrender.com/predict_by_region")
        response.raise_for_status()
        result = response.json()

        data = pd.DataFrame(result["predictions_par_region"])
        st.subheader("📊 Prévalence prédite par région")
        st.dataframe(data)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=data, x="q100_region", y="prediction", palette="viridis", ax=ax)
        ax.set_title("Prévalence d'insécurité alimentaire par région")
        ax.set_ylabel("Proportion prédite")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")
        if 'response' in locals():
            st.text(f"Réponse brute : {response.text}")

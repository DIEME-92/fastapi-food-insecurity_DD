import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

###########################################################
# 🔹 Chargement des modèles
###########################################################
@st.cache_resource
def load_models():
    rf_model = joblib.load("modele_food_insecurity_D1.pkl")   # ⚠️ doit être entraîné avec 5 variables
    xgb_model = joblib.load("modele_xgboost.pkl")             # ⚠️ doit être entraîné avec 5 variables
    return {"RandomForest": rf_model, "XGBoost": xgb_model}

models = load_models()

###########################################################
# 🔹 Chargement des données
###########################################################
@st.cache(persist=True)
def load_data():
    df = pd.read_csv("data_encoded_3.csv")
    return df

df = load_data()
df_sample = df.sample(100)

if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données 'data_encoded_3.csv' : Echantillon de 100 observateurs")
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

###########################################################
# 🔹 Matrice de corrélation
###########################################################
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

###########################################################
# 🔹 Performances des modèles
###########################################################
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
modele_perf = st.sidebar.selectbox("Sélectionnez un modèle :", ["RandomForest", "XGBoost"])

if modele_perf == "RandomForest":
    st.subheader("📋 Performance - Random Forest")
    st.dataframe(rf_perf)
    fig, ax = plt.subplots()
    rf_perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax, color=["#4CAF50", "#2196F3"])
    ax.set_title("Random Forest - Performance")
    st.pyplot(fig)

elif modele_perf == "XGBoost":
    st.subheader("📋 Performance - XGBoost")
    st.dataframe(xgb_perf)
    fig, ax = plt.subplots()
    xgb_perf.set_index("Métrique")[["Train","Test"]].plot(kind="bar", ax=ax, color=["#FF9800", "#9C27B0"])
    ax.set_title("XGBoost - Performance")
    st.pyplot(fig)

###########################################################
# 🔹 Formulaire de prédiction
###########################################################
st.title("🧠 Prédiction d'insécurité alimentaire")

modele_pred = st.selectbox("Choisissez le modèle pour la prédiction :", ["RandomForest", "XGBoost"])
model = models[modele_pred]

q606 = st.number_input("Combien de fois avez-vous eu faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Combien de fois avez-vous manqué de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Combien de fois avez-vous mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Combien de repas avez-vous sauté aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Combien de fois avez-vous mangé une nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

if st.button("🔍 Lancer la prédiction"):
    input_df = pd.DataFrame([{
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q603_sauter_un_repas": q603,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601
    }])

    # ✅ Les deux modèles utilisent les 5 variables
    selected_features = [
        "q604_manger_moins_que_ce_que_vous_auriez_du",
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
        "q606_1_avoir_faim_mais_ne_pas_manger",
        "q603_sauter_un_repas",
        "q601_ne_pas_manger_nourriture_saine_nutritive"
    ]
    input_filtered = input_df[selected_features]

    try:
        if input_filtered.sum().sum() == 0:
            st.write("### 🟢 Aucun signe d'insécurité alimentaire (Neutre)")
            st.write("📊 Score de risque : 0.00")
            st.progress(0.0)
        else:
            proba = model.predict_proba(input_filtered)[0]

            seuil_severe = 0.5
            if proba[1] >= seuil_severe:
                niveau = "sévère"
                couleur = "🔴"
            else:
                niveau = "modérée"
                couleur = "🟠"

            st.write(f"### {couleur} Niveau d'insécurité alimentaire : {niveau.capitalize()}")
            st.write(f"📊 Score de risque : {round(float(proba[1]), 4)}")
            st.progress(float(proba[1]))

            # ✅ Pie chart
            st.write("### 📊 Répartition des probabilités")
            fig, ax = plt.subplots()
            labels = ["Modérée", "Sévère"]
            sizes = [proba[0], proba[1]]
            colors = ['#4CAF50', '#FF9800']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

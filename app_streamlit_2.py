import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ✅ Charger les modèles
@st.cache_resource
def load_models():
    rf_model = joblib.load("modele_food_insecurity_D1.pkl")
    xgb_model = joblib.load("modele_xgboost.pkl")  # Assure-toi d'avoir sauvegardé ce fichier
    return {"RandomForest": rf_model, "XGBoost": xgb_model}

models = load_models()

# ✅ Interface Streamlit
st.title("🧠 Prédiction d'insécurité alimentaire")

# Choix du modèle
model_choice = st.selectbox("⚙️ Choix du modèle à afficher", list(models.keys()))
model = models[model_choice]

# ✅ Comparaison des performances
st.subheader("📋 Comparaison des performances des modèles")
perf_data = {
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "RandomForest (Test)": [0.95, 0.94, 0.92],  # ⚠️ Remplace par tes vraies métriques
    "XGBoost (Test)": [0.9811, 0.9778, 0.9557]  # ⚠️ Valeurs issues de ton PDF
}
perf_df = pd.DataFrame(perf_data)
st.table(perf_df)

# Variables d'entrée
st.subheader("🧾 Données d'entrée")
q606 = st.number_input("Combien de fois avez-vous eu faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Combien de fois avez-vous manqué de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Combien de fois avez-vous mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Combien de repas avez-vous sauté aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Combien de fois avez-vous mangé une nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

if st.button("🔍 Lancer la prédiction"):
    # Créer un DataFrame avec les variables
    input_df = pd.DataFrame([{
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q603_sauter_un_repas": q603,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601
    }])

    selected_features = [
        "q604_manger_moins_que_ce_que_vous_auriez_du",
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
        "q606_1_avoir_faim_mais_ne_pas_manger"
    ]
    input_filtered = input_df[selected_features]

    try:
        # ✅ Prédiction
        proba = model.predict_proba(input_filtered.values)[0]
        seuil_severe = 0.4
        prediction_binaire = int(proba[1] > seuil_severe)

        if input_filtered.sum().sum() == 0:
            niveau = "aucune"
            profil = "neutre"
        else:
            niveau = "sévère" if prediction_binaire == 1 else "modérée"
            profil = "critique" if prediction_binaire == 1 else "intermédiaire"

        st.write(f"### 🔴 Niveau d'insécurité alimentaire : {niveau.capitalize()}")
        st.write(f"🔎 Profil détecté : {profil}")
        st.write(f"📊 Score de risque : {round(float(proba[1]), 4)}")

        # ✅ Affichage des probabilités en bar chart
        st.bar_chart({"Modérée": [proba[0]], "Sévère": [proba[1]]})

        # ✅ Explicabilité avec SHAP
        explainer = shap.Explainer(model, input_filtered)
        shap_values = explainer(input_filtered)

        st.write("📌 Explication des variables (SHAP)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

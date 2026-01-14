import streamlit as st
import requests
import matplotlib.pyplot as plt

# ✅ Titre de l'application
st.title("🧠 Prédiction d'insécurité alimentaire")

st.write("Cette application permet de prédire le niveau d'insécurité alimentaire à partir de quelques variables clés.")

# ✅ Formulaire de saisie
q606 = st.number_input("Faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Manque de nourriture par manque d'argent ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Repas sautés aujourd'hui ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

# ✅ Bouton de prédiction
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
        # ⚠️ Mets ici l’URL de ton API FastAPI (local ou Render)
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        response.raise_for_status()
        result = response.json()

        niveau = result.get("niveau", "inconnu")
        score = result.get("score", 0.00)
        profil = result.get("profil", "inconnu")
        probabilites = result.get("probabilités", {})

        # ✅ Affichage du niveau
        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        elif niveau == "aucune":
            st.success("🟢 Aucun signe d'insécurité alimentaire")
        else:
            st.info("ℹ️ Niveau inconnu")

        # ✅ Score
        st.write("### 🔎 Score de risque")
        st.progress(score)
        st.write(f"Profil détecté : **{profil.capitalize()}**")

        # ✅ Probabilités en camembert
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

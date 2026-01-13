##########################################################################################################################################
# 🔹 Prédiction agrégée par région
##########################################################################################################################################

st.sidebar.subheader("📊 Analyse par région")

if st.sidebar.button("Lancer la prédiction par région"):
    try:
        # ⚠️ Mets ici l’URL correcte de ton API (local ou Render)
        response = requests.post("https://fastapi-food-insecurity-dd-1.onrender.com/predict_by_region")
        response.raise_for_status()
        result = response.json()

        # Convertir en DataFrame
        data = pd.DataFrame(result["predictions_par_region"])

        # ✅ Affichage tableau
        st.subheader("📊 Prévalence prédite par région")
        st.dataframe(data)

        # ✅ Affichage graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=data, x="region", y="prediction", palette="viridis", ax=ax)
        ax.set_title("Prévalence d'insécurité alimentaire par région")
        ax.set_ylabel("Proportion prédite")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")
        if 'response' in locals():
            st.text(f"Réponse brute : {response.text}")

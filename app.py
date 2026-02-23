from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

# Initialisation de l'application
app = FastAPI()

# Charger directement le modèle sauvegardé avec joblib.dump(rf_model, ...)
rf_model = joblib.load("modele_food_insecurity.pkl")
rf_model = joblib.load("modele_food_insecurity_xgb1.pkl")


# Variables utilisées pour la prédiction individuelle
selected_features = [
    "604_manger_moins_que_ce_que_vous_auriez_du",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q607_1_passer_toute_une_journee_sans_manger"
    
]

# Schéma d'entrée pour prédiction individuelle
class InputData(BaseModel):
    q606_1_avoir_faim_mais_ne_pas_manger: int
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: int
    q604_manger_moins_que_ce_que_vous_auriez_du: int
    q603_sauter_un_repas: int
    q601_ne_pas_manger_nourriture_saine_nutritive: int
    modele: str = "rf_model"

# Endpoint de santé
@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}

# Endpoint de prédiction individuelle
@app.post("/predict")
def predict(data: InputData):
    try:
        # Transformer les données en DataFrame
        input_df = pd.DataFrame([data.dict()])
        input_filtered = input_df[selected_features]

        # Prédiction
        proba = rf_model.predict_proba(input_filtered)[0]

        seuil_severe = 0.4
        prediction_binaire = int(proba[1] > seuil_severe)

        # Déterminer niveau et profil
        if input_filtered.sum().sum() == 0:
            niveau = "aucune"
            profil = "neutre"
        else:
            niveau = "sévère" if prediction_binaire == 1 else "modérée"
            profil = "critique" if prediction_binaire == 1 else "intermédiaire"

        # Réponse JSON
        return JSONResponse(content={
            "prediction": prediction_binaire,
            "niveau": niveau,
            "profil": profil,
            "score": round(float(proba[1]), 4),
            "probabilités": {
                "classe_0": round(float(proba[0]), 4),
                "classe_1": round(float(proba[1]), 4)
            }
        })

    except Exception as e:
        return JSONResponse(content={
            "error": "Une erreur est survenue",
            "details": str(e)
        }, status_code=500)






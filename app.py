from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

# ✅ Initialisation de l'application
app = FastAPI()

# ✅ Charger le modèle RandomForest
rf_model = joblib.load("modele_food_insecurity_D.pkl")

# ✅ Variables utilisées pour la prédiction individuelle ET par région
selected_features = [
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q606_1_avoir_faim_mais_ne_pas_manger"
]

# ✅ Schéma d'entrée pour prédiction individuelle
class InputData(BaseModel):
    q606_1_avoir_faim_mais_ne_pas_manger: int
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: int
    q604_manger_moins_que_ce_que_vous_auriez_du: int
    q603_sauter_un_repas: int
    q601_ne_pas_manger_nourriture_saine_nutritive: int
    modele: str = "rf_model"   # valeur par défaut

# ✅ Endpoint de santé
@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}

# ✅ Endpoint de prédiction individuelle
@app.post("/predict")
def predict(data: InputData):
    try:
        # Transformer les données en DataFrame
        input_df = pd.DataFrame([data.dict()])
        input_filtered = input_df[selected_features]

        # Toujours utiliser le modèle RF
        proba = rf_model.predict_proba(input_filtered)[0]

        # Seuil de classification
        seuil_severe = 0.4
        prediction_binaire = int(proba[1] > seuil_severe)

        # Déterminer niveau et profil
        if input_filtered.sum().sum() == 0:
            niveau = "aucune"
            profil = "neutre"
        else:
            niveau = "sévère" if prediction_binaire == 1 else "modérée"
            profil = "critique" if prediction_binaire == 1 else "intermédiaire"

        # ✅ Réponse JSON
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

# ✅ Endpoint de prédiction par région
@app.post("/predict_by_region")
def predict_by_region():
    try:
        # Charger les données complètes
        df = pd.read_csv("data_encoded_4.csv")

        # Vérifie que les colonnes nécessaires existent
        for col in selected_features + ["q100_region"]:
            if col not in df.columns:
                return JSONResponse(content={
                    "error": f"La colonne '{col}' est absente du dataset",
                    "colonnes_disponibles": df.columns.tolist()
                }, status_code=400)

        # Sélectionner uniquement les colonnes utilisées par le modèle
        X = df[selected_features]

        # Prédictions
        y_pred = rf_model.predict(X)
        df["prediction"] = y_pred

        # Agrégation par région
        resultats_region = (
            df.groupby("q100_region")["prediction"]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        )

        # ✅ Réponse JSON
        return JSONResponse(content={
            "predictions_par_region": resultats_region
        })

    except Exception as e:
        return JSONResponse(content={
            "error": "Une erreur est survenue lors de la prédiction par région",
            "details": str(e)   # ✅ affiche l'erreur exacte pour déboguer
        }, status_code=500)

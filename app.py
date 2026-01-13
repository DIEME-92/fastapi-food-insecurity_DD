from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import pandas as pd

app = FastAPI()

# Charger le modèle
rf_model = joblib.load("modele_food_insecurity_D.pkl")

@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}

# 🔹 Endpoint de prédiction par région
@app.post("/predict_by_region")
def predict_by_region():
    try:
        df = pd.read_csv("data_encoded_4.csv")

        # Vérifie que les colonnes existent
        if "q100_region" not in df.columns:
            return JSONResponse(content={
                "error": "La colonne 'q100_region' est absente du dataset",
                "colonnes_disponibles": df.columns.tolist()
            }, status_code=400)

        if "insecurite_alimentaire" in df.columns:
            X = df.drop(columns=["insecurite_alimentaire", "q100_region"])
        else:
            X = df.drop(columns=["q100_region"])

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

        return JSONResponse(content={"predictions_par_region": resultats_region})

    except Exception as e:
        return JSONResponse(content={
            "error": "Une erreur est survenue lors de la prédiction par région",
            "details": str(e)
        }, status_code=500)

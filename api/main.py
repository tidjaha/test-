from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

# Charger les modèles
model = joblib.load('model randomforest.pkl')
scaler = joblib.load('scale des notes.pkl')
scaler0 = joblib.load("scale du nbslibing.pkl")
encoder0 = joblib.load("encoder de la colone ethni.pkl")
encoder1 = joblib.load("encoder de la colone parent educ.pkl")
encoder2 = joblib.load("encoder de la colone parent marital status.pkl")
encoder3 = joblib.load("encoder de la colone practice sport.pkl")
encoder4 = joblib.load("encoder de la colone wkly study hours.pkl")

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    try:
        # Prétraitement des données
        input_data = pd.DataFrame({
            "EthnicGroup": [encoder0.transform([data["EthnicGroup"]])[0]],
            "ParentEduc": [encoder1.transform([data["ParentEduc"]])[0]],
            "ParentMaritalStatus": [encoder2.transform([data["ParentMaritalStatus"]])[0]],
            "PracticeSport": [encoder3.transform([data["PracticeSport"]])[0]],
            "NrSiblings": [scaler0.transform([[data["NrSiblings"]]])[0][0]],
            "WklyStudyHours": [encoder4.transform([data["WklyStudyHours"]])[0]],
            "Gender_male": [data["Gender_male"]],
            "LunchType_standard": [data["LunchType_standard"]],
            "TestPrep_none": [data["TestPrep_none"]],
            "IsFirstChild_yes": [data["IsFirstChild_yes"]],
            "TransportMeans_school_bus": [data["TransportMeans_school_bus"]],
        })

        # Prédictions
        prediction = model.predict(input_data)
        prediction = scaler.inverse_transform(prediction)

        return {
            "math": round(prediction[0][0], 2),
            "lecture": round(prediction[0][1], 2),
            "ecriture": round(prediction[0][2], 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

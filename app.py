from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

# Charger le modèle à partir du fichier
filename = 'finalized_model.sav'
model = joblib.load(filename)

# Définir un modèle de données pour l'API
class PatientData(BaseModel):
    preg: float
    plas: float
    pres: float
    skin: float
    test: float
    mass: float
    pedi: float
    age: float

# Initialiser l'application FastAPI
app = FastAPI()

@app.post("/predict/")
def predict(data: PatientData):
    # Convertir les données reçues en un format utilisable par le modèle
    input_data = np.array([[data.preg, data.plas, data.pres, data.skin, data.test, data.mass, data.pedi, data.age]])
    
    # Faire la prédiction
    prediction = model.predict(input_data)

    # Log des prédictions avec MLflow
    with mlflow.start_run():
        mlflow.log_metric("prediction_value", prediction[0])

    # Renvoi de la prédiction
    return {"prediction": int(prediction[0])}

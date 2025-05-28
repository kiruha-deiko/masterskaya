from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
import joblib
from typing import List
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Создание FastAPI-приложения с понятным описанием
app = FastAPI(
    title="💓 Heart Attack Risk Predictor",
    description="Загрузите CSV-файл с данными пациентов, чтобы получить предсказания риска сердечного приступа.",
    version="1.0.0"
)

# Загрузка артефактов модели
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder_gender.pkl")

# Схема для валидации ответа
class PredictionResponse(BaseModel):
    id: int
    prediction: int

@app.post("/predict", response_model=List[PredictionResponse], summary="🔍 Предсказание по CSV", tags=["Предсказания"])
async def predict(file: UploadFile = File(..., description="CSV-файл с данными пациентов (должен содержать колонку 'id')")):
    try:
        df = pd.read_csv(file.file)

        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        # Заполнение пропусков
        binary_features = [
            "Diabetes", "Family History", "Smoking", "Obesity",
            "Previous Heart Problems", "Medication Use"
        ]
        for col in binary_features:
            df[col].fillna(0, inplace=True)

        df["Alcohol Consumption"].fillna(df["Alcohol Consumption"].median(), inplace=True)
        df["Stress Level"].fillna(df["Stress Level"].median(), inplace=True)
        df["Physical Activity Days Per Week"].fillna(df["Physical Activity Days Per Week"].mode()[0], inplace=True)

        df["Gender"] = label_encoder.transform(df["Gender"])

        features = df.drop(columns=["id"])
        features_scaled = scaler.transform(features)

        preds = model.predict(features_scaled)

        return [PredictionResponse(id=row_id, prediction=int(pred)) for row_id, pred in zip(df["id"], preds)]

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
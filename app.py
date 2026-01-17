# making the app file
from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd
import boto3
from src.features.build_features import build_features
import os


class PredictionInput(BaseModel):
    student_id: int
    age: int
    gender: str
    course: str
    study_hours: float
    class_attendance: float
    internet_access: str
    sleep_hours: float
    sleep_quality: str
    study_method: str
    facility_rating: str
    exam_difficulty: str

BUCKET = "students-mlops"
PREFIX = "models/creditcard/v1/"
MODEL_DIR = "models"

FILES = [
    "model.joblib",
    "feature_columns.joblib"
]

def load_artifact():
    os.makedirs(MODEL_DIR, exist_ok=True)

    s3 = boto3.client('s3')

    for f in FILES:
        local_path = f'{MODEL_DIR}/{f}'

        if not os.path.exists(local_path):
            s3.download_file(
                BUCKET,
                PREFIX + f,
                local_path
            )

    model = joblib.load(f"{MODEL_DIR}/model.joblib")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.joblib")

    return model, feature_columns



app = FastAPI()

@app.get('/')
def home():
    return 'working'

@app.post('/predict')
def predict(input_data: PredictionInput):
    features = [
        input_data.student_id,
        input_data.age,
        input_data.gender,
        input_data.course,
        input_data.study_hours,
        input_data.class_attendance,
        input_data.internet_access,
        input_data.sleep_hours,
        input_data.sleep_quality,
        input_data.study_method,
        input_data.facility_rating,
        input_data.exam_difficulty,
    ]
    df = pd.DataFrame([input_data.dict()])
    # print(df)
    df_transformed = build_features(df)
    # print(df_transformed)
    model ,feature_columns = load_artifact()

    df_transformed = df_transformed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(df_transformed)

    result = int(prediction[0])

    return {'prediction':result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

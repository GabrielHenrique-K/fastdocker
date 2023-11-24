from fastapi import FastAPI
from predict import make_prediction

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    predictions = make_prediction(data)
    return {"predictions": predictions}

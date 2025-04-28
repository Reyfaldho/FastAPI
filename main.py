from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd


app = FastAPI(title="COVID-19 Impact Prediction API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


class CovidData_final(BaseModel):
    Country: str
    Deaths: int
    Recovered: int
    Active: int
    Newcases: int
    Newdeaths: int
    Newrecovered: int
    Confirmedlastweek: int
    Deathsper100: float
    WHORegionCode: int

def classify_risk(Deaths: int):
    if Deaths > 5000:
        return "High Risk"
    else:
        return "Low Risk"

def preprocess_input(data: CovidData_final):
    df = pd.DataFrame([{
        "Deaths": data.Deaths,  
        "Recovered": data.Recovered,
        "Active": data.Active,
        "Newcases": data.Newcases,
        "Newdeaths": data.Newdeaths,
        "Newrecovered": data.Newrecovered,
        "Confirmedlastweek": data.Confirmedlastweek,
        "Deathsper100": data.Deathsper100,
        "WHORegionCode": data.WHORegionCode
    }])

    # Normalisasi data
    df_scaled = scaler.transform(df)
    return df_scaled


@app.get("/")
def read_root():
    return {"message": "COVID-19 Impact Prediction API is running"}


@app.post("/predict")
def predict_impact(data: CovidData_final):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    result = "Severe Impact" if prediction == 1 else "Low Impact"

    risk_level = classify_risk(data.Deaths)
    
    return {
        "Country": data.Country,
        "prediction": int(prediction),
        "result": result,
        "risk_level": risk_level
    }

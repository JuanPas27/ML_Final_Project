from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="Dashboard GTZAN Models")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/api/metrics")
def get_metrics():
    try:
        with open("models/metrics_base.json", "r") as f:
            base_metrics = json.load(f)
        with open("models_boost/metrics_boost.json", "r") as f:
            boost_metrics = json.load(f)
        return {"base": base_metrics, "boost": boost_metrics}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Execute models first")

@app.get("/api/confusion/{model_name}")
def get_confusion(model_name: str):
    base_path = f"models/confusion_{model_name}.json"
    boost_path = f"models_boost/confusion_{model_name}_boost.json"
    
    base_cm, boost_cm = [], []
    
    if os.path.exists(base_path):
        with open(base_path, "r") as f:
            base_cm = json.load(f)
    if os.path.exists(boost_path):
        with open(boost_path, "r") as f:
            boost_cm = json.load(f)
            
    return {"base": base_cm, "boost": boost_cm}
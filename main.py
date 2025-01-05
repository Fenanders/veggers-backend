from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from plant_recommendation import PlantRecommendationSystem 
app = FastAPI()
plant_system = PlantRecommendationSystem()

class EnvironmentConditions(BaseModel):
    temperature: float
    humidity: float
    uv: float

class RecommendationResponse(BaseModel):
    recommended_plants: List[str]
    soil_moisture: str

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(conditions: EnvironmentConditions):
    current_conditions = [
        conditions.temperature,
        conditions.humidity,
        conditions.uv
    ]
    
    recommended_plants = plant_system.recommend_plants(current_conditions)
    soil_moisture = plant_system.predict_soil_moisture(current_conditions)
    
    return RecommendationResponse(
        recommended_plants=recommended_plants,
        soil_moisture=soil_moisture
    )

if __name__ == "__main__":
    # Load trained model if you have one
    try:
        plant_system.load_modeel('plant_model.pkl')
    except:
        # Train with sample data if no model exists
        # Your training code here
        pass
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

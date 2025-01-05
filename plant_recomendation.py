import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle

class PlantRecommendationSystem:
    def __init__(self):
        self.soil_classifier = GaussianNB()
        self.plant_conditions = {
            'tomato': {
                'temp_range': (20, 30),
                'humidity_range': (60, 80),
                'uv_range': (0.3, 0.7)
            },
            'bok choy': { 
                'temp_range': (15, 22),
                'humidity_range': (60, 70),
                'uv_range': (0.2, 0.5)
            },
            'choy sum': {
                'temp_range': (18, 24),
                'humidity_range': (65, 75),
                'uv_range': (0.3, 0.6)
            },
            'spinach': {
                'temp_range': (16, 24),
                'humidity_range': (45, 65),
                'uv_range': (0.1, 0.4)
            }
        }
        
    def train_soil_classifier(self, X_train, y_train):
        """
        Train the soil moisture classifier
        X_train: array of [temp, humidity, uv] measurements
        y_train: array of soil moisture labels ('D', 'M', 'W')
        """
        self.soil_classifier.fit(X_train, y_train)
        
    def predict_soil_moisture(self, conditions):
        """
        Predict soil moisture based on environmental conditions
        conditions: array of [temp, humidity, uv]
        Returns: predicted soil moisture ('D', 'M', 'W')
        """
        return self.soil_classifier.predict([conditions])[0](citation_0)
    
    def recommend_plants(self, conditions):
        """
        Recommend suitable plants based on current conditions
        conditions: array of [temp, humidity, uv]
        Returns: list of recommended plants
        """
        temp, humidity, uv = conditions
        recommended = []
        
        for plant, ranges in self.plant_conditions.items():
            if (ranges['temp_range'][0](citation_0) <= temp <= ranges['temp_range'][1](citation_1) and
                ranges['humidity_range'][0](citation_0) <= humidity <= ranges['humidity_range'][1](citation_1) and
                ranges['uv_range'][0](citation_0) <= uv <= ranges['uv_range'][1](citation_1)):
                recommended.append(plant)
                
        return recommended
    
    def save_model(self, filename):
        """Save the trained model to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.soil_classifier, f)
            
    def load_model(self, filename):
        """Load a trained model from a file"""
        with open(filename, 'rb') as f:
            self.soil_classifier = pickle.load(f)

# Example usage
if __name__ == "__main__":
    # Create sample training data
    X_train = np.array([
        [25, 75, 0.3],  # temp, humidity, uv
        [30, 60, 0.6],
        [20, 80, 0.2],
        [28, 70, 0.5]
    ])
    
    y_train = np.array(['M', 'D', 'W', 'M'])  # soil moisture labels
    
    # Initialize and train the system
    plant_system = PlantRecommendationSystem()
    plant_system.train_soil_classifier(X_train, y_train)
    
    # Test the system
    current_conditions = [25, 75, 0.3]
    
    # Predict soil moisture
    predicted_moisture = plant_system.predict_soil_moisture(current_conditions)
    print(f"Predicted soil moisture: {predicted_moisture}")
    
    # Get plant recommendations
    recommended_plants = plant_system.recommend_plants(current_conditions)
    print(f"Recommended plants: {recommended_plants}")
    
    # Save the model
    plant_system.save_model('plant_model.pkl')

import joblib
from app.features import extract_features

# load trained model
model = joblib.load("model.pkl")

# change this to your audio file
file_path = "sample.wav"

features = extract_features(file_path)
prediction = model.predict([features])

print("Predicted Genre:", prediction[0])

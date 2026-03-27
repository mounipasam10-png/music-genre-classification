import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from app.features import extract_features

DATASET_PATH = "data"

X = []
y = []

genres = os.listdir(DATASET_PATH)

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    
    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)

        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(genre)
        except:
            continue

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model.pkl")
print("Model saved!")

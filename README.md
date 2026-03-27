# Music Genre Classification

A machine learning project for classifying music into genres using audio feature extraction and SVM.

## Features
- MFCC feature extraction from audio files
- Audio preprocessing using Librosa
- Genre classification using SVM
- Model saving and prediction support

## Tech Stack
- Python
- Librosa
- NumPy
- Scikit-learn
- TensorFlow

## Project Structure
music-genre-classification/
├── app/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── data/
├── requirements.txt
└── README.md

## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python app/train.py

Predict genre from a sample file:

python app/predict.py

import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    return mfccs_mean

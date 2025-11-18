"""
Command-line interface to predict the genre of a single audio file.
"""

import argparse
from pathlib import Path

import joblib
from extract_features import extract_feature_vector


MODEL_PATH = Path("models/knn_genre_classifier.joblib")
LABELS_PATH = Path("models/label_encoder.joblib")


def main():
    parser = argparse.ArgumentParser(description="Predict music genre from an audio file.")
    parser.add_argument("audio_file", type=str, help="Path to .wav file")
    args = parser.parse_args()

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise SystemExit("Model and label encoder not found. Please run train_knn.py first.")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABELS_PATH)

    features = extract_feature_vector(args.audio_file).reshape(1, -1)
    prediction_encoded = model.predict(features)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    print(f"🎶 Predicted genre: {prediction_label}")


if __name__ == "__main__":
    main()

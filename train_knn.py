"""
Train a KNN-based music genre classifier.

This script expects audio files organized in subfolders under data/raw,
where each subfolder name is treated as the genre label.
"""

import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from extract_features import extract_feature_vector

DATA_ROOT = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def collect_files_and_labels(root: Path) -> Tuple[List[str], List[str]]:
    file_paths: List[str] = []
    labels: List[str] = []

    for genre_dir in root.iterdir():
        if not genre_dir.is_dir():
            continue

        genre = genre_dir.name
        for file in genre_dir.glob("*.wav"):
            file_paths.append(str(file))
            labels.append(genre)

    return file_paths, labels


def build_dataset(file_paths: List[str], labels: List[str]):
    X = []
    y = []

    for path, label in zip(file_paths, labels):
        try:
            features = extract_feature_vector(path)
            X.append(features)
            y.append(label)
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")

    return np.array(X), np.array(y)


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(
            f"Data directory '{DATA_ROOT}' does not exist. "
            "Please place your dataset under data/raw/<genre>/."
        )

    files, labels = collect_files_and_labels(DATA_ROOT)
    print(f"Found {len(files)} audio files.")

    X, y = build_dataset(files, labels)
    print(f"Dataset size after feature extraction: {X.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=7, metric="minkowski")),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model_path = MODEL_DIR / "knn_genre_classifier.joblib"
    labels_path = MODEL_DIR / "label_encoder.joblib"

    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, labels_path)

    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {labels_path}")


if __name__ == "__main__":
    main()

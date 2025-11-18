# Music Genre Classifier 🎵

This project builds a **K-Nearest Neighbors (KNN)** classifier to predict the **genre of a music track**
based on extracted audio features (MFCC and spectral features).

## Features

- Extracts robust audio features using `librosa`
- Trains a configurable KNN classifier
- Command-line prediction script
- Easily extendable to other models (SVM, RandomForest, etc.)

## Tech Stack

- Python
- Librosa
- scikit-learn
- NumPy

## Project Structure

```text
music-genre-classifier/
├── data/
│   └── raw audio files organized by genre
├── models/
│   └── saved model + label encoder
└── src/
    ├── extract_features.py
    ├── train_knn.py
    └── predict_cli.py

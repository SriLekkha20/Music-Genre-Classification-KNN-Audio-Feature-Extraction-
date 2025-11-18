"""
Audio feature extraction utilities.

This module provides a single function `extract_feature_vector`
that converts an audio file into a 1D numeric feature vector.
"""

from typing import Tuple

import librosa
import numpy as np


def extract_feature_vector(
    file_path: str,
    target_sr: int = 22050,
    max_duration: float = 30.0,
) -> np.ndarray:
    """
    Load an audio file and compute a compact feature representation.

    Features used:
    - MFCC (mean over time)
    - Spectral centroid (mean + std)
    - Spectral bandwidth (mean + std)

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    target_sr : int, optional
        Sampling rate to resample the audio to.
    max_duration : float, optional
        Maximum duration (in seconds) to load.

    Returns
    -------
    np.ndarray
        1D feature vector.
    """
    y, sr = librosa.load(file_path, sr=target_sr, duration=max_duration, mono=True)

    if y.size == 0:
        raise ValueError(f"Empty audio signal for file: {file_path}")

    # MFCChttps://github.com/SriLekkha20/Music-Genre-Classification-KNN-Audio-Feature-Extraction-/tree/main
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = centroid.mean()
    centroid_std = centroid.std()

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = bandwidth.mean()
    bandwidth_std = bandwidth.std()

    feature_vector = np.concatenate(
        [
            mfcc_mean,
            np.array([centroid_mean, centroid_std, bandwidth_mean, bandwidth_std]),
        ]
    )

    return feature_vector

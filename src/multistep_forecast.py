from __future__ import annotations

"""Solution (mentor-only) for Task 1 — One-step time series forecasting with LSTM."""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_windows(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError("series must be 1D (shape: (T,))")
    if not (1 <= window < len(series)):
        raise ValueError("window must satisfy 1 <= window < len(series)")

    X = []
    y = []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.float32)[..., None]
    return X, y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must satisfy: train_frac > 0, val_frac >= 0, train_frac + val_frac < 1")

    n = len(X)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Split results in empty train/val/test. Adjust fractions or provide more data.")

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(
    window: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = Sequential(
        [
            LSTM(n_units, input_shape=(window, 1)),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def train_model(
    series: np.ndarray,
    window: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
):
    tf.keras.utils.set_random_seed(seed)

    X, y = make_windows(series, window)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y, train_frac, val_frac)

    model = build_model(window)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
        callbacks=callbacks,
    )

    return model, X_test, y_test, history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test, verbose=0)
    return {"mae": mae(y_test, y_pred), "rmse": rmse(y_test, y_pred)}

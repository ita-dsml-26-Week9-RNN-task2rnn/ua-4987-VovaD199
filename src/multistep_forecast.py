from __future__ import annotations

"""Solution (mentor-only) for Task 2 — multi-step forecasting strategies."""

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


def make_windows(series: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if window < 1 or window >= len(series):
        raise ValueError("window must satisfy 1 <= window < len(series)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(series) - window - horizon + 1 <= 0:
        raise ValueError("series too short for given window and horizon")

    X, y = [], []
    T = len(series)
    for t in range(0, T - window - horizon + 1):
        X.append(series[t : t + window])
        y.append(series[t + window : t + window + horizon])

    X = np.array(X, dtype=np.float32)[..., None]
    y = np.array(y, dtype=np.float32)
    if horizon == 1:
        y = y[..., None]  # (N,1)
    return X, y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must satisfy: train_frac > 0, val_frac >= 0, train_frac + val_frac < 1")

    n = len(X)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Split results in empty train/val/test")

    return (X[:train_end], y[:train_end]), (X[train_end:val_end], y[train_end:val_end]), (X[val_end:], y[val_end:])


def build_model(
    window: int,
    output_dim: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = Sequential([
        LSTM(n_units, input_shape=(window, 1)),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(output_dim),
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def train_model(
    series: np.ndarray,
    window: int,
    horizon: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
):
    tf.keras.utils.set_random_seed(seed)

    X, y = make_windows(series, window=window, horizon=horizon)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y, train_frac, val_frac)

    model = build_model(window=window, output_dim=horizon)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
        callbacks=callbacks,
    )

    return model, X_test, y_test


def recursive_rollout_one_step(model: tf.keras.Model, init_window: np.ndarray, horizon: int = 100) -> np.ndarray:
    window = np.asarray(init_window, dtype=np.float32).reshape(-1)
    preds = []
    for _ in range(horizon):
        yhat = model.predict(window[None, :, None], verbose=0)[0, 0]
        preds.append(float(yhat))
        window = np.concatenate([window[1:], [yhat]]).astype(np.float32)
    return np.array(preds, dtype=np.float32)


def recursive_rollout_k_step_stride_k(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    if horizon % k != 0:
        raise ValueError("horizon must be divisible by k for stride-k rollout")

    window = np.asarray(init_window, dtype=np.float32).reshape(-1)
    preds = []
    steps = horizon // k
    for _ in range(steps):
        block = model.predict(window[None, :, None], verbose=0)[0]  # (k,)
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if len(block) != k:
            raise ValueError("model output does not match k")
        preds.extend(block.tolist())
        window = np.concatenate([window[k:], block]).astype(np.float32)
    return np.array(preds, dtype=np.float32)


def recursive_rollout_k_step_stride_1(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    window = np.asarray(init_window, dtype=np.float32).reshape(-1)
    preds = []
    for _ in range(horizon):
        block = model.predict(window[None, :, None], verbose=0)[0]  # (k,)
        yhat = float(np.asarray(block, dtype=np.float32).reshape(-1)[0])
        preds.append(yhat)
        window = np.concatenate([window[1:], [yhat]]).astype(np.float32)
    return np.array(preds, dtype=np.float32)


def horizon_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}

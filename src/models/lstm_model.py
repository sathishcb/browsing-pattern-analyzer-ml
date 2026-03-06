# src/models/lstm_model.py
"""
LSTM model for next-category prediction from browsing sequences.
Input: sequence of N browsing categories
Output: predicted next category
"""

import pandas as pd
import numpy as np
import os
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_SEQ = 'category'


def build_sequences(categories: list, seq_len: int):
    X, y = [], []
    for i in range(len(categories) - seq_len):
        X.append(categories[i:i + seq_len])
        y.append(categories[i + seq_len])
    return np.array(X), np.array(y)


def run_lstm(history_path: str, seq_len: int = 5,
             epochs: int = 20, batch_size: int = 32,
             use_gru: bool = False) -> dict:

    print(f"Running {'GRU' if use_gru else 'LSTM'} Next-Category Prediction...")

    df = pd.read_csv(history_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    if 'category' not in df.columns:
        print("  ❌ 'category' column not found.")
        return {}

    # Encode categories
    le = LabelEncoder()
    df['cat_encoded'] = le.fit_transform(df['category'])
    n_classes = len(le.classes_)

    print(f"  Categories ({n_classes}): {list(le.classes_)}")

    sequences = df['cat_encoded'].values
    X, y = build_sequences(sequences, seq_len)

    if len(X) < 50:
        print("  ⚠️  Not enough data for LSTM. Need at least 50 + seq_len records.")
        return {}

    # Train/test split (80/20, no shuffle to preserve time order)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if not HAS_TF:
        print("  ⚠️  TensorFlow not installed. Skipping LSTM training.")
        # Baseline: always predict most common class
        most_common = pd.Series(y_train).mode()[0]
        baseline_acc = (y_test == most_common).mean()
        print(f"  Baseline accuracy (most-common): {baseline_acc:.3f}")
        return {"baseline_accuracy": float(baseline_acc)}

    # ── Build model ────────────────────────────────────────────────────────────
    model = Sequential([
        Embedding(input_dim=n_classes, output_dim=16, input_length=seq_len),
        GRU(64, return_sequences=False) if use_gru else LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=5,
                               restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = (y_pred == y_test).mean()

    # Baseline (most common)
    most_common = pd.Series(y_train).mode()[0]
    baseline_acc = (y_test == most_common).mean()

    print(f"  LSTM Accuracy:  {acc:.4f}")
    print(f"  Baseline Acc:   {baseline_acc:.4f}  (most-common-class)")
    print(f"  Improvement:    {acc - baseline_acc:+.4f}")

    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    model.save("data/lstm_model.keras")
    np.save("data/label_encoder_classes.npy", le.classes_)

    results = {
        "accuracy": float(acc),
        "baseline_accuracy": float(baseline_acc),
        "improvement": float(acc - baseline_acc),
        "n_classes": n_classes,
        "categories": list(le.classes_),
        "report": report
    }
    return results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    dl = cfg['deep_learning']
    run_lstm(
        history_path=cfg['paths']['clean_history'],
        seq_len=dl['sequence_length'],
        epochs=dl['epochs'],
        batch_size=dl['batch_size']
    )

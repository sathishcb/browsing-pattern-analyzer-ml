# src/models/autoencoder.py
"""
Autoencoder for anomaly detection on browsing sessions.
Sessions with high reconstruction error = anomalous behavior.
Anomalies typically = unusual time + high RAM + heavy switching.
"""

import pandas as pd
import numpy as np
import os
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.preprocessing import StandardScaler

FEATURES = [
    'duration_min', 'total_visits', 'unique_domains',
    'social_ratio', 'video_ratio', 'learning_ratio',
    'shopping_ratio', 'switching_rate', 'hour'
]


def build_autoencoder(input_dim: int, encoding_dim: int = 5):
    inp = Input(shape=(input_dim,))
    # Encoder
    x = Dense(16, activation='relu')(inp)
    encoded = Dense(encoding_dim, activation='relu')(x)
    # Decoder
    x = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def run_autoencoder(sessions_path: str, output_path: str,
                    epochs: int = 50, batch_size: int = 16,
                    anomaly_percentile: int = 95) -> pd.DataFrame:

    print("Running Autoencoder Anomaly Detection...")

    df = pd.read_csv(sessions_path)
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if not HAS_TF:
        print("  ⚠️  TensorFlow not installed. Using statistical anomaly detection fallback.")
        # Fallback: Z-score anomaly
        z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / (X_scaled.std(axis=0) + 1e-8))
        errors = z_scores.mean(axis=1)
    else:
        print(f"  Building autoencoder (input_dim={len(available)}, encoding_dim=5)")
        model = build_autoencoder(len(available))

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        reconstructions = model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - reconstructions) ** 2, axis=1)

        final_loss = history.history['val_loss'][-1]
        print(f"  Final validation loss: {final_loss:.4f}")
        model.save("data/autoencoder_model.keras")

    # ── Flag anomalies ────────────────────────────────────────────────────────
    threshold = np.percentile(errors, anomaly_percentile)
    df['reconstruction_error'] = errors
    df['is_anomaly'] = (errors > threshold).astype(int)

    n_anomalies = df['is_anomaly'].sum()
    print(f"  Threshold (p{anomaly_percentile}): {threshold:.4f}")
    print(f"  Anomalous sessions detected: {n_anomalies}")

    # ── Explain top anomalies ─────────────────────────────────────────────────
    top_anomalies = df[df['is_anomaly'] == 1].nlargest(5, 'reconstruction_error')
    print("\n  Top 5 Anomalous Sessions:")
    cols_to_show = ['session_id', 'start_time', 'hour', 'duration_min',
                    'top_category', 'social_ratio', 'switching_rate',
                    'reconstruction_error']
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(top_anomalies[cols_to_show].to_string(index=False))

    # ── Anomaly explanations ──────────────────────────────────────────────────
    explanations = []
    for _, row in top_anomalies.iterrows():
        reasons = []
        if row.get('hour', 12) >= 22 or row.get('hour', 12) <= 4:
            reasons.append("late-night browsing")
        if row.get('switching_rate', 0) > 0.7:
            reasons.append("very high tab switching")
        if row.get('social_ratio', 0) > 0.6:
            reasons.append("heavy social media use")
        if row.get('duration_min', 0) > 90:
            reasons.append("unusually long session")
        reasons_str = ", ".join(reasons) if reasons else "unusual pattern"
        explanations.append(f"Session {row.get('session_id', '?')}: {reasons_str}")

    print("\n  Anomaly Explanations:")
    for e in explanations:
        print(f"    → {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  ✅ Results saved → {output_path}")
    return df, errors, threshold


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    dl = cfg['deep_learning']
    run_autoencoder(
        sessions_path=cfg['paths']['sessions_clustered'],
        output_path=cfg['paths']['sessions_clustered'],
        epochs=dl['epochs'],
        batch_size=dl['batch_size'],
        anomaly_percentile=dl['anomaly_percentile']
    )

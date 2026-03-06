# src/prep/sessionize.py
"""
Builds sessions from browsing events using an inactivity gap threshold.
Each gap > session_gap_minutes = new session.
Computes session-level features for clustering and DL models.
"""

import pandas as pd
import numpy as np
import os
import yaml


def sessionize(input_path: str, output_path: str, gap_minutes: int = 15) -> pd.DataFrame:
    print(f"Sessionizing (gap = {gap_minutes} min)...")

    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Assign session IDs based on time gap
    df['time_diff_min'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    df['new_session'] = (df['time_diff_min'] > gap_minutes).astype(int)
    df['session_id'] = df['new_session'].cumsum()

    print(f"  Total events: {len(df)}")
    print(f"  Total sessions: {df['session_id'].nunique()}")

    # ── Session-level aggregation ──────────────────────────────────────────────
    def mode_val(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else "unknown"

    def switching_rate(x):
        if len(x) <= 1:
            return 0.0
        changes = (x != x.shift()).sum()
        return changes / len(x)

    sessions = df.groupby('session_id').agg(
        start_time        = ('timestamp', 'min'),
        end_time          = ('timestamp', 'max'),
        total_visits      = ('domain', 'count'),
        unique_domains    = ('domain', 'nunique'),
        top_category      = ('category', mode_val),
        top_domain        = ('domain', mode_val),
        social_ratio      = ('category', lambda x: (x == 'social').mean()),
        video_ratio       = ('category', lambda x: (x == 'video').mean()),
        learning_ratio    = ('category', lambda x: (x == 'learning').mean()),
        shopping_ratio    = ('category', lambda x: (x == 'shopping').mean()),
        ai_tools_ratio    = ('category', lambda x: (x == 'ai_tools').mean()),
        other_ratio       = ('category', lambda x: (x == 'other').mean()),
        switching_rate    = ('category', switching_rate),
        is_weekend        = ('is_weekend', 'first'),
        time_block        = ('time_block', mode_val),
        categories_list   = ('category', lambda x: '|'.join(x.tolist()))
    ).reset_index()

    sessions['duration_min'] = (
        sessions['end_time'] - sessions['start_time']
    ).dt.total_seconds() / 60
    sessions['hour'] = pd.to_datetime(sessions['start_time']).dt.hour
    sessions['date'] = pd.to_datetime(sessions['start_time']).dt.date
    sessions['day_name'] = pd.to_datetime(sessions['start_time']).dt.day_name()

    # Clamp duration (sessions under 1 min → 1 min)
    sessions['duration_min'] = sessions['duration_min'].clip(lower=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sessions.to_csv(output_path, index=False)

    # Summary stats
    print(f"  ✅ Saved {len(sessions)} sessions → {output_path}")
    print(f"  Avg session duration: {sessions['duration_min'].mean():.1f} min")
    print(f"  Top category overall: {sessions['top_category'].mode()[0]}")
    return sessions


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    sessionize(
        input_path=cfg['paths']['clean_history'],
        output_path=cfg['paths']['sessions'],
        gap_minutes=cfg['data']['session_gap_minutes']
    )

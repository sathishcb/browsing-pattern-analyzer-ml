# src/analytics/ram_correlation.py
"""
Merges browsing events with RAM logs using nearest-timestamp join.
Produces RAM stats per session and per category.
"""

import pandas as pd
import numpy as np
import os
import yaml


def merge_ram(browsing_path: str, ram_path: str, output_path: str) -> tuple:
    print("Merging browsing data with RAM logs...")

    browsing = pd.read_csv(browsing_path)
    browsing['timestamp'] = pd.to_datetime(browsing['timestamp'])

    ram = pd.read_csv(ram_path)
    ram['timestamp'] = pd.to_datetime(ram['timestamp'])

    # Nearest-timestamp merge (merge_asof requires sorted data)
    browsing_sorted = browsing.sort_values('timestamp')
    ram_sorted = ram.sort_values('timestamp')

    merged = pd.merge_asof(
        browsing_sorted,
        ram_sorted[['timestamp', 'ram_used_mb', 'ram_available_mb',
                    'browser_ram_mb', 'cpu_percent']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('5min')  # Only match within 5 min
    )

    print(f"  Merged {len(merged)} records")

    # ── RAM stats by category ──────────────────────────────────────────────────
    ram_by_category = merged.groupby('category').agg(
        mean_browser_ram_mb = ('browser_ram_mb', 'mean'),
        peak_browser_ram_mb = ('browser_ram_mb', 'max'),
        mean_system_ram_mb  = ('ram_used_mb', 'mean'),
        mean_cpu_percent    = ('cpu_percent', 'mean'),
        visit_count         = ('domain', 'count')
    ).round(1).sort_values('peak_browser_ram_mb', ascending=False)

    print("\n  📊 RAM by Category (Top 5):")
    print(ram_by_category.head())

    # ── RAM stats by hour ──────────────────────────────────────────────────────
    ram_by_hour = merged.groupby('hour').agg(
        mean_browser_ram_mb = ('browser_ram_mb', 'mean'),
        peak_browser_ram_mb = ('browser_ram_mb', 'max'),
    ).round(1)

    # Top 3 memory-heavy categories
    top3 = ram_by_category.head(3).index.tolist()
    print(f"\n  🔴 Top 3 memory-heavy categories: {top3}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    ram_by_category.to_csv("data/ram_by_category.csv")
    ram_by_hour.to_csv("data/ram_by_hour.csv")

    print(f"  ✅ Saved merged data → {output_path}")
    return merged, ram_by_category, ram_by_hour


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    merge_ram(
        browsing_path=cfg['paths']['clean_history'],
        ram_path=cfg['paths']['ram_log'],
        output_path=cfg['paths']['merged']
    )

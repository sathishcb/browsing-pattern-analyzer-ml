# src/analytics/recommendations.py
"""
Generates actionable recommendations based on:
- Session cluster patterns
- RAM correlation findings
- Time-based usage patterns
- Deep learning signals (anomalies / predictions)
"""

import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime


def generate_recommendations(sessions_path: str,
                              ram_by_cat_path: str = None) -> list:
    print("Generating recommendations...")

    df = pd.read_csv(sessions_path)
    recs = []

    # ── Late-night social media ────────────────────────────────────────────────
    late_mask = df['hour'] >= 22
    if late_mask.sum() > 0:
        late_social = df[late_mask]['social_ratio'].mean()
        if late_social > 0.35:
            recs.append({
                "type": "Digital Wellbeing",
                "icon": "🌙",
                "title": "Late-Night Social Media Loop Detected",
                "evidence": f"{late_social*100:.0f}% of late-night sessions are social media",
                "recommendation": "Set a screen-off rule after 10 PM. "
                                   "Use 'Do Not Disturb' or app timers."
            })

    # ── High overall social ratio ──────────────────────────────────────────────
    avg_social = df['social_ratio'].mean()
    if avg_social > 0.40:
        recs.append({
            "type": "Productivity",
            "icon": "📱",
            "title": "Social Media Dominates Your Browsing",
            "evidence": f"Social media accounts for {avg_social*100:.0f}% of all sessions",
            "recommendation": "Schedule two focused 'no-social' blocks per day (9-12 AM and 2-5 PM). "
                               "Use site blockers like Cold Turkey or Freedom."
        })

    # ── Peak browsing hour – recommend focus block ────────────────────────────
    hourly = df.groupby('hour')['total_visits'].sum()
    peak_hour = hourly.idxmax()
    recs.append({
        "type": "Productivity",
        "icon": "⏰",
        "title": f"Peak Distraction Hour: {peak_hour}:00",
        "evidence": f"Highest browsing volume at {peak_hour}:00 with {int(hourly[peak_hour])} visits",
        "recommendation": f"Protect {peak_hour}:00–{peak_hour+1}:00 for deep work. "
                            "Turn off notifications, close non-essential tabs."
    })

    # ── Good learning ratio ───────────────────────────────────────────────────
    avg_learn = df['learning_ratio'].mean()
    if avg_learn > 0.25:
        recs.append({
            "type": "Positive Habit",
            "icon": "📚",
            "title": "Strong Learning Behavior Detected",
            "evidence": f"Learning sites account for {avg_learn*100:.0f}% of browsing",
            "recommendation": "Great job! Consider scheduling your learning sessions in the morning "
                               "when focus is highest, and tracking progress with Notion or Obsidian."
        })

    # ── High switching rate (attention fragmentation) ─────────────────────────
    avg_switch = df['switching_rate'].mean()
    if avg_switch > 0.55:
        recs.append({
            "type": "Focus",
            "icon": "🔄",
            "title": "High Tab-Switching Detected",
            "evidence": f"Average category switching rate: {avg_switch:.2f} (>0.55 = fragmented)",
            "recommendation": "Try single-tab browsing for focused tasks. "
                               "OneTab or Tab Suspender extensions can help reduce tab overload."
        })

    # ── Session duration outliers ─────────────────────────────────────────────
    long_sessions = df[df['duration_min'] > 90]
    if len(long_sessions) > 0:
        top_cat = long_sessions['top_category'].mode()[0]
        recs.append({
            "type": "Digital Wellbeing",
            "icon": "⏳",
            "title": "Unusually Long Sessions Detected",
            "evidence": f"{len(long_sessions)} sessions > 90 min, mostly on '{top_cat}'",
            "recommendation": "Take a 5-10 min break every 45 minutes. "
                               "Try the Pomodoro technique: 25 min work, 5 min break."
        })

    # ── RAM correlation recommendations ──────────────────────────────────────
    if ram_by_cat_path and os.path.exists(ram_by_cat_path):
        ram_df = pd.read_csv(ram_by_cat_path, index_col=0)
        top3_ram = ram_df.head(3).index.tolist()
        peak_vals = ram_df.head(3)['peak_browser_ram_mb'].tolist()
        recs.append({
            "type": "IT Performance",
            "icon": "💾",
            "title": "Memory-Heavy Browsing Categories",
            "evidence": f"Top 3 RAM-heavy: {', '.join(top3_ram)} "
                        f"(peaks: {', '.join(f'{v:.0f} MB' for v in peak_vals)})",
            "recommendation": "Close tabs from these categories when doing RAM-intensive work. "
                               "Use lightweight browsers or enable hardware acceleration."
        })

    # ── Weekend vs weekday pattern ────────────────────────────────────────────
    if 'is_weekend' in df.columns:
        wknd = df[df['is_weekend'] == 1]
        wkday = df[df['is_weekend'] == 0]
        if len(wknd) > 0 and len(wkday) > 0:
            wknd_social = wknd['social_ratio'].mean()
            wkday_social = wkday['social_ratio'].mean()
            if wknd_social > wkday_social + 0.15:
                recs.append({
                    "type": "Digital Wellbeing",
                    "icon": "📅",
                    "title": "Weekend Social Media Spike",
                    "evidence": f"Social ratio: Weekends {wknd_social:.0%} vs Weekdays {wkday_social:.0%}",
                    "recommendation": "Use weekends for offline activities. "
                                       "Set a stricter screen-time limit on Saturdays and Sundays."
                })

    # ── Anomaly recommendation ────────────────────────────────────────────────
    if 'is_anomaly' in df.columns:
        n_anomalies = df['is_anomaly'].sum()
        if n_anomalies > 0:
            recs.append({
                "type": "Behavior Insight",
                "icon": "🚨",
                "title": f"{n_anomalies} Anomalous Sessions Detected",
                "evidence": "Sessions with unusual time, switching, or duration patterns",
                "recommendation": "Review these sessions — they often represent stress-browsing "
                                   "or unplanned rabbit holes. Set alarms as session reminders."
            })

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  📋 {len(recs)} Recommendations Generated:\n")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['icon']} [{r['type']}] {r['title']}")
        print(f"     Evidence: {r['evidence']}")
        print(f"     Action:   {r['recommendation']}\n")

    # Save as CSV
    rec_df = pd.DataFrame(recs)
    rec_df.to_csv("data/recommendations.csv", index=False)
    print(f"  ✅ Recommendations saved → data/recommendations.csv")
    return recs


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    ram_cat_path = "data/ram_by_category.csv"
    generate_recommendations(
        sessions_path=cfg['paths']['sessions_clustered'],
        ram_by_cat_path=ram_cat_path
    )

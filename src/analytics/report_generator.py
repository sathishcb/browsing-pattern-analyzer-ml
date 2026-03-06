# src/analytics/report_generator.py
"""
Auto-generates a Markdown final report from all outputs.
"""

import pandas as pd
import os
import yaml
from datetime import datetime


def generate_report(output_path: str = "reports/final_report.md"):
    print("Generating final report...")
    os.makedirs("reports", exist_ok=True)

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    lines = []

    # ── Header ─────────────────────────────────────────────────────────────────
    lines += [
        "# DS105 Final Project Report",
        "## Time-Based Browsing Pattern Analyzer with RAM Correlation",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Analysis Window:** Last {cfg['data']['days_window']} days",
        "",
        "---",
        ""
    ]

    # ── Section 1: Overview ────────────────────────────────────────────────────
    lines += ["## 1. Overview", ""]
    if os.path.exists("data/browsing_history.csv"):
        df = pd.read_csv("data/browsing_history.csv")
        lines += [
            f"- **Total page visits analyzed:** {len(df):,}",
            f"- **Unique domains:** {df['domain'].nunique()}",
            f"- **Categories found:** {', '.join(df['category'].unique())}",
            ""
        ]

    # ── Section 2: Top Domains ────────────────────────────────────────────────
    lines += ["## 2. Top Websites / Domains", ""]
    if os.path.exists("data/browsing_history.csv"):
        df = pd.read_csv("data/browsing_history.csv")
        top10 = df['domain'].value_counts().head(10)
        lines += ["| Rank | Domain | Visits |", "|------|--------|--------|"]
        for i, (domain, count) in enumerate(top10.items(), 1):
            lines.append(f"| {i} | {domain} | {count} |")
        lines.append("")

    # ── Section 3: Hourly Patterns ────────────────────────────────────────────
    lines += ["## 3. Hourly & Day-wise Usage Patterns", ""]
    if os.path.exists("data/browsing_history.csv"):
        df = pd.read_csv("data/browsing_history.csv")
        hourly = df.groupby('hour')['domain'].count()
        peak = hourly.idxmax()
        lines += [
            f"- **Peak browsing hour:** {peak}:00 ({int(hourly[peak])} visits)",
            f"- **Quietest hour:** {hourly.idxmin()}:00",
            "",
            "![Hourly Activity](figures/01_hourly_activity.png)",
            "![Heatmap](figures/05_day_hour_heatmap.png)",
            ""
        ]

    # ── Section 4: Category Analysis ─────────────────────────────────────────
    lines += ["## 4. Category Analysis", ""]
    if os.path.exists("data/browsing_history.csv"):
        df = pd.read_csv("data/browsing_history.csv")
        cat = df['category'].value_counts(normalize=True) * 100
        lines += ["| Category | Share (%) |", "|----------|-----------|"]
        for cat_name, pct in cat.items():
            lines.append(f"| {cat_name} | {pct:.1f}% |")
        lines.append("")
        lines.append("![Category Distribution](figures/02_category_distribution.png)")
        lines.append("")

    # ── Section 5: Session Clusters ───────────────────────────────────────────
    lines += ["## 5. Session Cluster Summary", ""]
    for path in ["data/sessions_clustered.csv", "data/sessions.csv"]:
        if os.path.exists(path):
            sessions = pd.read_csv(path)
            lines += [
                f"- **Total sessions:** {len(sessions)}",
                f"- **Average session duration:** {sessions['duration_min'].mean():.1f} minutes",
            ]
            if 'cluster_label' in sessions.columns:
                lines += ["", "| Cluster | Sessions | Avg Duration (min) | Avg Social Ratio |",
                          "|---------|----------|--------------------|------------------|"]
                grp = sessions.groupby('cluster_label').agg(
                    count=('session_id', 'count'),
                    avg_dur=('duration_min', 'mean'),
                    avg_social=('social_ratio', 'mean')
                )
                for label, row in grp.iterrows():
                    lines.append(f"| {label} | {int(row['count'])} | "
                                 f"{row['avg_dur']:.1f} | {row['avg_social']:.2f} |")
            lines.append("")
            lines.append("![Clusters](figures/06_cluster_analysis.png)")
            lines.append("")
            break

    # ── Section 6: RAM Correlation ────────────────────────────────────────────
    lines += ["## 6. RAM Correlation Findings", ""]
    if os.path.exists("data/ram_by_category.csv"):
        ram = pd.read_csv("data/ram_by_category.csv", index_col=0)
        lines += ["| Category | Mean Browser RAM (MB) | Peak Browser RAM (MB) |",
                  "|----------|-----------------------|-----------------------|"]
        for cat, row in ram.head(5).iterrows():
            lines.append(f"| {cat} | {row['mean_browser_ram_mb']:.0f} | "
                         f"{row['peak_browser_ram_mb']:.0f} |")
        top3 = ram.head(3).index.tolist()
        lines += ["", f"**Top 3 memory-heavy categories:** {', '.join(top3)}", "",
                  "![RAM by Category](figures/07_ram_by_category.png)", ""]
    else:
        lines += ["*RAM data not available.*", ""]

    # ── Section 7: Deep Learning ──────────────────────────────────────────────
    lines += ["## 7. Deep Learning Results", ""]
    if os.path.exists("data/sessions_clustered.csv"):
        s = pd.read_csv("data/sessions_clustered.csv")
        if 'is_anomaly' in s.columns:
            n = s['is_anomaly'].sum()
            thresh = s[s['is_anomaly'] == 1]['reconstruction_error'].min() if n > 0 else "N/A"
            lines += [
                "**Model Used:** Autoencoder (anomaly detection)",
                f"- Anomalous sessions detected: **{n}**",
                f"- Reconstruction error threshold: **{thresh:.4f}** (95th percentile)" if isinstance(thresh, float) else f"- Threshold: {thresh}",
                "",
                "![Anomaly Detection](figures/08_anomaly_detection.png)",
                ""
            ]
        else:
            lines += ["*Deep learning results not yet generated.*", ""]
    else:
        lines += ["*Sessions not yet available.*", ""]

    # ── Section 8: Recommendations ────────────────────────────────────────────
    lines += ["## 8. Recommendations", ""]
    if os.path.exists("data/recommendations.csv"):
        rec_df = pd.read_csv("data/recommendations.csv")
        for i, row in rec_df.iterrows():
            icon = row.get('icon', '•')
            title = row.get('title', '')
            rec = row.get('recommendation', '')
            evidence = row.get('evidence', '')
            lines += [
                f"### {icon} {title}",
                f"**Evidence:** {evidence}",
                f"**Action:** {rec}",
                ""
            ]
    else:
        lines += ["*Recommendations not yet generated.*", ""]

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Tech Stack",
        "Python · Pandas · NumPy · scikit-learn · TensorFlow/Keras · psutil · Streamlit · Matplotlib · Seaborn",
        "",
        "*Generated automatically by DS105 Final Project pipeline.*"
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  ✅ Report saved → {output_path}")


if __name__ == "__main__":
    generate_report()

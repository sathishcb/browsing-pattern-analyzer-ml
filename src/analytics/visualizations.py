# src/analytics/visualizations.py
"""
Generates all project visualizations and saves them to reports/figures/.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(name):
    path = f"{FIGURES_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_hourly_activity(df):
    hourly = df.groupby('hour')['domain'].count().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(hourly.index, hourly.values,
                  color=cm.RdYlGn_r(hourly.values / hourly.values.max()))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Visits")
    ax.set_title("Browsing Activity by Hour of Day")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, ha='right')
    plt.tight_layout()
    save("01_hourly_activity")


def plot_category_distribution(df):
    cat_counts = df['category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(cat_counts))
    bars = ax.barh(cat_counts.index, cat_counts.values, color=colors)
    ax.set_xlabel("Number of Visits")
    ax.set_title("Website Category Distribution")
    for bar, val in zip(bars, cat_counts.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=9)
    plt.tight_layout()
    save("02_category_distribution")


def plot_top_domains(df, top_n=10):
    top = df['domain'].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top.values, y=top.index, ax=ax, palette="Blues_r")
    ax.set_xlabel("Visits")
    ax.set_title(f"Top {top_n} Domains")
    plt.tight_layout()
    save("03_top_domains")


def plot_daily_pattern(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily = df.groupby(['date', 'category'])['domain'].count().unstack(fill_value=0)
    ax = daily.plot(kind='bar', stacked=True, figsize=(12, 5), colormap='tab10')
    ax.set_xlabel("Date")
    ax.set_ylabel("Visits")
    ax.set_title("Daily Browsing by Category")
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save("04_daily_category_stack")


def plot_heatmap(df):
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_name'] = pd.to_datetime(df['timestamp']).dt.day_name()
    ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = df.groupby(['day_name', 'hour'])['domain'].count().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in ORDER if d in pivot.index])

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3, linecolor='white')
    ax.set_title("Browsing Heatmap (Day × Hour)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day")
    plt.tight_layout()
    save("05_day_hour_heatmap")


def plot_clusters(sessions_df):
    if 'cluster_label' not in sessions_df.columns:
        return
    label_counts = sessions_df['cluster_label'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    axes[0].pie(label_counts.values, labels=label_counts.index,
                autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("Set3", len(label_counts)))
    axes[0].set_title("Session Cluster Distribution")

    # Scatter (duration vs social ratio)
    if 'duration_min' in sessions_df.columns and 'social_ratio' in sessions_df.columns:
        unique_labels = sessions_df['cluster_label'].unique()
        palette = sns.color_palette("tab10", len(unique_labels))
        for i, label in enumerate(unique_labels):
            subset = sessions_df[sessions_df['cluster_label'] == label]
            axes[1].scatter(subset['duration_min'], subset['social_ratio'],
                            label=label, color=palette[i], alpha=0.6, s=40)
        axes[1].set_xlabel("Session Duration (min)")
        axes[1].set_ylabel("Social Media Ratio")
        axes[1].set_title("Clusters: Duration vs Social Ratio")
        axes[1].legend(fontsize=7)

    plt.tight_layout()
    save("06_cluster_analysis")


def plot_ram_by_category(ram_df):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ram_df))
    ax.bar(x, ram_df['mean_browser_ram_mb'], label='Mean RAM', color='steelblue', alpha=0.8)
    ax.bar(x, ram_df['peak_browser_ram_mb'], label='Peak RAM', color='tomato', alpha=0.5, width=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ram_df.index, rotation=30, ha='right')
    ax.set_ylabel("Browser RAM (MB)")
    ax.set_title("Browser RAM Usage by Category")
    ax.legend()
    plt.tight_layout()
    save("07_ram_by_category")


def plot_anomalies(sessions_df):
    if 'reconstruction_error' not in sessions_df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Distribution of reconstruction errors
    axes[0].hist(sessions_df['reconstruction_error'], bins=30,
                 color='steelblue', edgecolor='white')
    if 'is_anomaly' in sessions_df.columns:
        thresh = sessions_df[sessions_df['is_anomaly'] == 1]['reconstruction_error'].min()
        axes[0].axvline(thresh, color='red', linestyle='--', label=f'Threshold: {thresh:.3f}')
        axes[0].legend()
    axes[0].set_xlabel("Reconstruction Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Autoencoder Reconstruction Error Distribution")

    # Anomaly sessions scatter
    if 'is_anomaly' in sessions_df.columns and 'hour' in sessions_df.columns:
        normal = sessions_df[sessions_df['is_anomaly'] == 0]
        anomal = sessions_df[sessions_df['is_anomaly'] == 1]
        axes[1].scatter(normal['hour'], normal['reconstruction_error'],
                        alpha=0.4, color='steelblue', label='Normal', s=30)
        axes[1].scatter(anomal['hour'], anomal['reconstruction_error'],
                        alpha=0.8, color='red', label='Anomaly', s=50, marker='X')
        axes[1].set_xlabel("Session Start Hour")
        axes[1].set_ylabel("Reconstruction Error")
        axes[1].set_title("Anomalous Sessions by Hour")
        axes[1].legend()

    plt.tight_layout()
    save("08_anomaly_detection")


def run_all_visualizations():
    print("Generating all visualizations...")

    # Browsing history plots
    if os.path.exists("data/browsing_history.csv"):
        df = pd.read_csv("data/browsing_history.csv")
        plot_hourly_activity(df)
        plot_category_distribution(df)
        plot_top_domains(df)
        plot_daily_pattern(df)
        plot_heatmap(df)

    # Session cluster plots
    for path in ["data/sessions_clustered.csv", "data/sessions.csv"]:
        if os.path.exists(path):
            sessions = pd.read_csv(path)
            plot_clusters(sessions)
            plot_anomalies(sessions)
            break

    # RAM plots
    if os.path.exists("data/ram_by_category.csv"):
        ram_df = pd.read_csv("data/ram_by_category.csv", index_col=0)
        plot_ram_by_category(ram_df)

    print(f"\n  ✅ All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    run_all_visualizations()

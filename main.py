# main.py
"""
DS105 Final Project - Full Pipeline Runner
Runs all 7 modules in sequence.

Usage:
    python main.py                    # Full run with sample data
    python main.py --real             # Use real Chrome browser data
    python main.py --model lstm       # Use LSTM instead of Autoencoder
    python main.py --days 3           # Analyze last 3 days
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# ── Add src to path ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_banner():
    print("=" * 60)
    print("  DS105 Final Project")
    print("  Time-Based Browsing Pattern Analyzer")
    print("  with Deep Learning & RAM Correlation")
    print("=" * 60)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()


def run_pipeline(use_real_data=False, model_type=None, days=None):
    print_banner()

    # Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    if days:
        cfg['data']['days_window'] = days
    if model_type:
        cfg['deep_learning']['model_type'] = model_type

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 1: DATA COLLECTION
    # ─────────────────────────────────────────────────────────────────────────
    print("━" * 60)
    print("  MODULE 1: Data Collection")
    print("━" * 60)

    if use_real_data:
        from collect.extract_history import extract_history
        result = extract_history(days=cfg['data']['days_window'],
                                  browser=cfg['browser']['type'])
        if result is None:
            print("  → Falling back to sample data.")
            use_real_data = False

    if not use_real_data:
        from collect.generate_sample import (
            generate_browsing_history,
            generate_ram_log,
            generate_domain_category_map
        )
        generate_browsing_history(days=cfg['data']['days_window'])
        generate_ram_log(days=cfg['data']['days_window'])
        generate_domain_category_map()

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 2: PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  MODULE 2: Preprocessing")
    print("━" * 60)

    from prep.preprocess import preprocess
    raw = cfg['paths']['raw_history']
    src = raw if os.path.exists(raw) else cfg['paths']['clean_history']
    df = preprocess(
        input_path=src,
        output_path=cfg['paths']['clean_history'],
        map_path=cfg['paths']['domain_map']
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 3: SESSIONIZATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  MODULE 3: Sessionization")
    print("━" * 60)

    from prep.sessionize import sessionize
    sessions = sessionize(
        input_path=cfg['paths']['clean_history'],
        output_path=cfg['paths']['sessions'],
        gap_minutes=cfg['data']['session_gap_minutes']
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 4: RAM CORRELATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  MODULE 4: RAM Correlation")
    print("━" * 60)

    from analytics.ram_correlation import merge_ram
    merged, ram_by_cat, ram_by_hour = merge_ram(
        browsing_path=cfg['paths']['clean_history'],
        ram_path=cfg['paths']['ram_log'],
        output_path=cfg['paths']['merged']
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 5: CLUSTERING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  MODULE 5: Clustering")
    print("━" * 60)

    from models.clustering import run_clustering
    sessions_clust, sil_score, profiles = run_clustering(
        sessions_path=cfg['paths']['sessions'],
        output_path=cfg['paths']['sessions_clustered'],
        n_clusters=cfg['clustering']['n_clusters'],
        algorithm=cfg['clustering']['algorithm'],
        random_seed=cfg['clustering']['random_seed']
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 6: DEEP LEARNING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  MODULE 6: Deep Learning")
    print("━" * 60)

    dl_type = cfg['deep_learning']['model_type']
    dl = cfg['deep_learning']

    if dl_type == "lstm":
        from models.lstm_model import run_lstm
        results = run_lstm(
            history_path=cfg['paths']['clean_history'],
            seq_len=dl['sequence_length'],
            epochs=dl['epochs'],
            batch_size=dl['batch_size']
        )
    else:
        from models.autoencoder import run_autoencoder
        sessions_final, errors, threshold = run_autoencoder(
            sessions_path=cfg['paths']['sessions_clustered'],
            output_path=cfg['paths']['sessions_clustered'],
            epochs=dl['epochs'],
            batch_size=dl['batch_size'],
            anomaly_percentile=dl['anomaly_percentile']
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 7: RECOMMENDATIONS + REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  MODULE 7: Recommendations & Report")
    print("━" * 60)

    from analytics.recommendations import generate_recommendations
    recs = generate_recommendations(
        sessions_path=cfg['paths']['sessions_clustered'],
        ram_by_cat_path="data/ram_by_category.csv"
    )

    from analytics.visualizations import run_all_visualizations
    run_all_visualizations()

    from analytics.report_generator import generate_report
    generate_report(output_path=cfg['paths']['report'])

    # ─────────────────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  📊 Report:      {cfg['paths']['report']}")
    print(f"  📈 Figures:     reports/figures/")
    print(f"  💾 Data:        data/")
    print(f"  🌐 Dashboard:   streamlit run dashboard.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DS105 Browsing Analyzer Pipeline")
    parser.add_argument("--real", action="store_true",
                        help="Use real Chrome browser history instead of sample data")
    parser.add_argument("--model", choices=["lstm", "autoencoder"], default=None,
                        help="Deep learning model to use")
    parser.add_argument("--days", type=int, choices=[3, 4, 5], default=None,
                        help="Number of days to analyze")
    args = parser.parse_args()

    run_pipeline(
        use_real_data=args.real,
        model_type=args.model,
        days=args.days
    )

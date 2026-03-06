# src/collect/ram_logger.py
"""
Logs RAM and CPU usage periodically while you browse.
Run this in a separate terminal window BEFORE you start browsing.

Usage:
    python src/collect/ram_logger.py
"""

import psutil
import pandas as pd
import time
import os
import yaml
from datetime import datetime


def log_ram(duration_minutes=60, interval_seconds=10, output_path="data/ram_log.csv"):
    """
    Logs system RAM, browser RAM, and CPU usage every `interval_seconds`.
    Runs for `duration_minutes` total.
    """
    os.makedirs("data", exist_ok=True)
    records = []
    end_time = time.time() + duration_minutes * 60
    total_logs = int((duration_minutes * 60) / interval_seconds)
    count = 0

    print(f"📊 RAM Logger started. Will run for {duration_minutes} minutes.")
    print(f"   Logging every {interval_seconds} seconds (~{total_logs} records)")
    print(f"   Output: {output_path}")
    print("   Press Ctrl+C to stop early.\n")

    try:
        while time.time() < end_time:
            mem = psutil.virtual_memory()
            browser_ram = 0.0
            browser_processes = []

            for proc in psutil.process_iter(['name', 'memory_info', 'pid']):
                try:
                    name = proc.info['name']
                    if name in ['chrome.exe', 'msedge.exe', 'Google Chrome', 'Microsoft Edge']:
                        ram_mb = proc.info['memory_info'].rss / (1024 ** 2)
                        browser_ram += ram_mb
                        browser_processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            record = {
                "timestamp": datetime.now().isoformat(),
                "ram_used_mb": round(mem.used / (1024 ** 2), 1),
                "ram_available_mb": round(mem.available / (1024 ** 2), 1),
                "ram_total_mb": round(mem.total / (1024 ** 2), 1),
                "ram_percent": mem.percent,
                "browser_ram_mb": round(browser_ram, 1),
                "browser_tab_count": len(browser_processes),
                "cpu_percent": psutil.cpu_percent(interval=1)
            }
            records.append(record)
            count += 1

            elapsed = count * interval_seconds
            print(f"  [{count}/{total_logs}] RAM: {record['ram_percent']}% | "
                  f"Browser: {record['browser_ram_mb']:.0f} MB | "
                  f"CPU: {record['cpu_percent']}%", end='\r')

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Stopped early at {count} records.")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\n✅ RAM log saved: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    log_ram(
        duration_minutes=cfg['ram_logger']['duration_minutes'],
        interval_seconds=cfg['ram_logger']['interval_seconds'],
        output_path=cfg['paths']['ram_log']
    )

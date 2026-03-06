# src/collect/extract_history.py
"""
Extracts real browsing history from Chrome or Edge SQLite database.
Works on Windows. Adjust path for Mac/Linux if needed.
"""

import sqlite3
import shutil
import pandas as pd
import os
import platform
import yaml
from datetime import datetime, timedelta

def get_browser_path(browser="chrome"):
    system = platform.system()
    if system == "Windows":
        if browser == "chrome":
            return os.path.expanduser(
                r"~\AppData\Local\Google\Chrome\User Data\Default\History"
            )
        elif browser == "edge":
            return os.path.expanduser(
                r"~\AppData\Local\Microsoft\Edge\User Data\Default\History"
            )
    elif system == "Darwin":  # Mac
        if browser == "chrome":
            return os.path.expanduser(
                "~/Library/Application Support/Google/Chrome/Default/History"
            )
    elif system == "Linux":
        if browser == "chrome":
            return os.path.expanduser("~/.config/google-chrome/Default/History")
    raise ValueError(f"Unsupported OS/browser combo: {system}/{browser}")


def extract_history(days=5, browser="chrome"):
    """
    Extract browsing history from Chrome/Edge SQLite DB.
    Must close or copy the file because Chrome locks it while running.
    """
    print(f"Extracting {browser} history for last {days} days...")
    
    try:
        history_path = get_browser_path(browser)
        if not os.path.exists(history_path):
            print(f"  ⚠️  Browser history file not found at: {history_path}")
            print("  → Run generate_sample.py to use synthetic data instead.")
            return None

        # Must copy because the browser locks the file
        temp_path = "data/temp_history.db"
        os.makedirs("data", exist_ok=True)
        shutil.copy2(history_path, temp_path)

        conn = sqlite3.connect(temp_path)

        # Chrome/Edge store time as microseconds since Jan 1, 1601
        query = """
            SELECT
                url,
                title,
                datetime(last_visit_time/1000000 - 11644473600, 'unixepoch', 'localtime') AS timestamp
            FROM urls
            ORDER BY last_visit_time DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        os.remove(temp_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff].copy()
        df = df.dropna(subset=['url', 'timestamp'])

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/browsing_history_raw.csv", index=False)
        print(f"  ✅ Extracted {len(df)} records → data/browsing_history_raw.csv")
        return df

    except PermissionError:
        print("  ❌ Permission denied. Close Chrome/Edge first, then run again.")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print("  → Run generate_sample.py to use synthetic data instead.")
        return None


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    extract_history(
        days=cfg['data']['days_window'],
        browser=cfg['browser']['type']
    )

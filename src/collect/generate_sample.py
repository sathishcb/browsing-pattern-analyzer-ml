# src/collect/generate_sample.py
"""
Generates realistic synthetic browsing history and RAM log data for testing.
Run this if you don't want to extract real browser data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

DOMAINS = {
    "youtube.com": "video",
    "netflix.com": "video",
    "instagram.com": "social",
    "twitter.com": "social",
    "facebook.com": "social",
    "reddit.com": "social",
    "github.com": "learning",
    "stackoverflow.com": "learning",
    "coursera.org": "learning",
    "kaggle.com": "learning",
    "medium.com": "learning",
    "amazon.com": "shopping",
    "flipkart.com": "shopping",
    "gmail.com": "email",
    "outlook.com": "email",
    "google.com": "search",
    "bing.com": "search",
    "linkedin.com": "professional",
    "chatgpt.com": "ai_tools",
    "claude.ai": "ai_tools",
    "news.google.com": "news",
    "bbc.com": "news",
    "ndtv.com": "news",
    "whatsapp.com": "messaging",
    "discord.com": "messaging",
}

DOMAIN_LIST = list(DOMAINS.keys())
CATEGORY_LIST = list(DOMAINS.values())

# Hour weights: more browsing in morning, lunch, evening
HOUR_WEIGHTS = [
    0.2, 0.1, 0.05, 0.05, 0.05, 0.1,   # 0-5 AM
    0.3, 0.8, 1.2, 1.5, 1.8, 1.5,       # 6-11 AM
    1.8, 1.2, 1.0, 1.2, 1.5, 2.0,       # 12-5 PM
    2.5, 2.8, 2.5, 1.8, 1.2, 0.8        # 6-11 PM
]

def generate_browsing_history(days=5, records_per_day=120):
    print(f"Generating {days} days of browsing history...")
    rows = []
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for day_offset in range(days):
        date = base - timedelta(days=days - day_offset)
        n = records_per_day + random.randint(-20, 20)

        # Pick hours based on weights
        hours = random.choices(range(24), weights=HOUR_WEIGHTS, k=n)
        hours.sort()

        for hour in hours:
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = date + timedelta(hours=hour, minutes=minute, seconds=second)

            # Bias categories by hour
            if 22 <= hour or hour <= 2:
                domain = random.choices(DOMAIN_LIST, weights=[
                    3, 2, 5, 4, 4, 5, 1, 1, 1, 1, 1,
                    2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 4, 3
                ], k=1)[0]
            elif 9 <= hour <= 17:
                domain = random.choices(DOMAIN_LIST, weights=[
                    2, 1, 2, 2, 2, 2, 5, 5, 4, 4, 4,
                    2, 2, 3, 3, 4, 2, 4, 4, 4, 2, 1, 1, 2, 2
                ], k=1)[0]
            else:
                domain = random.choice(DOMAIN_LIST)

            rows.append({
                "timestamp": ts.isoformat(),
                "domain": domain,
                "category": DOMAINS[domain],
                "title": f"Page on {domain}",
                "hour": hour,
                "date": date.date(),
                "day_name": date.strftime("%A")
            })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/browsing_history.csv", index=False)
    print(f"  ✅ Saved {len(df)} records to data/browsing_history.csv")
    return df


def generate_ram_log(days=5, interval_sec=10):
    print("Generating RAM log...")
    rows = []
    base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = base - timedelta(days=days)
    current = start
    end = base

    base_system_ram = 6000  # MB
    base_browser_ram = 800  # MB

    while current <= end:
        hour = current.hour
        # Simulate RAM spikes during peak browsing hours
        if 18 <= hour <= 23:
            spike = random.uniform(1.0, 1.6)
        elif 9 <= hour <= 17:
            spike = random.uniform(1.0, 1.4)
        else:
            spike = random.uniform(0.7, 1.1)

        system_ram = base_system_ram * spike + random.gauss(0, 100)
        browser_ram = base_browser_ram * spike + random.gauss(0, 80)
        browser_ram = max(200, browser_ram)
        system_ram = max(3000, min(system_ram, 15000))

        rows.append({
            "timestamp": current.isoformat(),
            "ram_used_mb": round(system_ram, 1),
            "ram_available_mb": round(16384 - system_ram, 1),
            "browser_ram_mb": round(browser_ram, 1),
            "cpu_percent": round(random.uniform(5, 80) * spike, 1)
        })
        current += timedelta(seconds=interval_sec)

    df = pd.DataFrame(rows)
    df.to_csv("data/ram_log.csv", index=False)
    print(f"  ✅ Saved {len(df)} RAM records to data/ram_log.csv")
    return df


def generate_domain_category_map():
    print("Generating domain-category map...")
    rows = [{"domain": d, "category": c} for d, c in DOMAINS.items()]
    df = pd.DataFrame(rows)
    df.to_csv("data/domain_category_map.csv", index=False)
    print(f"  ✅ Saved {len(df)} entries to data/domain_category_map.csv")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("  Generating Sample Dataset")
    print("=" * 50)
    generate_browsing_history(days=5)
    generate_ram_log(days=5)
    generate_domain_category_map()
    print("\n🎉 All sample data generated successfully!")

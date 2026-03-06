# src/prep/preprocess.py
"""
Cleans browsing history:
- Extracts domain from URL
- Strips query strings (privacy)
- Maps domain to category
- Creates time features
"""

import pandas as pd
import os
import yaml

try:
    import tldextract
    USE_TLDEXTRACT = True
except ImportError:
    USE_TLDEXTRACT = False
    from urllib.parse import urlparse

# Built-in domain → category map (used if domain_category_map.csv missing)
DEFAULT_CATEGORY_MAP = {
    "youtube.com": "video", "netflix.com": "video", "primevideo.com": "video",
    "hotstar.com": "video", "twitch.tv": "video", "vimeo.com": "video",
    "instagram.com": "social", "twitter.com": "social", "facebook.com": "social",
    "reddit.com": "social", "snapchat.com": "social", "pinterest.com": "social",
    "github.com": "learning", "stackoverflow.com": "learning",
    "coursera.org": "learning", "kaggle.com": "learning", "medium.com": "learning",
    "udemy.com": "learning", "edx.org": "learning", "khanacademy.org": "learning",
    "geeksforgeeks.org": "learning", "w3schools.com": "learning",
    "amazon.com": "shopping", "flipkart.com": "shopping", "myntra.com": "shopping",
    "meesho.com": "shopping", "nykaa.com": "shopping",
    "gmail.com": "email", "outlook.com": "email", "yahoo.com": "email",
    "google.com": "search", "bing.com": "search", "duckduckgo.com": "search",
    "linkedin.com": "professional", "naukri.com": "professional",
    "chatgpt.com": "ai_tools", "claude.ai": "ai_tools", "gemini.google.com": "ai_tools",
    "news.google.com": "news", "bbc.com": "news", "ndtv.com": "news",
    "thehindu.com": "news", "timesofindia.com": "news", "cnn.com": "news",
    "whatsapp.com": "messaging", "discord.com": "messaging", "telegram.org": "messaging",
    "docs.google.com": "productivity", "notion.so": "productivity",
    "figma.com": "productivity", "canva.com": "productivity",
    "spotify.com": "music", "gaana.com": "music", "jiosaavn.com": "music",
}


def extract_domain(url: str) -> str:
    """Extract registered domain from URL, stripping paths and query strings."""
    try:
        if USE_TLDEXTRACT:
            ext = tldextract.extract(url)
            domain = ext.registered_domain
        else:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
        return domain.lower() if domain else "unknown"
    except Exception:
        return "unknown"


def load_category_map(map_path: str) -> dict:
    if os.path.exists(map_path):
        df = pd.read_csv(map_path)
        return dict(zip(df['domain'], df['category']))
    return DEFAULT_CATEGORY_MAP


def preprocess(input_path: str, output_path: str, map_path: str) -> pd.DataFrame:
    print("Preprocessing browsing history...")

    if not os.path.exists(input_path):
        print(f"  ❌ Input file not found: {input_path}")
        print("  → Run generate_sample.py or extract_history.py first.")
        return None

    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} records")

    # If 'url' column exists, extract domain (raw history)
    if 'url' in df.columns:
        df['domain'] = df['url'].apply(extract_domain)
        df = df.drop(columns=['url'], errors='ignore')

    # Drop unknowns and duplicates
    df = df[df['domain'] != 'unknown'].drop_duplicates(subset=['timestamp', 'domain'])

    # Load category map
    cat_map = load_category_map(map_path)
    df['category'] = df['domain'].map(cat_map).fillna("other")

    # Time features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    df['day_name'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_name'].isin(['Saturday', 'Sunday']).astype(int)

    # Time block labels
    def time_block(h):
        if 5 <= h < 12: return "morning"
        if 12 <= h < 17: return "afternoon"
        if 17 <= h < 21: return "evening"
        return "late_night"

    df['time_block'] = df['hour'].apply(time_block)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(df)} clean records → {output_path}")
    print(f"  Categories found: {df['category'].unique().tolist()}")
    return df


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Use raw if available, else use already-processed
    raw = cfg['paths']['raw_history']
    src = raw if os.path.exists(raw) else cfg['paths']['clean_history']

    preprocess(
        input_path=src,
        output_path=cfg['paths']['clean_history'],
        map_path=cfg['paths']['domain_map']
    )

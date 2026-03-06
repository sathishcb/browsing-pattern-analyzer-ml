# DS105 Final Project
## Time-Based Browsing Pattern Analyzer using Deep Learning with RAM Usage Correlation

---

## 📌 Overview

This project analyzes your personal browsing history to uncover:
- Time-based browsing patterns (hour/day/session level)
- Dominant website categories
- Session behavior clusters
- Deep learning predictions / anomaly detection
- RAM usage correlation with browsing behavior

---

## 🗂️ Project Structure

```
DS105_Project/
├── config.yaml                  # All configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── main.py                      # Run full pipeline
├── src/
│   ├── collect/
│   │   ├── extract_history.py   # Browser SQLite extraction
│   │   ├── ram_logger.py        # RAM monitoring
│   │   └── generate_sample.py   # Sample data generator (for testing)
│   ├── prep/
│   │   ├── preprocess.py        # URL cleaning, category mapping
│   │   └── sessionize.py        # Session building
│   ├── models/
│   │   ├── clustering.py        # KMeans/GMM/DBSCAN
│   │   ├── lstm_model.py        # LSTM next-category prediction
│   │   └── autoencoder.py       # Autoencoder anomaly detection
│   └── analytics/
│       ├── ram_correlation.py   # RAM + browsing merge & analysis
│       ├── recommendations.py   # Actionable recommendations
│       └── visualizations.py    # All plots
├── data/                        # CSV datasets (gitignored for privacy)
├── notebooks/
│   └── DS105_Analysis.ipynb     # Jupyter notebook walkthrough
├── reports/
│   └── final_report.md          # Auto-generated report
└── dashboard.py                 # Streamlit dashboard
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project

```bash
cd DS105_Project
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: Full Pipeline (Recommended)

```bash
python main.py
```

This runs all 7 modules in sequence automatically.

### Option B: Step by Step

```bash
# Step 1: Generate sample data (if no real browser data)
python src/collect/generate_sample.py

# Step 2: (Optional) Extract real Chrome history
python src/collect/extract_history.py

# Step 3: (Optional) Start RAM logger in background
python src/collect/ram_logger.py

# Step 4: Preprocess
python src/prep/preprocess.py

# Step 5: Sessionize
python src/prep/sessionize.py

# Step 6: Cluster
python src/models/clustering.py

# Step 7: Deep Learning
python src/models/autoencoder.py
# OR
python src/models/lstm_model.py

# Step 8: Recommendations
python src/analytics/recommendations.py

# Step 9: Generate visualizations
python src/analytics/visualizations.py
```

### Option C: Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `data/browsing_history.csv` | Cleaned browsing history (domain-level) |
| `data/sessions.csv` | Session-level features |
| `data/sessions_clustered.csv` | Sessions with cluster labels |
| `data/merged_browsing_ram.csv` | Browsing + RAM merged |
| `reports/final_report.md` | Auto-generated analysis report |

---

## 🔒 Privacy Rules

- Raw URLs are **never stored** — only domain names
- Query strings and paths are stripped
- Do NOT upload `data/` folder to GitHub
- Add `data/` to `.gitignore`

---

## 📁 Git Workflow

```bash
git init
git add src/ config.yaml requirements.txt README.md main.py dashboard.py notebooks/
git commit -m "feat: initial project setup"

# Good commit messages:
# feat: add sessionization
# fix: url parsing bug
# exp: try GRU model
# docs: update README
```

---

## 🧑‍💻 Tech Stack

`Python` `Pandas` `NumPy` `SQLite` `psutil` `TensorFlow/Keras` `scikit-learn` `Streamlit` `Plotly` `Matplotlib`

---

## 📬 Doubt Sessions

- **DS/AIML:** Mon–Sat, 3:30–4:30 PM (book by 12 PM same day)
- **Live Evaluation:** Mon–Sat, 5:30–7:00 PM

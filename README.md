# ⏱️ Time-Based Browsing Pattern Analyzer

### Deep Learning + RAM Usage Correlation for User Behavior Analytics

---

## 📌 Overview

The **Time-Based Browsing Pattern Analyzer** is an AI-powered system that analyzes browser history and system RAM usage to understand **user browsing behavior over time**.

The system detects:

* ⏰ Time-based browsing patterns (hour/day/session)
* 🌐 Dominant website categories
* 📊 Browsing session clusters using ML
* 🤖 Deep learning predictions & anomaly detection
* 💾 Correlation between browsing behavior and RAM usage
* 💡 Actionable recommendations to improve productivity

This project demonstrates **behavior analytics, machine learning, deep learning, and system monitoring** in a real-world application.

---

## 🎯 Project Objectives

The goal of this project is to build an **AI system that analyzes browsing activity for the last 3–5 days and identifies:**

1. Browsing patterns based on time
2. Dominant website categories
3. User behavior clusters
4. Unusual browsing sessions (anomaly detection)
5. RAM usage correlation with browsing activity

---

## 🗂️ Project Structure

```
Time-Based-Browsing-Pattern-Analyzer/
│
├── config.yaml                  # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── main.py                      # Run the full pipeline
│
├── src/
│   ├── collect/
│   │   ├── extract_history.py   # Extract browser history from SQLite
│   │   ├── ram_logger.py        # System RAM monitoring
│   │   └── generate_sample.py   # Generate sample dataset
│   │
│   ├── prep/
│   │   ├── preprocess.py        # URL cleaning + feature creation
│   │   └── sessionize.py        # Session generation
│   │
│   ├── models/
│   │   ├── clustering.py        # KMeans / GMM / DBSCAN
│   │   ├── lstm_model.py        # LSTM next-category prediction
│   │   └── autoencoder.py       # Anomaly detection model
│   │
│   └── analytics/
│       ├── ram_correlation.py   # RAM usage correlation analysis
│       ├── recommendations.py   # Productivity recommendations
│       └── visualizations.py    # Charts and graphs
│
├── data/                        # Processed datasets (gitignored)
│
├── notebooks/
│   └── DS105_Analysis.ipynb     # Jupyter analysis notebook
│
├── reports/
│   └── final_report.md          # Auto-generated report
│
└── dashboard.py                 # Streamlit dashboard
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/browsing-pattern-analyzer-ml.git
cd browsing-pattern-analyzer-ml
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```
venv\Scripts\activate
```

**Mac / Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### Option A — Run Full Pipeline (Recommended)

```
python main.py
```

This executes all modules automatically:

1. Data collection
2. Data preprocessing
3. Session creation
4. Clustering analysis
5. Deep learning model
6. RAM correlation analysis
7. Recommendation generation

---

### Option B — Run Modules Individually

```
# Generate sample data
python src/collect/generate_sample.py

# Extract browser history
python src/collect/extract_history.py

# Start RAM monitoring
python src/collect/ram_logger.py

# Preprocess browsing data
python src/prep/preprocess.py

# Create browsing sessions
python src/prep/sessionize.py

# Run clustering
python src/models/clustering.py

# Run deep learning model
python src/models/autoencoder.py
# OR
python src/models/lstm_model.py

# Generate recommendations
python src/analytics/recommendations.py

# Generate visualizations
python src/analytics/visualizations.py
```

---

### Option C — Launch Interactive Dashboard

```
streamlit run dashboard.py
```

The dashboard shows:

* Hourly browsing activity
* Top visited domains
* Cluster visualization
* RAM usage trends
* Behavior insights

---

## 📊 Output Files

| File                           | Description              |
| ------------------------------ | ------------------------ |
| `data/browsing_history.csv`    | Cleaned browsing history |
| `data/sessions.csv`            | Session-level features   |
| `data/sessions_clustered.csv`  | Clustered sessions       |
| `data/merged_browsing_ram.csv` | RAM + browsing merged    |
| `reports/final_report.md`      | Generated project report |

---

## 🔒 Privacy & Data Protection

To protect user privacy:

* Only **domain names are stored**
* URL paths and query parameters are removed
* Raw browsing history is **never uploaded**
* The `data/` folder is excluded using `.gitignore`

---

## 🧑‍💻 Tech Stack

| Category          | Tools              |
| ----------------- | ------------------ |
| Programming       | Python             |
| Data Processing   | Pandas, NumPy      |
| Machine Learning  | Scikit-learn       |
| Deep Learning     | TensorFlow / Keras |
| Visualization     | Matplotlib, Plotly |
| Dashboard         | Streamlit          |
| System Monitoring | psutil             |
| Database          | SQLite             |

---

## 📈 Key Features

✔ Browsing history extraction
✔ Time-based session analysis
✔ Behavior clustering
✔ Deep learning anomaly detection
✔ RAM usage monitoring
✔ Interactive Streamlit dashboard

---

## 📚 Learning Outcomes

This project demonstrates practical experience in:

* Behavior analytics
* Feature engineering
* Sessionization techniques
* Unsupervised machine learning
* Deep learning for anomaly detection
* System resource monitoring
* Data visualization & dashboards

---

## 👨‍💻 Author

**Sathishkumar CB**

Machine Learning & Data Science Enthusiast

---

## ⭐ If You Like This Project

Give the repository a ⭐ on GitHub!

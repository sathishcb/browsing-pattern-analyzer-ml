# dashboard.py
"""
Streamlit Dashboard for DS105 Browsing Pattern Analyzer.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Browsing Pattern Analyzer",
    page_icon="🌐",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .rec-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = {}
    if os.path.exists("data/browsing_history.csv"):
        data['browsing'] = pd.read_csv("data/browsing_history.csv")
        data['browsing']['timestamp'] = pd.to_datetime(data['browsing']['timestamp'])
    if os.path.exists("data/sessions_clustered.csv"):
        data['sessions'] = pd.read_csv("data/sessions_clustered.csv")
    elif os.path.exists("data/sessions.csv"):
        data['sessions'] = pd.read_csv("data/sessions.csv")
    if os.path.exists("data/ram_by_category.csv"):
        data['ram_cat'] = pd.read_csv("data/ram_by_category.csv", index_col=0)
    if os.path.exists("data/ram_log.csv"):
        data['ram_log'] = pd.read_csv("data/ram_log.csv")
        data['ram_log']['timestamp'] = pd.to_datetime(data['ram_log']['timestamp'])
    if os.path.exists("data/recommendations.csv"):
        data['recs'] = pd.read_csv("data/recommendations.csv")
    return data


data = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🌐 Browsing Analyzer")
st.sidebar.markdown("---")

if 'browsing' in data:
    df = data['browsing']
    categories = ['All'] + sorted(df['category'].unique().tolist())
    selected_cat = st.sidebar.selectbox("Filter by Category", categories)
    hours = st.sidebar.slider("Hour Range", 0, 23, (0, 23))

    if selected_cat != 'All':
        df = df[df['category'] == selected_cat]
    df = df[(df['hour'] >= hours[0]) & (df['hour'] <= hours[1])]
else:
    df = pd.DataFrame()
    st.sidebar.warning("No data loaded. Run `python main.py` first.")

st.sidebar.markdown("---")
st.sidebar.info("Run `python main.py` to generate/refresh data.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌐 Time-Based Browsing Pattern Analyzer")
st.caption("DS105 Final Project — Behavior Analytics with Deep Learning & RAM Correlation")
st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────────────────────
if len(df) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Visits", f"{len(df):,}")
    col2.metric("Unique Domains", df['domain'].nunique())
    col3.metric("Categories", df['category'].nunique())
    if 'sessions' in data:
        col4.metric("Sessions", len(data['sessions']))
        col5.metric("Avg Session", f"{data['sessions']['duration_min'].mean():.0f} min")
    else:
        col4.metric("Sessions", "–")
        col5.metric("Avg Session", "–")
    st.markdown("---")

# ── Tab Layout ─────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "⏰ Time Patterns", "🔵 Clusters", "💾 RAM", "🤖 AI Insights", "💡 Recommendations"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tabs[0]:
    if len(df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Category Distribution")
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'visits']
            fig = px.pie(cat_counts, names='category', values='visits',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 Domains")
            top10 = df['domain'].value_counts().head(10).reset_index()
            top10.columns = ['domain', 'visits']
            fig = px.bar(top10, x='visits', y='domain', orientation='h',
                         color='visits', color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Browsing Activity")
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby(['date', 'category'])['domain'].count().reset_index()
        daily.columns = ['date', 'category', 'visits']
        fig = px.bar(daily, x='date', y='visits', color='category',
                     barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Time Patterns ──────────────────────────────────────────────────────
with tabs[1]:
    if len(df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hourly Activity")
            hourly = df.groupby('hour')['domain'].count().reset_index()
            hourly.columns = ['hour', 'visits']
            fig = px.bar(hourly, x='hour', y='visits',
                         color='visits', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Activity by Day of Week")
            if 'day_name' in df.columns:
                order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = df['day_name'].value_counts().reindex(order, fill_value=0).reset_index()
                day_counts.columns = ['day', 'visits']
                fig = px.bar(day_counts, x='day', y='visits', color='visits',
                             color_continuous_scale='Purples')
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Browsing Heatmap (Day × Hour)")
        if 'day_name' in df.columns:
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot = df.groupby(['day_name', 'hour'])['domain'].count().unstack(fill_value=0)
            pivot = pivot.reindex([d for d in order if d in pivot.index])
            fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                            labels={'x': 'Hour', 'y': 'Day', 'color': 'Visits'})
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Clusters ────────────────────────────────────────────────────────────
with tabs[2]:
    if 'sessions' in data and 'cluster_label' in data['sessions'].columns:
        sessions = data['sessions']
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Distribution")
            lc = sessions['cluster_label'].value_counts().reset_index()
            lc.columns = ['label', 'count']
            fig = px.pie(lc, names='label', values='count',
                         color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Cluster Scatter: Duration vs Social Ratio")
            fig = px.scatter(sessions, x='duration_min', y='social_ratio',
                             color='cluster_label', size='total_visits',
                             hover_data=['top_category', 'hour'],
                             color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Profiles")
        features = ['duration_min', 'social_ratio', 'learning_ratio',
                    'video_ratio', 'switching_rate', 'total_visits']
        features = [f for f in features if f in sessions.columns]
        profile = sessions.groupby('cluster_label')[features].mean().round(3)
        st.dataframe(profile, use_container_width=True)
    else:
        st.info("Run clustering first: `python src/models/clustering.py`")

# ── Tab 4: RAM ─────────────────────────────────────────────────────────────────
with tabs[3]:
    if 'ram_cat' in data:
        st.subheader("Browser RAM by Category")
        ram_df = data['ram_cat'].reset_index()
        fig = go.Figure()
        fig.add_bar(x=ram_df['category'], y=ram_df['mean_browser_ram_mb'],
                    name='Mean RAM', marker_color='steelblue')
        fig.add_bar(x=ram_df['category'], y=ram_df['peak_browser_ram_mb'],
                    name='Peak RAM', marker_color='tomato')
        fig.update_layout(barmode='group', xaxis_title='Category',
                           yaxis_title='Browser RAM (MB)')
        st.plotly_chart(fig, use_container_width=True)

    if 'ram_log' in data:
        st.subheader("RAM Over Time")
        ram_log = data['ram_log'].sort_values('timestamp')
        fig = px.line(ram_log, x='timestamp', y=['browser_ram_mb', 'ram_used_mb'],
                      labels={'value': 'RAM (MB)', 'variable': 'Type'})
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: AI Insights ────────────────────────────────────────────────────────
with tabs[4]:
    if 'sessions' in data and 'is_anomaly' in data['sessions'].columns:
        sessions = data['sessions']
        n_anom = sessions['is_anomaly'].sum()

        st.subheader(f"Anomaly Detection Results — {n_anom} anomalous sessions detected")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(sessions, x='reconstruction_error', nbins=30,
                               color='is_anomaly', color_discrete_map={0: 'steelblue', 1: 'red'},
                               title='Reconstruction Error Distribution')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(sessions, x='hour', y='reconstruction_error',
                             color='is_anomaly', color_discrete_map={0: 'steelblue', 1: 'red'},
                             title='Anomalous Sessions by Hour',
                             hover_data=['top_category', 'duration_min'])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Anomalous Sessions")
        anom_sessions = sessions[sessions['is_anomaly'] == 1].nlargest(10, 'reconstruction_error')
        cols = ['session_id', 'start_time', 'hour', 'duration_min',
                'top_category', 'social_ratio', 'switching_rate', 'reconstruction_error']
        cols = [c for c in cols if c in anom_sessions.columns]
        st.dataframe(anom_sessions[cols], use_container_width=True)
    else:
        st.info("Run the autoencoder to see anomaly results: `python src/models/autoencoder.py`")

# ── Tab 6: Recommendations ────────────────────────────────────────────────────
with tabs[5]:
    if 'recs' in data:
        st.subheader("💡 Actionable Recommendations")
        for _, row in data['recs'].iterrows():
            icon = row.get('icon', '•')
            title = row.get('title', '')
            rec_type = row.get('type', '')
            evidence = row.get('evidence', '')
            recommendation = row.get('recommendation', '')
            st.markdown(f"""
            <div class="rec-card">
                <strong>{icon} {title}</strong> <code>{rec_type}</code><br>
                <small>📊 {evidence}</small><br>
                ✅ {recommendation}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run recommendations: `python src/analytics/recommendations.py`")

st.markdown("---")
st.caption("DS105 Final Project · Browsing Pattern Analyzer · Built with Python + Streamlit")

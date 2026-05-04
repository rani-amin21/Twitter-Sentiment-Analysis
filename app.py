import streamlit as st
import re
import pandas as pd
from datetime import datetime
from faker import Faker
import random
import plotly.express as px
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TweetInsightsAI · Twitter Sentiment Analysis",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
.main { background: #f4f6f9 !important; padding-top: 0 !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }
[data-testid="stAppViewContainer"] { background: #f4f6f9 !important; }
[data-testid="stMain"] { background: #f4f6f9 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

/* ── NAVBAR ── */
.navbar {
    background: white; border-radius: 14px;
    padding: 14px 28px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
    margin-bottom: 20px; display: flex;
    align-items: center; justify-content: space-between;
}
.navbar-brand { display: flex; align-items: center; gap: 12px; }
.navbar-logo {
    width: 36px; height: 36px; background: #1d9bf0; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 18px; font-weight: 800;
}
.navbar-title { font-size: 16px; font-weight: 700; color: #0f1419; line-height: 1.2; }
.navbar-sub   { font-size: 11px; color: #6e7681; }

/* ── BUTTONS ── */
.stButton > button {
    background-color: #1d9bf0 !important; color: white !important; border-radius: 50px !important; font-weight: 600 !important;
    font-size: 13px !important; transition: all .2s !important;
    padding: 8px 20px !important; border: none !important;
}

.stButton > button:hover {
    background-color: #0c85d0 !important;
    color: white !important;
}

div[data-testid="column"] .stButton > button {
    width: 100% !important;
}

/* ── CARD ── */
.card {
    background: white; border-radius: 16px; padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07); border: 1px solid #f0f2f5;
    margin-bottom: 20px;
}
.card-sm {
    background: white; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06); border: 1px solid #f0f2f5;
}

/* ── BADGE ── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 50px; font-weight: 700; font-size: 12px;
}
.badge-pos { background: #dcfce7; color: #15803d; }
.badge-neg { background: #fee2e2; color: #b91c1c; }
.badge-neu { background: #fef3c7; color: #b45309; }
.badge-lg  { padding: 8px 18px; font-size: 14px; }

/* ── HERO ── */
.hero { text-align: center !important; padding: 50px 20px 30px !important; width: 100% !important; }
.hero p, .hero h1, .hero-sub, .hero-badge { text-align: center !important; margin-left: auto !important; margin-right: auto !important; }
.hero-badge {
    display: inline-block; background: #eff6ff; color: #1d9bf0;
    border: 1px solid #bfdbfe; border-radius: 20px; padding: 4px 16px;
    font-size: 12px; font-weight: 600; margin-bottom: 20px;
}

.hero-sub { font-size: 17px; color: #6e7681; max-width: 560px; margin: 0 auto 32px; line-height: 1.6; }

/* ── FEATURE CARDS ── */
.feature-grid { display: flex; justify-content: center; gap: 24px; margin: 40px 0; flex-wrap: wrap; }
.feature-card {
    background: white; border-radius: 16px; padding: 28px; width: 300px;
    border: 1px solid #f0f2f5; box-shadow: 0 1px 3px rgba(0,0,0,.05); text-align: center;
}
.feature-icon {
    width: 44px; height: 44px; border-radius: 12px; background: #eff6ff;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin: 0 auto 14px;
}
.feature-title { font-weight: 700; font-size: 16px; color: #0f1419; margin-bottom: 6px; }
.feature-desc  { font-size: 14px; color: #6e7681; line-height: 1.5; }

/* ── CTA BANNER ── */
.cta-banner {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-radius: 20px; padding: 50px 40px; text-align: center;
    margin: 30px 0; border: 1px solid #bfdbfe;
}
.cta-banner h2 { font-size: 30px; font-weight: 800; color: #0f1419; margin: 0 0 10px; }
.cta-banner p  { color: #4b5563; margin-bottom: 26px; font-size: 15px; }

/* ── PAGE TITLE ── */
.page-title { font-size: 28px; font-weight: 800; color: #0f1419; margin-bottom: 4px; }
.page-sub   { font-size: 14px; color: #6e7681; margin-bottom: 24px; }

/* ── SCORE/STAT CARDS ── */
.stat-card { background: white; border-radius: 14px; padding: 18px 22px; border: 1px solid #f0f2f5; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.stat-label { font-size: 11px; font-weight: 600; color: #6e7681; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
.stat-value { font-size: 28px; font-weight: 800; }
.stat-sub   { font-size: 12px; color: #9ca3af; margin-top: 4px; }

/* ── SCORE BOX ── */
.score-box {
    background: #f8fafc; border-radius: 12px; padding: 14px 18px;
    border: 1px solid #e5e7eb; margin-bottom: 12px;
}
.score-box-label { font-size: 12px; color: #6e7681; font-weight: 500; margin-bottom: 4px; }
.score-box-val   { font-size: 28px; font-weight: 800; }

/* ── PROGRESS BAR ── */
.progress-row { margin-bottom: 14px; }
.progress-label { display: flex; justify-content: space-between; font-size: 13px; font-weight: 500; color: #374151; margin-bottom: 6px; }
.progress-bg    { height: 8px; background: #f3f4f6; border-radius: 99px; overflow: hidden; }
.progress-fill  { height: 100%; border-radius: 99px; transition: width .4s ease; }

/* ── OVERALL SCORE BAR ── */
.overall-bar-labels { display: flex; justify-content: space-between; font-size: 11px; color: #9ca3af; margin: 10px 0 6px; }
.overall-bar-bg     { height: 10px; background: #f3f4f6; border-radius: 99px; overflow: hidden; }

/* ── RESULT EMPTY STATE ── */
.result-empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 220px; color: #9ca3af; gap: 14px;
    text-align: center;
}
.result-empty-icon { font-size: 40px; }
.result-empty-text { font-size: 14px; }

/* ── RECENT ITEM ── */
.recent-item {
    display: flex; align-items: center; gap: 14px; padding: 13px 0;
    border-bottom: 1px solid #f3f4f6;
}
.recent-item:last-child { border-bottom: none; }
.recent-tweet { font-size: 14px; color: #0f1419; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 500px; }
.recent-time  { font-size: 12px; color: #9ca3af; margin-top: 2px; }
.recent-score { font-size: 14px; font-weight: 700; min-width: 36px; text-align: right; }

/* ── BULK TABLE ── */
.results-table { width: 100%; border-collapse: collapse; }
.results-table th {
    text-align: left; padding: 10px 12px; font-size: 13px;
    color: #6e7681; font-weight: 600; border-bottom: 2px solid #f3f4f6;
}
.results-table th:last-child { text-align: right; }
.results-table td { padding: 12px; font-size: 14px; color: #0f1419; border-bottom: 1px solid #f9fafb; }
.results-table td:last-child { text-align: right; font-weight: 700; }

/* ── SEND TO DASHBOARD BANNER ── */
.dash-banner {
    background: white; border-radius: 16px; padding: 20px 24px;
    border: 1px solid #f0f2f5; box-shadow: 0 1px 4px rgba(0,0,0,.07);
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px; flex-wrap: wrap; gap: 14px;
}
.dash-banner-title { font-weight: 600; font-size: 15px; color: #0f1419; }
.dash-banner-sub   { font-size: 13px; color: #6e7681; }

/* ── KEYWORD TAG ── */
.kw-tag {
    display: inline-block; background: #f3f4f6; border: 1px solid #e5e7eb;
    border-radius: 6px; padding: 3px 9px; font-size: 12px; font-weight: 500;
    color: #374151; margin: 3px;
}

/* ── EMPTY STATE ── */
.empty-state { text-align: center; padding: 70px 20px; }
.empty-icon  { font-size: 52px; margin-bottom: 16px; }
.empty-title { font-size: 20px; font-weight: 700; color: #0f1419; margin-bottom: 8px; }
.empty-sub   { font-size: 14px; color: #6e7681; margin-bottom: 24px; }

/* ── OVERALL SENTIMENT ROW ── */
.overall-row {
    background: white; border-radius: 16px; padding: 20px 24px;
    border: 1px solid #f0f2f5; box-shadow: 0 1px 4px rgba(0,0,0,.07);
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
}
.overall-row-title { font-weight: 700; font-size: 16px; color: #0f1419; }
.overall-row-sub   { font-size: 13px; color: #6e7681; }

/* ── INSIGHTS RECENT LIST ── */
.ins-recent-item {
    display: flex; align-items: center; gap: 14px; padding: 13px 0;
    border-bottom: 1px solid #f3f4f6;
}
.ins-recent-item:last-child { border-bottom: none; }

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 28px; font-size: 12px; color: #9ca3af;
    border-top: 1px solid #f0f2f5; margin-top: 40px;
}

/* ── SECTION TITLE ── */
.section-title { font-size: 16px; font-weight: 700; color: #0f1419; margin-bottom: 4px; }
.section-sub   { font-size: 13px; color: #6e7681; margin-bottom: 12px; }

/* ── TEXTAREA ── */
textarea, .stTextArea textarea {
    border-radius: 12px !important; border: 1.5px solid #e5e7eb !important;
    padding: 14px !important; font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: #1d9bf0 !important;
    box-shadow: 0 0 0 3px rgba(29,155,240,.12) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #e5e7eb !important; border-radius: 12px !important;
    background: #fafafa !important;
}

/* ── DATAFRAME ── */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── ALERTS ── */
.stAlert { border-radius: 12px !important; }

/* ── SEARCH INPUT ── */
.stTextInput input {
    border-radius: 8px !important; border: 1.5px solid #e5e7eb !important;
    font-size: 13px !important;
}

/* Nav button overrides */
/* REPLACE the existing nav-btn CSS block with: */
.nav-btn .stButton > button {
    background: #e6f2ff !important;
    color: #1d9bf0 !important;
    box-shadow: none !important;
    border: none !important;
    font-size: 13px !important;
    padding: 7px 12px !important;
    width: 100% !important;
    white-space: nowrap !important;
    border-radius: 999px !important;
}

.nav-btn-active .stButton > button {
    background: #1d9bf0 !important;
    color: white !important;
    box-shadow: none !important;
    border: none !important;
    font-size: 13px !important;
    padding: 7px 12px !important;
    width: 100% !important;
    white-space: nowrap !important;
    border-radius: 999px !important;
}
}
/* REMOVE the div[data-testid="column"]:has(.nav-btn) block entirely */

/* ADD this new rule after the existing .nav-btn rules: */

.nav-btn > button, .nav-btn-active > button {
    white-space: nowrap !important;
    min-width: 80px !important;
}


</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VADER-LIKE SENTIMENT ENGINE (Pure Python)
# ─────────────────────────────────────────────
POSITIVE_WORDS = {
    "love":3,"great":3,"excellent":3,"amazing":3,"wonderful":3,"fantastic":3,"best":3,"awesome":3,
    "happy":2,"good":2,"nice":2,"beautiful":2,"brilliant":2,"positive":2,"strong":2,"growth":2,
    "stable":2,"steady":2,"increasing":2,"performing":2,"well":2,"profit":2,"profits":2,
    "blessed":3,"progress":2,"hardworking":2,"friend":1,"healthy":2,"long":1,"service":1,
    "clean":2,"fast":2,"mutual":1,"funds":1,"investing":1,"recover":2,"boom":2,"surge":2,
    "outstanding":3,"superb":3,"perfect":3,"exceptional":3,"impressive":2,"remarkable":2,
}
NEGATIVE_WORDS = {
    "hate":3,"terrible":3,"worst":3,"awful":3,"horrible":3,"bad":2,"poor":2,"ugly":2,
    "crash":3,"falling":2,"loss":2,"losses":2,"risky":2,"risk":2,"failed":2,
    "negative":2,"problem":2,"issue":2,"disappointing":2,"disaster":2,"crisis":2,
    "corrupt":2,"fraud":2,"scam":2,"heavy":1,"decline":2,"drop":2,"plunge":2,
    "slump":2,"weak":2,"volatile":2,"uncertain":2,"fear":2,"panic":2,"collapse":2,
}
NEGATORS = {"not","no","never","neither","nor","barely","hardly","scarcely"}
INTENSIFIERS = {"very":1.3,"really":1.2,"extremely":1.5,"absolutely":1.4,"totally":1.3,"quite":1.1,"so":1.2}
STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","of",
    "in","on","at","to","for","with","by","from","up","about","into","through",
    "during","before","after","above","below","between","out","off","over","under",
    "again","further","then","once","and","but","or","nor","so","yet","both",
    "either","whether","each","few","more","most","other","some","such","than",
    "too","very","just","as","this","that","these","those","i","my","we","our",
    "you","your","he","his","she","her","it","its","they","their","what","which",
    "who","all","per","said","also","can","us","am","its",
}

def vader_score(text):
    lower = text.lower()
    words = re.sub(r"[^a-z\s]", "", lower).split()
    pos_sum, neg_sum, neu_count = 0.0, 0.0, 0
    for i, w in enumerate(words):
        prev  = words[i-1] if i > 0 else ""
        prev2 = words[i-2] if i > 1 else ""
        negated   = prev in NEGATORS or prev2 in NEGATORS
        intensity = INTENSIFIERS.get(prev, 1.0)
        if w in POSITIVE_WORDS:
            v = POSITIVE_WORDS[w] * intensity
            if negated: neg_sum += v * 0.74
            else:       pos_sum += v
        elif w in NEGATIVE_WORDS:
            v = NEGATIVE_WORDS[w] * intensity
            if negated: pos_sum += v * 0.74
            else:       neg_sum += v
        elif w not in STOPWORDS and len(w) > 1:
            neu_count += 1
    excl      = text.count("!")
    caps_boost = 0.733 if text == text.upper() and len(text) > 3 else 0
    pos_sum   += min(excl, 4) * 0.292 + caps_boost
    total     = pos_sum + neg_sum + neu_count + 1
    pos = pos_sum / total
    neg = neg_sum / total
    neu = (neu_count + 1) / total
    if pos_sum > 0 or neg_sum > 0:
        diff     = pos_sum - neg_sum
        compound = diff / ((diff ** 2 + 15) ** 0.5)
    else:
        compound = 0.0
    label = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")
    return {"compound": compound, "pos": pos, "neg": neg, "neu": neu, "label": label}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def badge_html(label, size="sm"):
    cls  = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}[label]
    icon = {"Positive": "😊",        "Negative": "😤",         "Neutral": "😐"}[label]
    lg   = " badge-lg" if size == "lg" else ""
    return f'<span class="badge {cls}{lg}">{icon} {label}</span>'

def score_color(label):
    return {"Positive": "#15803d", "Negative": "#b91c1c", "Neutral": "#6e7681"}[label]

def fmt_score(score):
    return f"+{score}" if score >= 0 else str(score)

def progress_html(label, pct, color):
    return f"""
    <div class="progress-row">
        <div class="progress-label"><span>{label}</span><span>{pct}%</span></div>
        <div class="progress-bg">
            <div class="progress-fill" style="width:{pct}%;background:{color}"></div>
        </div>
    </div>"""

def extract_keywords(entries, label):
    freq = {}
    for r in entries:
        if r["label"] == label:
            words = re.sub(r"[^a-z\s]", "", r["tweet"].lower()).split()
            for w in words:
                if w not in STOPWORDS and len(w) > 2:
                    freq[w] = freq.get(w, 0) + 1
    return sorted(freq.items(), key=lambda x: -x[1])[:8]


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page"       not in st.session_state: st.session_state.page       = "About"
if "recent"     not in st.session_state: st.session_state.recent     = []
if "bulk_data"  not in st.session_state: st.session_state.bulk_data  = None
if "bulk_tab"   not in st.session_state: st.session_state.bulk_tab   = "manual"
if "live_result" not in st.session_state: st.session_state.live_result = None
if "live_input" not in st.session_state: st.session_state.live_input  = ""


# ─────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-logo">✦</div>
        <div>
            <div class="navbar-title">TweetInsightsAI</div>
            <div class="navbar-sub">Twitter Sentiment Analysis</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Nav buttons row
nav_cols = st.columns(7)
pages = ["About", "Live Analyzer", "Bulk Analyzer", "Insights"]
labels = ["About", "Live Analyzer", "Bulk Analyzer", "Insights"]
for col, pg, lbl in zip(nav_cols, pages, labels):
    with col:
        active = st.session_state.page == pg
        btn_class = "nav-btn-active" if active else "nav-btn"
        st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
        if st.button(lbl, key=f"nav_{pg}"):
            st.session_state.page = pg
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page


# ═══════════════════════════════════════════
# ██  ABOUT  ██
# ═══════════════════════════════════════════
if page == "About":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">TweetInsightsAI</div>
        <h1 style="font-size:52px;font-weight:800;color:#000000 !important;margin:0 0 16px;line-height:1.15;">Twitter <span style="color:#1d9bf0;">Sentiment</span> Analysis</h1>
        <p class="hero-sub">A clean, interactive web app that classifies tweet sentiment in real time using
        Logistic Regression and VADER — with bulk processing and a rich visual dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    _, bc, _ = st.columns([2, 1.5, 2])
    with bc:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Try Live Analyzer", key="hero_live", use_container_width=True):
                st.session_state.page = "Live Analyzer"; st.rerun()
        with c2:
            if st.button("View Dashboard", key="hero_dash", use_container_width=True):
                st.session_state.page = "Insights"; st.rerun()

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Live Analyzer</div>
            <div class="feature-desc">Simulates real-time tweet streaming and provides dynamic sentiment trend analysis via graph.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📁</div>
            <div class="feature-title">Bulk Analyzer</div>
            <div class="feature-desc">Paste bulk tweets or upload a CSV with a 'text' column to analyze hundreds at once.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Insights Dashboard</div>
            <div class="feature-desc">Pie, line, and histogram charts plus top keywords per sentiment.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-banner">
        <h2>Ready to analyze?</h2>
        <p>Start with a single tweet, then upload a CSV to see the dashboard come alive.</p>
    </div>
    """, unsafe_allow_html=True)

    _, cb, _ = st.columns([1.5, 1.5, 1.5])
    with cb:
        ca, cc = st.columns(2)
        with ca:
            if st.button("Live Analyzer", key="cta_live", use_container_width=True):
                st.session_state.page = "Live Analyzer"; st.rerun()
        with cc:
            if st.button("Bulk Analyzer", key="cta_bulk", use_container_width=True):
                st.session_state.page = "Bulk Analyzer"; st.rerun()


# ═══════════════════════════════════════════
# ██  LIVE ANALYZER  ██
# ═══════════════════════════════════════════
elif page == "Live Analyzer":



    st.markdown('<div class="page-title">Live Tweet Stream</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Simulated real-time Tweet Analysis.</div>', unsafe_allow_html=True)

    # ----------- FAKE TWEETS -----------
    sample_tweets = [
        "I love this product, amazing experience!",
        "Worst service ever, very disappointed",
        "It was okay, nothing special",
        "Great performance and smooth usage",
        "Terrible battery life",
        "Absolutely fantastic quality!",
        "Not worth the price",
        "Average experience overall",
        "Highly recommend this!",
        "Very bad support service"
    ]

    if "tweets" not in st.session_state:
        st.session_state.tweets = [random.choice(sample_tweets) for _ in range(500)]

    # ----------- STATE INIT -----------
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    if "i" not in st.session_state:
        st.session_state.i = 0

    if "pos" not in st.session_state:
        st.session_state.pos = []
        st.session_state.neg = []
        st.session_state.neu = []
        st.session_state.p = 0
        st.session_state.n = 0
        st.session_state.u = 0

    # ----------- BUTTONS -----------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Live Stream", use_container_width=True):
            st.session_state.streaming = True

    with col2:
        if st.button("⏹ Stop Live Stream", use_container_width=True):
            st.session_state.streaming = False

    # ----------- LAYOUT -----------
    left, right = st.columns([1, 1])
    feed = left.empty()
    chart = right.empty()

    # ----------- PROCESS ONE STEP -----------
    if st.session_state.streaming and st.session_state.i < 100:

        tweet = st.session_state.tweets[st.session_state.i]

        score = analyzer.polarity_scores(tweet)

        if score['compound'] >= 0.05:
            st.session_state.p += 1
        elif score['compound'] <= -0.05:
            st.session_state.n += 1
        else:
            st.session_state.u += 1

        total = st.session_state.p + st.session_state.n + st.session_state.u

        st.session_state.pos.append((st.session_state.p / total) * 100)
        st.session_state.neg.append((st.session_state.n / total) * 100)
        st.session_state.neu.append((st.session_state.u / total) * 100)

        st.session_state.i += 1

    # ----------- SHOW FEED -----------
    with feed.container():
        st.subheader("📡 Live Tweets")

        for j in range(st.session_state.i):
            t = st.session_state.tweets[j]

            sc = analyzer.polarity_scores(t)

            if sc['compound'] >= 0.05:
                emoji = "🟢"
            elif sc['compound'] <= -0.05:
                emoji = "🔴"
            else:
                emoji = "🟡"

            st.write(f"{j + 1}. {emoji} {t}")

    # ----------- GRAPH -----------
    df = pd.DataFrame({
        "Index": list(range(1, len(st.session_state.pos) + 1)),
        "Positive": st.session_state.pos,
        "Negative": st.session_state.neg,
        "Neutral": st.session_state.neu
    })

    fig = px.area(
        df,
        x="Index",
        y=["Positive", "Negative", "Neutral"],
        color_discrete_map={
            "Positive": "green",
            "Negative": "red",
            "Neutral": "orange"
        }
    )

    with chart.container():
        st.subheader("📈 Sentiment Trend")
        st.plotly_chart(fig, use_container_width=True)

    # ----------- AUTO REFRESH -----------
    if st.session_state.streaming:
        time.sleep(0.2)  # fast
        st.rerun()

    # Recent Analyses

    ra1, ra2 = st.columns([8, 1])
    with ra1:
        st.markdown('<div class="section-title">Recent Analyses</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:13px;color:#6e7681">Your last 5 — saved automatically to the dashboard.</div>', unsafe_allow_html=True)
    with ra2:
        if st.button("Dashboard →", key="view_dash_btn"):
            st.session_state.page = "Insights"; st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if not st.session_state.recent:
        st.markdown('<div style="font-size:14px;color:#9ca3af">No analyses yet.</div>', unsafe_allow_html=True)
    else:
        for r in st.session_state.recent:
            preview = r["tweet"][:90] + ("..." if len(r["tweet"]) > 90 else "")
            c = score_color(r["label"])
            st.markdown(f"""
            <div class="recent-item">
                <div>{badge_html(r["label"])}</div>
                <div style="flex:1">
                    <div class="recent-tweet">{preview}</div>
                    <div class="recent-time">{r["time"]}</div>
                </div>
                <div class="recent-score" style="color:{c}">{fmt_score(r["score"])}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# ██  BULK ANALYZER  ██
# ═══════════════════════════════════════════
elif page == "Bulk Analyzer":
    st.markdown('<div class="page-title">Bulk Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Process bulk tweets at once via manual entry or CSV upload.</div>', unsafe_allow_html=True)

    # Tab buttons

    t1, t2, t3 = st.columns([1.3, 1.3, 7])
    with t1:
        if st.button("🔥 Manual entry", key="tab_manual"):
            st.session_state.bulk_tab = "manual"
    with t2:
        if st.button("📤 CSV upload", key="tab_csv"):
            st.session_state.bulk_tab = "csv"

    tweets = []

    if st.session_state.bulk_tab == "manual":
        bulk_text = st.text_area(
            label="bulk_input",
            label_visibility="collapsed",
            placeholder="One tweet per line...\nLove this product!\nWorst service ever\nIt was okay",
            height=160,
            key="bulk_textarea"
        )
        if bulk_text and bulk_text.strip():
            tweets = [t.strip() for t in bulk_text.split("\n") if t.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV (with 'text' column)", type=["csv"], key="csv_upload")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if "text" not in df_up.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    tweets = df_up["text"].astype(str).tolist()
                    st.success(f"✅ {len(tweets)} tweets loaded from CSV")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    bc1, _, bc2 = st.columns([3, 7, 1])
    with bc1:
        st.markdown(f'<div style="font-size:13px;color:#9ca3af;padding-top:10px">{len(tweets)} tweets ready</div>', unsafe_allow_html=True)
    with bc2:
        analyze_bulk = st.button("🚀 Analyze", key="analyze_bulk")
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_bulk:
        if not tweets:
            st.warning("Please enter tweets or upload a CSV first.")
        else:
            with st.spinner("Analyzing tweets…"):
                rows = []
                for t in tweets:
                    sc = vader_score(t)
                    rows.append({
                        "tweet": t, "label": sc["label"],
                        "score": int(sc["compound"] * 100),
                        "compound": sc["compound"]
                    })
                rows.sort(key=lambda x: -x["score"])
            st.session_state.bulk_data = rows

    if st.session_state.bulk_data:
        rows    = st.session_state.bulk_data
        total   = len(rows)
        pos_cnt = sum(1 for r in rows if r["label"] == "Positive")
        neg_cnt = sum(1 for r in rows if r["label"] == "Negative")
        neu_cnt = sum(1 for r in rows if r["label"] == "Neutral")

        # Stat cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">TOTAL</div><div class="stat-value" style="color:#0f1419">{total}</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">POSITIVE</div><div class="stat-value" style="color:#15803d">{pos_cnt}</div><div class="stat-sub">{int(pos_cnt/total*100)}% of total</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">NEGATIVE</div><div class="stat-value" style="color:#b91c1c">{neg_cnt}</div><div class="stat-sub">{int(neg_cnt/total*100)}% of total</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="stat-card"><div class="stat-label">NEUTRAL</div><div class="stat-value" style="color:#b45309">{neu_cnt}</div><div class="stat-sub">{int(neu_cnt/total*100)}% of total</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Overall sentiment
        if pos_cnt >= neg_cnt and pos_cnt >= neu_cnt:
            overall_label = "Positive"
        elif neg_cnt > pos_cnt:
            overall_label = "Negative"
        else:
            overall_label = "Neutral"

        st.markdown(f"""
        <div class="overall-row">
            <div>
                <div class="overall-row-title">Overall Sentiment</div>
                <div class="overall-row-sub">Majority across this batch</div>
            </div>
            <div>{badge_html(overall_label, "lg")}</div>
        </div>
        """, unsafe_allow_html=True)

        # Results table with search & filter
        st.markdown('<div class="card">''<div class="section-title">Results</div>', unsafe_allow_html=True)
        sr1, sr2 = st.columns([3, 4])
        with sr1:
            st.markdown('<div class="section-title"></div>', unsafe_allow_html=True)
        with sr2:
            search_q = st.text_input("", placeholder="Search tweets...", key="bulk_search", label_visibility="collapsed")

        fl1, fl2, fl3, fl4, fl5 = st.columns([1, 1, 1, 1, 4])
        bulk_filter = st.session_state.get("bulk_filter", "all")
        with fl1:
            if st.button("All", key="f_all"):   st.session_state["bulk_filter"] = "all";      st.rerun()
        with fl2:
            if st.button("Positive", key="f_pos"): st.session_state["bulk_filter"] = "Positive"; st.rerun()
        with fl3:
            if st.button("Negative", key="f_neg"): st.session_state["bulk_filter"] = "Negative"; st.rerun()
        with fl4:
            if st.button("Neutral", key="f_neu"):  st.session_state["bulk_filter"] = "Neutral";  st.rerun()

        bulk_filter = st.session_state.get("bulk_filter", "all")
        filtered = [r for r in rows
                    if (bulk_filter == "all" or r["label"] == bulk_filter)
                    and (not search_q or search_q.lower() in r["tweet"].lower())]

        st.markdown(f'<div style="font-size:13px;color:#6e7681;margin-bottom:12px">Showing {len(filtered)} of {total}</div>', unsafe_allow_html=True)

        table_rows = ""
        for r in filtered:
            c = score_color(r["label"])
            table_rows += f"""
            <tr>
                <td>{r["tweet"]}</td>
                <td>{badge_html(r["label"])}</td>
                <td style="color:{c}">{fmt_score(r["score"])}</td>
            </tr>"""

        st.markdown(f"""
        <table class="results-table">
            <thead><tr><th>Tweet</th><th>Sentiment</th><th style="text-align:right">Score ↓</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Send to dashboard
        st.markdown(f"""
        <div class="dash-banner">
            <div>
                <div class="dash-banner-title">Send these to the dashboard?</div>
                <div class="dash-banner-sub">Adds all {total} results to your saved analyses for visualization.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        db1, db2, _ = st.columns([1.3, 1.5, 6])
        with db1:
            if st.button("Open dashboard", key="open_dash"):
                st.session_state.page = "Insights"; st.rerun()
        with db2:
            if st.button("Send to dashboard", key="send_dash"):
                entries = [{
                    "tweet": r["tweet"], "label": r["label"],
                    "compound": r["compound"], "score": r["score"],
                    "time": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
                } for r in rows]
                combined = entries + st.session_state.recent
                seen = set()
                deduped = []
                for r in combined:
                    if r["tweet"] not in seen:
                        seen.add(r["tweet"]); deduped.append(r)
                st.session_state.recent = deduped
                st.session_state.page = "Insights"; st.rerun()


# ═══════════════════════════════════════════
# ██  INSIGHTS DASHBOARD  ██
# ═══════════════════════════════════════════
elif page == "Insights":
    all_data = st.session_state.recent

    col_t, col_btn = st.columns([8, 1])
    with col_t:
        st.markdown('<div class="page-title">Insights Dashboard</div>', unsafe_allow_html=True)
        if all_data:
            st.markdown(f'<div class="page-sub">Aggregated view across {len(all_data)} analyzed tweets.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="page-sub">Visual summary of all analyzed tweets.</div>', unsafe_allow_html=True)
    with col_btn:
        if all_data and st.button("Clear all data", key="clear_data"):
            st.session_state.recent = []; st.rerun()

    if not all_data:
        st.markdown("""
        <div class="card">
            <div class="empty-state">
                <div class="empty-icon">📊</div>
                <div class="empty-title">No data yet</div>
                <div class="empty-sub">Analyze some tweets first — your results will appear here as charts.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        _, ec, _ = st.columns([1.5, 1.5, 1.5])
        with ec:
            ea, eb = st.columns(2)
            with ea:
                if st.button("Live Analyzer", key="e_live"): st.session_state.page = "Live Analyzer"; st.rerun()
            with eb:
                if st.button("Bulk Analyzer", key="e_bulk"): st.session_state.page = "Bulk Analyzer"; st.rerun()
    else:
        total   = len(all_data)
        pos_cnt = sum(1 for r in all_data if r["label"] == "Positive")
        neg_cnt = sum(1 for r in all_data if r["label"] == "Negative")
        neu_cnt = sum(1 for r in all_data if r["label"] == "Neutral")

        # Stat cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">TOTAL TWEETS</div><div class="stat-value" style="color:#1d9bf0">{total}</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">POSITIVE</div><div class="stat-value" style="color:#15803d">{int(pos_cnt/total*100)}%</div><div class="stat-sub">{pos_cnt} tweets</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">NEGATIVE</div><div class="stat-value" style="color:#b91c1c">{int(neg_cnt/total*100)}%</div><div class="stat-sub">{neg_cnt} tweets</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="stat-card"><div class="stat-label">NEUTRAL</div><div class="stat-value" style="color:#b45309">{int(neu_cnt/total*100)}%</div><div class="stat-sub">{neu_cnt} tweets</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Charts row 1: Pie + Trend
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="card">''<div class="section-title">Sentiment Distribution</div>''<div class="section-sub">Share of each class across all tweets</div>', unsafe_allow_html=True)

            pie_df = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Count":     [pos_cnt, neg_cnt, neu_cnt]
            })
            import plotly.graph_objects as go
            fig_pie = go.Figure(go.Pie(
                labels=pie_df["Sentiment"], values=pie_df["Count"],
                hole=0.45, marker_colors=["#22c55e","#ef4444","#f59e0b"],
                textinfo="label+percent"
            ))
            fig_pie.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=280,
                                   showlegend=True, legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="card">''<div class="section-title">Sentiment Trend</div>''<div class="section-sub">Compound score by analysis order</div>', unsafe_allow_html=True)

            trend_scores = [r.get("score", int(r["compound"]*100)) for r in reversed(all_data)]
            trend_df = pd.DataFrame({"Index": list(range(1, len(trend_scores)+1)), "Score": trend_scores})
            import plotly.express as px
            fig_trend = px.area(trend_df, x="Index", y="Score",
                                color_discrete_sequence=["#1d9bf0"])
            fig_trend.update_traces(fill="tozeroy", fillcolor="rgba(191,219,254,0.4)")
            fig_trend.update_layout(margin=dict(t=10,b=10,l=0,r=20), height=280,
                                     yaxis=dict(range=[-100,100]), xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Charts row 2: Histogram + Keywords
        ch3, ch4 = st.columns([2, 1])

        with ch3:
            st.markdown('<div class="card">''<div class="section-title">Score Distribution</div>''<div class="section-sub">Histogram of compound scores (-1 to +1)</div>', unsafe_allow_html=True)

            buckets = [(-1.0,-0.8),(-0.8,-0.6),(-0.6,-0.4),(-0.4,-0.2),(-0.2,0.0),
                       (0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]
            hist_data = []
            for lo, hi in buckets:
                count = sum(1 for r in all_data if lo <= r["compound"] < hi)
                hist_data.append({
                    "bucket": f"{lo} to {hi}",
                    "count": count,
                    "color": "#22c55e" if hi > 0 else "#ef4444"
                })
            hist_df = pd.DataFrame(hist_data)
            fig_hist = go.Figure(go.Bar(
                x=hist_df["bucket"], y=hist_df["count"],
                marker_color=hist_df["color"]
            ))
            fig_hist.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=230,
                                    xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch4:
            st.markdown('<div class="card">''<div class="section-title">Top Keywords</div>''<div class="section-sub">Most frequent words by sentiment</div>', unsafe_allow_html=True)

            for lbl in ["Positive", "Negative", "Neutral"]:
                kws = extract_keywords(all_data, lbl)
                tags = " ".join([f'<span class="kw-tag">{w} {c}</span>' for w, c in kws]) if kws else '<span style="font-size:12px;color:#9ca3af">No data</span>'
                st.markdown(f"""
                <div style="margin-bottom:14px">
                    {badge_html(lbl)}
                    <div style="margin-top:8px">{tags}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Recent analyses list
        st.markdown('<div class="card">''<div class="section-title">Recent Analyses</div>''<div class="section-sub">Last 10 saved tweets</div>', unsafe_allow_html=True)

        for i, r in enumerate(all_data[:10]):
            c = score_color(r["label"])
            score_val = r.get("score", int(r["compound"]*100))
            tweet_preview = r["tweet"][:100] + ("..." if len(r["tweet"]) > 100 else "")
            st.markdown(f"""
            <div class="ins-recent-item">
                <div>{badge_html(r["label"])}</div>
                <div style="flex:1">
                    <div style="font-size:14px;color:#0f1419;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:700px">{tweet_preview}</div>
                    <div style="font-size:12px;color:#9ca3af;margin-top:2px">{r["time"]}</div>
                </div>
                <div style="font-size:14px;font-weight:700;color:{c};min-width:40px;text-align:right">{fmt_score(score_val)}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)



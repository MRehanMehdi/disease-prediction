import streamlit as st
import pandas as pd
from utils import (
    predict_top3_diseases, get_disease_info,
    get_severity_score, generate_pdf_report, symptom_list
)

st.set_page_config(
    page_title="MedPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme State ───────────────────────────────────────────────
if 'dark_mode'  not in st.session_state: st.session_state.dark_mode  = True
if 'history'    not in st.session_state: st.session_state.history    = []
if 'predicted'  not in st.session_state: st.session_state.predicted  = False

D = st.session_state.dark_mode

BG         = "#0f0f0f"      if D else "#ffffff"
SBG        = "#171717"      if D else "#f7f7f8"
CARD       = "#1e1e1e"      if D else "#f4f4f4"
BORDER     = "#2a2a2a"      if D else "#e0e0e0"
TEXT       = "#ececec"      if D else "#111111"
SUB        = "#8e8ea0"      if D else "#666680"
ACC        = "#10a37f"
INP        = "#262626"      if D else "#ffffff"
PREC       = "#1a2e25"      if D else "#edfaf4"
FAINT      = "#3a3a3a"      if D else "#cccccc"
SHADOW     = "rgba(0,0,0,0.3)" if D else "rgba(0,0,0,0.08)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}}

/* ── App background ── */
.stApp {{
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

/* ── Hide only deploy button, keep sidebar toggle ── */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
[data-testid="stToolbar"] {{visibility: hidden;}}
[data-testid="stDecoration"] {{display: none;}}

/* ── Streamlit header — keep visible but transparent ── */
[data-testid="stHeader"] {{
    background-color: {BG} !important;
    border-bottom: 1px solid {BORDER} !important;
    height: 50px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SBG} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div {{
    padding-top: 16px !important;
    padding-left: 14px !important;
    padding-right: 14px !important;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: {TEXT} !important;
}}

/* ── Main content ── */
.main .block-container {{
    padding-top: 30px !important;
    padding-bottom: 30px !important;
    max-width: 720px !important;
    margin: 0 auto !important;
}}

/* ── Welcome ── */
.welcome {{
    text-align: center;
    padding: 30px 0 20px 0;
}}
.welcome-icon {{
    font-size: 2.6rem;
    margin-bottom: 10px;
    display: block;
}}
.welcome-title {{
    font-size: 1.85rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}}
.welcome-sub {{
    font-size: 0.88rem;
    color: {SUB};
    margin-bottom: 18px;
    line-height: 1.55;
}}
.chips {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 8px;
}}
.chip {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 0.76rem;
    color: {SUB};
    font-weight: 500;
}}

/* ── Symptom input ── */
[data-testid="stMultiSelect"] > div {{
    background-color: {INP} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 14px !important;
    color: {TEXT} !important;
    font-size: 0.93rem !important;
    padding: 4px 6px !important;
    box-shadow: 0 2px 12px {SHADOW} !important;
    transition: border-color 0.2s !important;
}}
[data-testid="stMultiSelect"] > div:focus-within {{
    border-color: {ACC} !important;
}}

/* ── All buttons ── */
.stButton > button {{
    background-color: {ACC} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 22px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    background-color: #0e8f70 !important;
    box-shadow: 0 4px 14px rgba(16,163,127,0.35) !important;
    transform: translateY(-1px) !important;
}}

/* ── Theme + New Prediction buttons in sidebar ── */
div[data-testid="stSidebar"] .stButton > button {{
    background-color: {CARD} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    width: 100% !important;
    text-align: left !important;
}}
div[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: {BORDER} !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Section label ── */
.sec-lbl {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {SUB};
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin: 22px 0 9px 0;
}}

/* ── Divider ── */
.div-line {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 20px 0;
}}

/* ── Disease cards ── */
.d-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    transition: transform 0.15s, box-shadow 0.15s;
    height: 100%;
}}
.d-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px {SHADOW};
}}
.d-rank {{
    font-size: 0.66rem;
    font-weight: 600;
    color: {SUB};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 7px;
}}
.d-name {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {TEXT};
    line-height: 1.3;
    margin-bottom: 12px;
    min-height: 36px;
}}
.d-conf {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {ACC};
    line-height: 1;
}}
.d-conf-sub {{
    font-size: 0.66rem;
    color: {SUB};
    margin-top: 3px;
}}

/* ── Severity badge ── */
.sev {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.83rem;
    font-weight: 600;
}}
.sev-High   {{ background:#2d1515; color:#ff6b6b; border:1px solid #4a2020; }}
.sev-Medium {{ background:#2d2515; color:#ffc107; border:1px solid #4a3b20; }}
.sev-Low    {{ background:#0d2d1e; color:{ACC};   border:1px solid #1b4a35; }}

/* ── Description box ── */
.desc-box {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    font-size: 0.88rem;
    color: {TEXT};
    line-height: 1.75;
}}

/* ── Precaution item ── */
.prec {{
    background: {PREC};
    border-left: 3px solid {ACC};
    border-radius: 7px;
    padding: 9px 13px;
    margin-bottom: 7px;
    font-size: 0.86rem;
    color: {TEXT};
    line-height: 1.5;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    background: {CARD} !important;
    color: {ACC} !important;
    border: 1.5px solid {ACC} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}}
.stDownloadButton > button:hover {{
    background: {ACC} !important;
    color: #fff !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    overflow: hidden !important;
}}

/* ── History item ── */
.hist {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 5px;
    background: {CARD};
    border: 1px solid {BORDER};
    font-size: 0.81rem;
    color: {TEXT};
}}
.hist-c {{ color: {ACC}; font-weight: 600; font-size: 0.79rem; }}

/* ── Metric ── */
[data-testid="stMetricValue"] {{ color: {ACC} !important; font-weight: 700 !important; font-size: 1.3rem !important; }}
[data-testid="stMetricLabel"] {{ color: {SUB} !important; font-size: 0.75rem !important; }}

/* ── Spinner ── */
[data-testid="stSpinner"] {{ color: {ACC} !important; }}

/* ── Warning ── */
[data-testid="stAlert"] {{ border-radius: 10px !important; }}

/* ── Disclaimer / footer ── */
.disclaimer {{
    text-align: center;
    font-size: 0.75rem;
    color: {FAINT};
    line-height: 1.6;
    padding: 8px 0;
}}
.footer-txt {{
    text-align: center;
    font-size: 0.72rem;
    color: {FAINT};
    padding-top: 24px;
    padding-bottom: 8px;
}}
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return (
        pd.read_csv('data/symptom_Description.csv'),
        pd.read_csv('data/symptom_precaution.csv'),
        pd.read_csv('data/Symptom-severity.csv')
    )
desc_df, prec_df, severity_df = load_data()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo row
    st.markdown(
        f"<p style='font-size:1rem;font-weight:700;color:{TEXT};margin:0 0 4px 0'>🩺 MedPredict AI</p>"
        f"<p style='font-size:0.75rem;color:{SUB};margin:0 0 14px 0'>AI Disease Prediction System</p>",
        unsafe_allow_html=True
    )

    # Theme toggle
    theme_label = "☀️  Switch to Light Mode" if D else "🌙  Switch to Dark Mode"
    if st.button(theme_label, key="theme"):
        st.session_state.dark_mode = not D
        st.rerun()

    # New prediction
    if st.button("＋  New Prediction", key="new_pred"):
        st.session_state.predicted = False
        st.rerun()

    st.markdown(f"<hr style='border-color:{BORDER};margin:14px 0'>", unsafe_allow_html=True)

    # Dataset info
    st.markdown(
        f"<p style='font-size:0.68rem;font-weight:600;color:{SUB};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>System Info</p>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    c1.metric("Diseases", "41")
    c2.metric("Symptoms", "131")
    st.markdown(
        f"<p style='font-size:0.78rem;color:{SUB};margin-top:6px'>"
        f"Model: <span style='color:{ACC};font-weight:600'>Random Forest</span><br>"
        f"Accuracy: <span style='color:{ACC};font-weight:600'>100%</span></p>",
        unsafe_allow_html=True
    )

    st.markdown(f"<hr style='border-color:{BORDER};margin:14px 0'>", unsafe_allow_html=True)

    # History
    st.markdown(
        f"<p style='font-size:0.68rem;font-weight:600;color:{SUB};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>Recent Predictions</p>",
        unsafe_allow_html=True
    )
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            name  = h['disease']
            short = (name[:20] + '…') if len(name) > 20 else name
            st.markdown(
                f"<div class='hist'><span>{short}</span>"
                f"<span class='hist-c'>{h['confidence']}%</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f"<p style='font-size:0.8rem;color:{FAINT};font-style:italic'>No predictions yet</p>",
            unsafe_allow_html=True
        )

    st.markdown(f"<hr style='border-color:{BORDER};margin:14px 0'>", unsafe_allow_html=True)

    # Team
    st.markdown(
        f"<p style='font-size:0.73rem;color:{FAINT};line-height:1.85'>"
        f"Muhammad Rehan Mehdi<br>Raheel Nazir<br>Mohammad Sami<br><br>"
        f"<span style='color:{SUB}'>Riphah International University</span><br>"
        f"AI Course · 6th Semester BSCS · 2026</p>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

# Welcome screen
if not st.session_state.predicted:
    st.markdown(f"""
        <div class="welcome">
            <span class="welcome-icon">🩺</span>
            <div class="welcome-title">What are your symptoms?</div>
            <div class="welcome-sub">
                Select your symptoms and MedPredict AI will predict<br>
                the top 3 most probable diseases using Machine Learning.
            </div>
            <div class="chips">
                <span class="chip">🦠 41 Diseases</span>
                <span class="chip">💊 131 Symptoms</span>
                <span class="chip">🤖 Random Forest</span>
                <span class="chip">🎯 100% Accuracy</span>
                <span class="chip">📄 PDF Report</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────
selected = st.multiselect(
    label="s",
    options=sorted(symptom_list),
    placeholder="   Search and select your symptoms...",
    label_visibility="collapsed"
)

col_info, col_btn = st.columns([2, 1])
with col_info:
    if selected:
        st.markdown(
            f"<p style='color:{SUB};font-size:0.82rem;margin-top:9px'>"
            f"✓ {len(selected)} symptom{'s' if len(selected)!=1 else ''} selected</p>",
            unsafe_allow_html=True
        )
with col_btn:
    go = st.button("Analyze Symptoms →", use_container_width=True)

# ── Predict ───────────────────────────────────────────────────
if go:
    if len(selected) == 0:
        st.warning("Please select at least one symptom.")
    elif len(selected) < 2:
        st.warning("Please select at least 2 symptoms for a reliable prediction.")
    else:
        st.session_state.predicted = True
        with st.spinner("Analyzing your symptoms..."):
            top3 = predict_top3_diseases(selected)
            sev_score, risk = get_severity_score(selected, severity_df)
            top_d, top_c   = top3[0]
            desc, precs    = get_disease_info(top_d, desc_df, prec_df)

        st.session_state.history.append({'disease': top_d, 'confidence': top_c})

        st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

        # Severity
        st.markdown("<div class='sec-lbl'>Symptom Severity</div>", unsafe_allow_html=True)
        e = "🔴" if risk=="High" else "🟡" if risk=="Medium" else "🟢"
        st.markdown(
            f'<span class="sev sev-{risk}">{e} {risk} Risk &nbsp;·&nbsp; Score: {sev_score} / 7</span>',
            unsafe_allow_html=True
        )

        # Top 3 cards
        st.markdown("<div class='sec-lbl'>Predicted Diseases</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        ranks = ["🥇 Most Likely", "🥈 Possible", "🥉 Less Likely"]
        for col, (dis, conf), rank in zip([c1,c2,c3], top3, ranks):
            with col:
                st.markdown(
                    f'<div class="d-card">'
                    f'<div class="d-rank">{rank}</div>'
                    f'<div class="d-name">{dis}</div>'
                    f'<div class="d-conf">{conf}%</div>'
                    f'<div class="d-conf-sub">confidence</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # About
        st.markdown(f"<div class='sec-lbl'>About — {top_d}</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="desc-box">{desc}</div>', unsafe_allow_html=True)

        # Precautions
        if precs:
            st.markdown("<div class='sec-lbl'>Recommended Precautions</div>", unsafe_allow_html=True)
            for i, p in enumerate(precs, 1):
                st.markdown(
                    f'<div class="prec"><b>{i}.</b> {p.capitalize()}</div>',
                    unsafe_allow_html=True
                )

        # Comparison table
        st.markdown("<div class='sec-lbl'>All Predictions Comparison</div>", unsafe_allow_html=True)
        rows = []
        for dis, conf in top3:
            _, ps = get_disease_info(dis, desc_df, prec_df)
            rows.append({
                'Disease':    dis,
                'Confidence': f"{conf}%",
                'Precaution 1': ps[0].capitalize() if ps else '—',
                'Precaution 2': ps[1].capitalize() if len(ps)>1 else '—',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # PDF
        st.markdown("<div class='sec-lbl'>Export Report</div>", unsafe_allow_html=True)
        pdf = generate_pdf_report(selected, top3, desc, precs, sev_score, risk)
        st.download_button(
            "⬇  Download PDF Report", pdf,
            f"MedPredict_{top_d.replace(' ','_')}.pdf",
            "application/pdf",
            use_container_width=True
        )

        # Disclaimer
        st.markdown("<hr class='div-line'>", unsafe_allow_html=True)
        st.markdown(
            "<p class='disclaimer'>⚕️ MedPredict AI is for educational purposes only. "
            "Always consult a qualified physician for proper diagnosis.</p>",
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    "<div class='footer-txt'>"
    "MedPredict AI · Python · Scikit-learn · Streamlit<br>"
    "Muhammad Rehan Mehdi · Raheel Nazir · Mohammad Sami · Riphah International University 2026"
    "</div>",
    unsafe_allow_html=True
)
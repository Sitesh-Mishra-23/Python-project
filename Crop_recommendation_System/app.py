import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommender",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f0e;
    color: #e2ede8;
}

/* ── Background grain texture ── */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 999;
    opacity: 0.4;
}

/* ── Main container ── */
.block-container {
    padding: 2rem 4rem 4rem 4rem !important;
    max-width: 1100px !important;
}

/* ── Header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: #4dff91;
}
.hero-sub {
    font-size: 1.05rem;
    color: #7a9488;
    font-weight: 300;
    margin-top: 0.4rem;
    letter-spacing: 0.01em;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, #4dff91 0%, #1a3328 100%);
    margin: 1.8rem 0 2.2rem 0;
    border: none;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4dff91;
    margin-bottom: 1rem;
}

/* ── Slider labels ── */
.slider-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.2rem;
}
.slider-name {
    font-size: 0.88rem;
    font-weight: 500;
    color: #c4d9ce;
}
.slider-unit {
    font-size: 0.75rem;
    color: #4a6b5c;
}

/* ── Streamlit slider overrides ── */
div[data-baseweb="slider"] > div {
    padding-top: 0 !important;
}
.stSlider > label { display: none !important; }

/* ── Model toggle buttons ── */
div.stRadio > div {
    display: flex;
    flex-direction: row;
    gap: 0.8rem;
}
div.stRadio > div > label {
    background: #111a16;
    border: 1px solid #1e3028;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    cursor: pointer;
    font-size: 0.85rem;
    color: #7a9488;
    transition: all 0.2s;
}
div.stRadio > div > label:has(input:checked) {
    background: #0d2a1e;
    border-color: #4dff91;
    color: #4dff91;
}
div.stRadio > div > label > div:first-child { display: none; }

/* ── Predict button ── */
.stButton > button {
    background: #4dff91 !important;
    color: #051209 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 28px rgba(77,255,145,0.18) !important;
}
.stButton > button:hover {
    background: #6effaa !important;
    box-shadow: 0 0 40px rgba(77,255,145,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #0d2a1e 0%, #091a12 100%);
    border: 1px solid #1e4030;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-top: 1.5rem;
    box-shadow: 0 8px 40px rgba(77,255,145,0.08);
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4dff91, #00c9a7);
}
.result-crop-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #4dff91;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.3rem;
    text-transform: capitalize;
}
.result-model-tag {
    font-size: 0.75rem;
    color: #4a6b5c;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.result-conf {
    display: inline-block;
    background: rgba(77,255,145,0.08);
    border: 1px solid rgba(77,255,145,0.2);
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    color: #4dff91;
}

/* ── Info cards (crop details) ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.info-card {
    background: #0b1a12;
    border: 1px solid #1a3028;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}
.info-card-label {
    font-size: 0.68rem;
    color: #4a6b5c;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.info-card-value {
    font-size: 0.92rem;
    color: #c4d9ce;
    font-weight: 500;
}

/* ── Input card wrapper ── */
.input-section {
    background: #0d1a14;
    border: 1px solid #1a2e22;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}

/* ── Accuracy badge ── */
.acc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #0d2a1e;
    border: 1px solid #1e4030;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    color: #4dff91;
    font-weight: 500;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Crop metadata ─────────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif",  "soil": "Clayey / Loamy",   "water": "High"},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "soil": "Well-drained Loam", "water": "Medium"},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",    "soil": "Sandy Loam",        "water": "Low"},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "soil": "Loamy",             "water": "Medium"},
    "pigeonpeas":  {"emoji": "🌿", "season": "Kharif",  "soil": "Sandy Loam",        "water": "Low"},
    "mothbeans":   {"emoji": "🌱", "season": "Kharif",  "soil": "Sandy",             "water": "Very Low"},
    "mungbean":    {"emoji": "🫛", "season": "Kharif",  "soil": "Loamy Sand",        "water": "Low"},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "soil": "Loamy",             "water": "Low"},
    "lentil":      {"emoji": "🌿", "season": "Rabi",    "soil": "Loamy",             "water": "Low"},
    "pomegranate": {"emoji": "🍎", "season": "Annual",  "soil": "Deep Loamy",        "water": "Low"},
    "banana":      {"emoji": "🍌", "season": "Annual",  "soil": "Rich Loamy",        "water": "High"},
    "mango":       {"emoji": "🥭", "season": "Summer",  "soil": "Deep Alluvial",     "water": "Medium"},
    "grapes":      {"emoji": "🍇", "season": "Annual",  "soil": "Sandy Loam",        "water": "Medium"},
    "watermelon":  {"emoji": "🍉", "season": "Summer",  "soil": "Sandy Loam",        "water": "Medium"},
    "muskmelon":   {"emoji": "🍈", "season": "Summer",  "soil": "Sandy Loam",        "water": "Medium"},
    "apple":       {"emoji": "🍏", "season": "Winter",  "soil": "Well-drained Loam", "water": "Medium"},
    "orange":      {"emoji": "🍊", "season": "Winter",  "soil": "Loamy",             "water": "Medium"},
    "papaya":      {"emoji": "🍑", "season": "Annual",  "soil": "Alluvial",          "water": "Medium"},
    "coconut":     {"emoji": "🥥", "season": "Annual",  "soil": "Laterite / Loamy",  "water": "High"},
    "cotton":      {"emoji": "🌸", "season": "Kharif",  "soil": "Black Cotton Soil", "water": "Medium"},
    "jute":        {"emoji": "🌿", "season": "Kharif",  "soil": "Alluvial",          "water": "High"},
    "coffee":      {"emoji": "☕", "season": "Annual",  "soil": "Red Loamy",         "water": "Medium"},
}

MODEL_ACC = {
    "Random Forest":        "99.55%",
    "Logistic Regression":  "97.27%",
}


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for key, fname in [("Random Forest", "rf_crop_model.pkl"),
                       ("Logistic Regression", "lr_crop_model.pkl")]:
        if os.path.exists(fname):
            models[key] = joblib.load(fname)
        elif os.path.exists(f"/mnt/user-data/outputs/{fname}"):
            models[key] = joblib.load(f"/mnt/user-data/outputs/{fname}")
    return models

models = load_models()

if not models:
    st.error("⚠️  No model files found. Place `rf_crop_model.pkl` and `lr_crop_model.pkl` in the same folder as app.py.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">🌾 Crop<span>Lens</span></div>
<div class="hero-sub">Soil & climate intelligence for smarter crop decisions</div>
<hr class="hero-divider"/>
""", unsafe_allow_html=True)

# ── Layout: left (inputs) | right (result) ───────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    # ── Model selector ──
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio("model", list(models.keys()), label_visibility="collapsed",
                            horizontal=True)
    st.markdown(f'<div style="margin-bottom:1.6rem"><span class="acc-badge">✦ Test accuracy {MODEL_ACC[model_choice]}</span></div>',
                unsafe_allow_html=True)

    # ── Soil nutrients ──
    st.markdown('<div class="section-label">Soil Nutrients</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)

        st.markdown('<div class="slider-row"><span class="slider-name">Nitrogen (N)</span><span class="slider-unit">kg/ha</span></div>', unsafe_allow_html=True)
        N = st.slider("N", 0, 140, 50, key="N")

        st.markdown('<div class="slider-row"><span class="slider-name">Phosphorus (P)</span><span class="slider-unit">kg/ha</span></div>', unsafe_allow_html=True)
        P = st.slider("P", 5, 145, 53, key="P")

        st.markdown('<div class="slider-row"><span class="slider-name">Potassium (K)</span><span class="slider-unit">kg/ha</span></div>', unsafe_allow_html=True)
        K = st.slider("K", 5, 205, 48, key="K")

        st.markdown('<div class="slider-row"><span class="slider-name">Soil pH</span><span class="slider-unit">0–14</span></div>', unsafe_allow_html=True)
        ph = st.slider("ph", 3.5, 10.0, 6.5, step=0.1, key="ph")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Climate conditions ──
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Climate Conditions</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)

        st.markdown('<div class="slider-row"><span class="slider-name">Temperature</span><span class="slider-unit">°C</span></div>', unsafe_allow_html=True)
        temperature = st.slider("temperature", 8.0, 44.0, 25.0, step=0.1, key="temp")

        st.markdown('<div class="slider-row"><span class="slider-name">Humidity</span><span class="slider-unit">%</span></div>', unsafe_allow_html=True)
        humidity = st.slider("humidity", 14.0, 100.0, 71.0, step=0.1, key="hum")

        st.markdown('<div class="slider-row"><span class="slider-name">Rainfall</span><span class="slider-unit">mm</span></div>', unsafe_allow_html=True)
        rainfall = st.slider("rainfall", 20.0, 300.0, 103.0, step=0.5, key="rain")

        st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("⟶  Recommend Crop", use_container_width=True)


# ── Right panel: result ───────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Recommendation</div>', unsafe_allow_html=True)

    if predict_btn:
        bundle  = models[model_choice]
        model   = bundle["model"]
        le      = bundle["label_encoder"]
        features= bundle["features"]

        sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=features)

        pred_int  = model.predict(sample)[0]
        pred_crop = le.inverse_transform([pred_int])[0]

        # Confidence from predict_proba if available
        conf_str = ""
        if hasattr(model, "predict_proba"):
            proba    = model.predict_proba(sample)[0]
            conf     = proba[pred_int] * 100
            conf_str = f'<span class="result-conf">Confidence: {conf:.1f}%</span>'

        info = CROP_INFO.get(pred_crop, {})
        emoji = info.get("emoji", "🌱")

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem; margin-bottom:0.5rem">{emoji}</div>
            <div class="result-crop-name">{pred_crop}</div>
            <div class="result-model-tag">via {model_choice}</div>
            {conf_str}
            <div class="info-grid">
                <div class="info-card">
                    <div class="info-card-label">Season</div>
                    <div class="info-card-value">{info.get('season','—')}</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Water Need</div>
                    <div class="info-card-value">{info.get('water','—')}</div>
                </div>
                <div class="info-card" style="grid-column: 1/-1">
                    <div class="info-card-label">Ideal Soil</div>
                    <div class="info-card-value">{info.get('soil','—')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Input summary ──
        st.markdown('<div style="margin-top:1.6rem"><div class="section-label">Input Summary</div></div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            "Feature": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
            "Value":   [N, P, K, f"{temperature}°C", f"{humidity}%", ph, f"{rainfall} mm"]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True,
                     column_config={
                         "Feature": st.column_config.TextColumn(width="medium"),
                         "Value":   st.column_config.TextColumn(width="medium"),
                     })

    else:
        # ── Placeholder state ──
        st.markdown("""
        <div style="
            border: 1px dashed #1e3028;
            border-radius: 14px;
            padding: 3.5rem 2rem;
            text-align: center;
            color: #2e4e3c;
        ">
            <div style="font-size:2.8rem; margin-bottom:1rem">🌱</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:#3a5e4a">
                Awaiting soil & climate data
            </div>
            <div style="font-size:0.82rem; margin-top:0.5rem; color:#2a3e34">
                Adjust the sliders and hit Recommend
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Feature guide ──
        st.markdown('<div style="margin-top:1.6rem"><div class="section-label">Feature Guide</div></div>', unsafe_allow_html=True)
        guide = pd.DataFrame({
            "Feature":     ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
            "Range":       ["0–140", "5–145", "5–205", "8–44°C", "14–100%", "3.5–10", "20–300 mm"],
            "Description": [
                "Nitrogen content in soil",
                "Phosphorus content in soil",
                "Potassium content in soil",
                "Average ambient temperature",
                "Relative humidity of air",
                "Soil acidity / alkalinity",
                "Annual average rainfall",
            ]
        })
        st.dataframe(guide, hide_index=True, use_container_width=True)
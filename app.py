"""
AI-Based Traffic Management — Streamlit App (Part 3)
Loads the models trained in the notebook and predicts congestion level / traffic volume.

Run locally:   streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Traffic Management", page_icon="🚦", layout="wide")


# ---------------------------------------------------------------- load models
@st.cache_resource
def load_artifacts():
    prep = joblib.load("preprocessor.joblib")
    clf = {
        "Logistic Regression": joblib.load("logistic_regression.joblib"),
        "Decision Tree": joblib.load("decision_tree.joblib"),
        "Random Forest (tuned)": joblib.load("random_forest.joblib"),
        "k-Nearest Neighbors": joblib.load("knn.joblib"),
        "Naive Bayes": joblib.load("naive_bayes.joblib"),
    }
    reg = joblib.load("linear_regression.joblib")
    meta = joblib.load("meta.joblib")
    return prep, clf, reg, meta


prep, clf_models, reg_model, meta = load_artifacts()


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the notebook."""
    d = d.copy()
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["is_peak_hour"] = d["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)
    return d


ICON = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
ADVICE = {
    "Low": "Traffic is flowing freely. Normal signal timings are sufficient.",
    "Medium": "Moderate load. Consider extending green phases on main corridors and monitoring the junction.",
    "High": "Heavy congestion expected. Activate peak-hour signal plan, deploy traffic personnel and publish diversion advisories.",
}

# ---------------------------------------------------------------- header
st.title("🚦 AI-Based Traffic Management")
st.write(
    "Predict **congestion level** and **traffic volume** at a city junction from time, weather and incident "
    "information using the machine-learning models trained in the case study."
)

# ---------------------------------------------------------------- inputs
st.sidebar.header("Input conditions")
hour = st.sidebar.slider("Hour of day (0–23)", 0, 23, 9)
day_name = st.sidebar.selectbox(
    "Day of week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
month = st.sidebar.slider("Month", 1, 12, 6)
is_holiday = st.sidebar.checkbox("Public holiday")
weather = st.sidebar.selectbox(
    "Weather", ["Clear", "Clouds", "Fog", "Rain", "Heavy Rain"]
)
temp_c = st.sidebar.slider("Temperature (°C)", 5.0, 45.0, 28.0, 0.5)
rain_mm = st.sidebar.slider(
    "Rainfall (mm in last hour)",
    0.0,
    30.0,
    0.0 if weather in ("Clear", "Clouds", "Fog") else 3.0,
    0.5,
)
clouds_pct = st.sidebar.slider("Cloud cover (%)", 0, 100, 40)
accident = st.sidebar.checkbox("Accident reported nearby")
road_work = st.sidebar.checkbox("Road work in progress")

model_choice = st.sidebar.selectbox(
    "Classification model", list(clf_models.keys()), index=2
)
day_idx = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
].index(day_name)

row = pd.DataFrame(
    [
        {
            "hour": hour,
            "day_of_week": day_idx,
            "month": month,
            "is_weekend": int(day_idx >= 5),
            "is_holiday": int(is_holiday),
            "temp_c": temp_c,
            "rain_mm": rain_mm,
            "clouds_pct": clouds_pct,
            "weather": weather,
            "accident_reported": int(accident),
            "road_work": int(road_work),
        }
    ]
)
X = prep.transform(add_features(row)[meta["num_features"] + meta["cat_features"]])

# ---------------------------------------------------------------- predict
if st.sidebar.button("🔮 Predict", type="primary"):
    model = clf_models[model_choice]
    pred = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    volume = max(0, int(reg_model.predict(X)[0]))

    st.subheader(f"Prediction ({model_choice})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Congestion level", f"{ICON[pred]} {pred}")
    c2.metric("Confidence", f"{proba[pred] * 100:.1f}%")
    c3.metric("Estimated volume (Linear Regression)", f"{volume:,} veh/hr")
    {"Low": st.success, "Medium": st.warning, "High": st.error}[pred](ADVICE[pred])

    st.markdown("**Class probabilities**")
    st.bar_chart(pd.Series({k: proba[k] for k in ["Low", "Medium", "High"]}))

    st.markdown("**All models compared on this input**")
    table = []
    for name, m in clf_models.items():
        p = m.predict(X)[0]
        pr = dict(zip(m.classes_, m.predict_proba(X)[0]))
        table.append(
            {
                "Model": name,
                "Prediction": f"{ICON[p]} {p}",
                "Confidence": f"{pr[p] * 100:.1f}%",
            }
        )
    st.table(pd.DataFrame(table))
else:
    st.info("Set the conditions in the sidebar and click **Predict**.")

# ---------------------------------------------------------------- model performance
with st.expander("📊 Model performance on the test set"):
    perf = pd.DataFrame(meta["test_results"]).round(3)
    st.dataframe(perf)
    st.caption(
        f"Best Random Forest parameters (GridSearchCV): {meta['best_params']}  |  "
        f"Linear Regression: RMSE ≈ {meta['reg']['rmse']:.0f} veh/hr, R² = {meta['reg']['r2']:.2f}"
    )

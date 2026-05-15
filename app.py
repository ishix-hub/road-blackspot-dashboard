import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Black Spot Predictor | Delhi & Pune",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths — works both on Streamlit Cloud and Colab ───────────────────────
# On Streamlit Cloud: all files are in the same folder as app.py
# On Colab: set PROJECT_BASE env variable to Drive path
PROJECT_BASE = os.environ.get("PROJECT_BASE", "")

if PROJECT_BASE:
    # Running in Colab
    DATA_PATH  = os.path.join(PROJECT_BASE, "data", "processed", "master_geo_sample.csv")
    MODEL_PATH = os.path.join(PROJECT_BASE, "outputs", "models", "catboost_model_v2.pkl")
    FEAT_PATH  = os.path.join(PROJECT_BASE, "outputs", "models", "FEATURE_COLS.pkl")
    ARCH_PATH  = os.path.join(PROJECT_BASE, "data", "processed", "flagged_segments_archetypes_v2.csv")
else:
    # Running on Streamlit Cloud — all files in same directory as app.py
    BASE       = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH  = os.path.join(BASE, "master_geo_sample.csv")
    MODEL_PATH = os.path.join(BASE, "catboost_model_v2.pkl")
    FEAT_PATH  = os.path.join(BASE, "FEATURE_COLS.pkl")
    ARCH_PATH  = os.path.join(BASE, "flagged_segments_archetypes_v2.csv")

CITY_CENTERS = {
    "Delhi": [28.6139, 77.2090],
    "Pune":  [18.5204, 73.8567]
}

# ── Load functions (cached) ───────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH, low_memory=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(FEAT_PATH):
        st.error(f"FEATURE_COLS file not found: {FEAT_PATH}")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEAT_PATH, "rb") as f:
        feat_cols = pickle.load(f)
    return model, feat_cols

@st.cache_data
def load_archetypes():
    if os.path.exists(ARCH_PATH):
        return pd.read_csv(ARCH_PATH)
    return pd.DataFrame()

# ── Load everything ───────────────────────────────────────────────────────
with st.spinner("Loading data and model..."):
    df          = load_data()
    cat_model, FEATURE_COLS = load_model()
    arch_df     = load_archetypes()

CAT_FEATURES     = ["highway_type"]
ALL_FEATURES_CAT = FEATURE_COLS + CAT_FEATURES

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("🚦 Black Spot Predictor")
st.sidebar.markdown("**Delhi & Pune Road Safety**")
st.sidebar.markdown("---")

city_filter = st.sidebar.selectbox("Select City", ["Both", "Delhi", "Pune"])
only_bs     = st.sidebar.checkbox("Show only black spots", value=False)
panel       = st.sidebar.radio("Panel", [
    "🗺️ Risk Heatmap",
    "🔍 SHAP Segment Detail",
    "🧩 Risk Archetypes",
    "⚙️ What-If Simulator"
])

st.sidebar.markdown("---")
n_total = len(df)
n_bs    = int(df["is_blackspot"].sum()) if "is_blackspot" in df.columns else 0
st.sidebar.markdown(f"**Segments:** {n_total:,}")
st.sidebar.markdown(f"**Black spots:** {n_bs:,} ({n_bs/n_total:.1%})")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** CatBoost (AUC 1.000)")
st.sidebar.markdown("**Data:** MoRTH · OSM · Open-Meteo · Census 2011")

# ── Filter ────────────────────────────────────────────────────────────────
df_v = df.copy()
if city_filter != "Both":
    df_v = df_v[df_v["city"] == city_filter]
if only_bs and "is_blackspot" in df_v.columns:
    df_v = df_v[df_v["is_blackspot"] == 1]

# ════════════════════════════════════════════════════════════════
# PANEL 1 — RISK HEATMAP
# ════════════════════════════════════════════════════════════════
if panel == "🗺️ Risk Heatmap":
    st.title("🗺️ Road Risk Heatmap")
    st.markdown(
        "Segments coloured by CatBoost risk score. "
        "**Red markers** = Government Blind Spots (ML found, MoRTH missed). "
        "**Orange markers** = Confirmed black spots."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Segments", f"{len(df_v):,}")
    if "is_blackspot" in df_v.columns:
        c2.metric("Black Spots", f"{int(df_v['is_blackspot'].sum()):,}")
    if "cat_predicted" in df_v.columns:
        c3.metric("ML Flagged", f"{int(df_v['cat_predicted'].sum()):,}")
    if "mismatch_category" in df_v.columns:
        n_blind = int((df_v["mismatch_category"] == "Government Blind Spot").sum())
        c4.metric("Govt Blind Spots", f"{n_blind:,}")

    center = [23.5, 77.5] if city_filter == "Both" else CITY_CENTERS[city_filter]
    zoom   = 5 if city_filter == "Both" else 11

    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    if "cat_risk_score" in df_v.columns:
        hd = df_v[["lat", "lon", "cat_risk_score"]].dropna()
        if len(hd) > 50000:
            hd = hd.sample(50000, random_state=42)
        HeatMap(
            hd[["lat", "lon", "cat_risk_score"]].values.tolist(),
            min_opacity=0.3, radius=10, blur=12,
            gradient={0.0: "green", 0.4: "yellow",
                      0.7: "orange", 1.0: "red"}
        ).add_to(m)

    if "is_blackspot" in df_v.columns:
        bs = df_v[df_v["is_blackspot"] == 1][
            ["lat", "lon", "highway_type", "cat_risk_score", "mismatch_category"]
        ].dropna()
        for _, r in bs.sample(min(2000, len(bs)), random_state=42).iterrows():
            col = "red" if str(r.get("mismatch_category", "")) == "Government Blind Spot" else "orange"
            popup_txt = (str(r["highway_type"]) + " | Risk: " +
                         str(round(float(r["cat_risk_score"]), 3)))
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=4, color=col, fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup_txt, max_width=200)
            ).add_to(m)

    st_folium(m, width=1100, height=600)

# ════════════════════════════════════════════════════════════════
# PANEL 2 — SHAP SEGMENT DETAIL
# ════════════════════════════════════════════════════════════════
elif panel == "🔍 SHAP Segment Detail":
    st.title("🔍 Segment-Level Risk Explanation (SHAP)")
    st.markdown("Select a road segment to see why the model flagged it.")

    city_s  = st.selectbox("City", ["Delhi", "Pune"])
    seg_pool = df_v[df_v["city"] == city_s].copy()

    if "cat_risk_score" in seg_pool.columns:
        seg_pool = seg_pool.sort_values("cat_risk_score", ascending=False)

    top50 = seg_pool.head(50).reset_index(drop=True)

    def fmt(i):
        r = top50.loc[i]
        risk = round(float(r["cat_risk_score"]), 4) if "cat_risk_score" in r else "N/A"
        label = "Black Spot" if r.get("is_blackspot") else "Safe"
        return f"Rank {i+1} | {r['highway_type']} | Risk: {risk} | {label}"

    idx = st.selectbox("Select segment", top50.index, format_func=fmt)
    seg = top50.loc[idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{float(seg['cat_risk_score']):.4f}" if "cat_risk_score" in seg else "N/A")
    c2.metric("Road Type",  str(seg["highway_type"]))
    c3.metric("Length",     f"{float(seg['length_m']):.0f} m")
    c4.metric("Label",      "Black Spot" if seg.get("is_blackspot") else "Safe")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("SHAP Feature Contributions")
        try:
            avail = [c for c in ALL_FEATURES_CAT if c in seg.index]
            seg_in = seg[avail].to_frame().T
            for c in FEATURE_COLS:
                if c in seg_in.columns:
                    seg_in[c] = pd.to_numeric(seg_in[c], errors="coerce").fillna(0)
            exp = shap.TreeExplainer(cat_model)
            sv  = exp.shap_values(seg_in)
            pairs = sorted(
                zip(avail, sv[0]),
                key=lambda x: abs(x[1]), reverse=True
            )[:10]
            feats  = [p[0].replace("_", " ").title() for p in pairs]
            vals   = [p[1] for p in pairs]
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
            fig = go.Figure(go.Bar(
                x=vals, y=feats, orientation="h",
                marker_color=colors
            ))
            fig.update_layout(
                title="Top 10 SHAP Values (red = raises risk)",
                xaxis_title="SHAP Value",
                height=420
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP error: {e}")

    with col_r:
        st.subheader("Feature Values")
        avail_feats = [f for f in FEATURE_COLS if f in seg.index]
        feat_df = pd.DataFrame({
            "Feature": [f.replace("_", " ").title() for f in avail_feats],
            "Value":   [seg[f] for f in avail_feats]
        })
        st.dataframe(feat_df, use_container_width=True, height=420)

    if pd.notna(seg.get("lat")) and pd.notna(seg.get("lon")):
        st.subheader("Segment Location")
        m2 = folium.Map(
            location=[seg["lat"], seg["lon"]],
            zoom_start=15, tiles="CartoDB positron"
        )
        folium.CircleMarker(
            location=[seg["lat"], seg["lon"]],
            radius=10, color="red", fill=True,
            popup=str(seg["highway_type"])
        ).add_to(m2)
        st_folium(m2, width=1100, height=320)

# ════════════════════════════════════════════════════════════════
# PANEL 3 — BERTOPIC ARCHETYPES
# ════════════════════════════════════════════════════════════════
elif panel == "🧩 Risk Archetypes":
    st.title("🧩 BERTopic Risk Archetype Clusters")
    st.markdown(
        "Risk archetypes discovered by applying BERTopic to SHAP value patterns "
        "across all flagged segments. First application of this method in road safety research."
    )

    if arch_df.empty:
        st.warning("flagged_segments_archetypes_v2.csv not found. Run Notebook 05 first.")
    else:
        ac = arch_df["archetype_name"].value_counts().reset_index()
        ac.columns = ["Archetype", "Count"]
        fig_bar = px.bar(
            ac, x="Count", y="Archetype", orientation="h",
            color="Archetype",
            title="Risk Archetypes — Number of Black Spot Segments",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_bar, use_container_width=True)

        col_a, col_b = st.columns(2)
        for city, col in zip(["Delhi", "Pune"], [col_a, col_b]):
            ca = arch_df[arch_df["city"] == city]["archetype_name"].value_counts()
            with col:
                st.subheader(f"{city}")
                if len(ca) > 0:
                    fig2 = px.pie(
                        values=ca.values, names=ca.index,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info(f"No archetype data for {city}.")

        st.subheader("Archetype Summary Table")
        summary = arch_df.groupby(
            ["archetype_name", "city"]
        ).size().unstack(fill_value=0).reset_index()
        summary.columns.name = None
        st.dataframe(summary, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PANEL 4 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════
elif panel == "⚙️ What-If Simulator":
    st.title("⚙️ What-If Infrastructure Simulator")
    st.markdown(
        "Adjust road features and see how the predicted risk changes in real time. "
        "Simulates the effect of infrastructure interventions."
    )

    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.subheader("Road Parameters")
        ht  = st.selectbox("Road Type", [
            "residential", "tertiary", "secondary",
            "primary", "trunk", "motorway",
            "service", "unclassified"
        ], index=3)
        lm  = st.slider("Segment Length (m)", 10, 2000, 250)
        ln  = st.slider("Number of Lanes", 1, 8, 2)
        sl  = st.slider("Speed Limit (km/h)", 10, 120, 50)
        ow  = int(st.checkbox("One-way road", value=False))
        id_ = st.slider("Intersection Density", 0, 30, 5,
                        help="Number of junctions within 200m")

        st.markdown("**POI Density (within 500m)**")
        ps  = st.slider("Schools",        0, 20, 2)
        ph  = st.slider("Hospitals",      0, 20, 1)
        pm  = st.slider("Markets",        0, 20, 1)
        pb  = st.slider("Bus Stops",      0, 30, 3)
        pf  = st.slider("Fuel Stations",  0, 10, 0)

        st.markdown("**Weather & Demographics**")
        rf  = st.slider("Annual Rainfall (mm)", 0, 1500, 650)
        fd  = st.slider("Fog Days per Year",    0, 60,   28)
        at  = st.slider("Avg Max Temp (°C)",    15.0, 45.0, 25.0)
        pd_ = st.slider("Population Density /km²", 100, 30000, 11000)
        us  = st.slider("Urban Share (%)",      0, 100, 97)

    with col_out:
        st.subheader("Predicted Risk")

        row = {
            "length_m":             lm,
            "lanes":                ln,
            "speed_limit":          sl,
            "is_oneway":            ow,
            "intersection_density": id_,
            "poi_schools":          ps,
            "poi_hospitals":        ph,
            "poi_markets":          pm,
            "poi_bus_stops":        pb,
            "poi_fuel":             pf,
            "annual_rainfall_mm":   rf,
            "fog_days_per_year":    fd,
            "avg_temp_c":           at,
            "pop_density_km2":      pd_,
            "urban_share_pct":      us,
            "highway_type":         ht
        }

        inp = pd.DataFrame([row])
        for c in FEATURE_COLS:
            if c in inp.columns:
                inp[c] = pd.to_numeric(inp[c], errors="coerce").fillna(0)

        try:
            risk = float(cat_model.predict_proba(inp[ALL_FEATURES_CAT])[0][1])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            risk = 0.0

        gc = "#2ecc71" if risk < 0.3 else "#f39c12" if risk < 0.6 else "#e74c3c"
        vd = ("🟢 LOW RISK" if risk < 0.3
              else "🟡 MODERATE RISK" if risk < 0.6
              else "🔴 HIGH RISK — BLACK SPOT")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk * 100, 1),
            title={"text": "Black Spot Risk Score (%)"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": gc},
                "steps": [
                    {"range": [0,  30],  "color": "#d5f5e3"},
                    {"range": [30, 60],  "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        ))
        fig_g.update_layout(height=320)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(
            f"<h2 style='text-align:center;color:{gc}'>{vd}</h2>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("What is driving this risk?")
        try:
            exp   = shap.TreeExplainer(cat_model)
            sv    = exp.shap_values(inp[ALL_FEATURES_CAT])
            pairs = sorted(
                zip(ALL_FEATURES_CAT, sv[0]),
                key=lambda x: abs(x[1]), reverse=True
            )[:8]
            feats  = [p[0].replace("_", " ").title() for p in pairs]
            vals   = [p[1] for p in pairs]
            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
            fig_s  = go.Figure(go.Bar(
                x=vals, y=feats, orientation="h",
                marker_color=colors
            ))
            fig_s.update_layout(
                xaxis_title="SHAP Value",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP error: {e}")

        st.subheader("💡 Suggested Interventions")
        sug = []
        if sl  > 60:  sug.append("Reduce speed limit — high speed significantly raises risk")
        if id_ > 8:   sug.append("Install junction controls — intersection density is elevated")
        if ps  > 3:   sug.append("Add school zone signage and speed humps")
        if pb  > 5:   sug.append("Designate bus bays to reduce conflict with traffic")
        if fd  > 20:  sug.append("Install fog warning signs and rumble strips")
        if not sug:   sug.append("No critical interventions identified for current parameters")
        for s in sug:
            st.markdown(f"• {s}")

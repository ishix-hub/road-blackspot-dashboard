
# ════════════════════════════════════════════════════════════════
# app.py — Road Accident Black Spot Dashboard
# Run: streamlit run app.py
# ════════════════════════════════════════════════════════════════

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
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Black Spot Predictor | Delhi & Pune",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = os.environ.get(
    "PROJECT_BASE",
    "/content/drive/MyDrive/Drive/road_accident_project"
)
DATA_PATH   = f"{BASE}/data/processed/master_geo.csv"
MODEL_PATH  = f"{BASE}/outputs/models/catboost_model_v2.pkl"
SCALER_PATH = f"{BASE}/outputs/models/scaler_v2.pkl"
FEAT_PATH   = f"{BASE}/outputs/models/FEATURE_COLS.pkl"
ARCH_PATH   = f"{BASE}/data/processed/flagged_segments_archetypes_v2.csv"

# ── Load data (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["lat", "lon"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEAT_PATH, "rb") as f:
        feat_cols = pickle.load(f)
    return model, scaler, feat_cols

@st.cache_data
def load_archetypes():
    if os.path.exists(ARCH_PATH):
        return pd.read_csv(ARCH_PATH)
    return pd.DataFrame()

df          = load_data()
cat_model, scaler, FEATURE_COLS = load_model()
arch_df     = load_archetypes()

CAT_FEATURES     = ["highway_type"]
ALL_FEATURES_CAT = FEATURE_COLS + CAT_FEATURES

CITY_CENTERS = {
    "Delhi": [28.6139, 77.2090],
    "Pune":  [18.5204, 73.8567],
}

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/240px-Emblem_of_India.svg.png",
    width=60
)
st.sidebar.title("🚦 Black Spot Predictor")
st.sidebar.markdown("**Delhi & Pune Road Safety**")
st.sidebar.markdown("---")

city_filter = st.sidebar.selectbox(
    "Select City", ["Both", "Delhi", "Pune"]
)
show_only_blackspots = st.sidebar.checkbox(
    "Show only black spots", value=False
)
panel = st.sidebar.radio(
    "Panel",
    ["🗺️ Risk Heatmap",
     "🔍 Segment Detail (SHAP)",
     "🧩 Risk Archetypes",
     "⚙️ What-If Simulator"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset:** {len(df):,} segments

"
    f"**Black spots:** {df["is_blackspot"].sum():,} "
    f"({df["is_blackspot"].mean():.1%})"
)

# ── Filter data ───────────────────────────────────────────────────────────
df_view = df.copy()
if city_filter != "Both":
    df_view = df_view[df_view["city"] == city_filter]
if show_only_blackspots:
    df_view = df_view[df_view["is_blackspot"] == 1]

# ════════════════════════════════════════════════════════════════
# PANEL 1 — RISK HEATMAP
# ════════════════════════════════════════════════════════════════
if panel == "🗺️ Risk Heatmap":
    st.title("🗺️ Road Risk Heatmap")
    st.markdown(
        "Segments coloured by predicted CatBoost risk score. "
        "Red = high risk, green = low risk. "
        "Use sidebar to filter by city or show only confirmed black spots."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Segments", f"{len(df_view):,}")
    col2.metric("Black Spots", f"{df_view["is_blackspot"].sum():,}")
    col3.metric("ML Flagged", f"{df_view["cat_predicted"].sum():,}")
    col4.metric("Govt Missed",
                f"{(df_view["mismatch_category"] == "Government Blind Spot").sum():,}")

    # Map center
    if city_filter == "Both":
        center = [23.5, 77.5]
        zoom   = 5
    else:
        center = CITY_CENTERS[city_filter]
        zoom   = 11

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron"
    )

    # Heatmap layer — weight by cat_risk_score
    heat_data = df_view[["lat", "lon", "cat_risk_score"]].dropna()
    if len(heat_data) > 50000:
        heat_data = heat_data.sample(50000, random_state=42)

    HeatMap(
        heat_data[["lat", "lon", "cat_risk_score"]].values.tolist(),
        min_opacity=0.3,
        radius=10,
        blur=12,
        gradient={0.0: "green", 0.4: "yellow",
                  0.7: "orange", 1.0: "red"}
    ).add_to(m)

    # Black spot markers (sample max 2000 for performance)
    blackspots = df_view[
        df_view["is_blackspot"] == 1
    ][["lat", "lon", "highway_type",
       "cat_risk_score", "mismatch_category"]].dropna()

    sample_bs = blackspots.sample(
        min(2000, len(blackspots)), random_state=42
    )
    for _, row in sample_bs.iterrows():
        color = ("red" if row["mismatch_category"] == "Government Blind Spot"
                 else "orange")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row["highway_type"]}</b><br>"
                f"Risk: {row["cat_risk_score"]:.3f}<br>"
                f"{row["mismatch_category"]}",
                max_width=200
            )
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;">
        <b>Legend</b><br>
        <span style="color:red">●</span> Govt Blind Spot<br>
        <span style="color:orange">●</span> Known Black Spot<br>
        <span style="background:linear-gradient(to right,green,yellow,red);
               display:inline-block;width:80px;height:10px;"></span>
        Risk Score
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=1100, height=600)

# ════════════════════════════════════════════════════════════════
# PANEL 2 — SHAP PER-SEGMENT DETAIL
# ════════════════════════════════════════════════════════════════
elif panel == "🔍 Segment Detail (SHAP)":
    st.title("🔍 Segment-Level Risk Explanation (SHAP)")
    st.markdown(
        "Select a road segment to see why the model flagged it as a black spot. "
        "SHAP values show each feature's contribution to the risk score."
    )

    # Filter to city
    city_sel = st.selectbox("City", ["Delhi", "Pune"])
    road_options = (
        df_view[df_view["city"] == city_sel]["highway_type"]
        .value_counts().index.tolist()
    )
    road_sel = st.selectbox("Filter by road type", ["All"] + road_options)

    seg_df = df_view[df_view["city"] == city_sel].copy()
    if road_sel != "All":
        seg_df = seg_df[seg_df["highway_type"] == road_sel]

    # Show top 50 highest-risk segments
    top_segs = (
        seg_df.sort_values("cat_risk_score", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )

    seg_idx = st.selectbox(
        "Select segment (sorted by risk score)",
        top_segs.index,
        format_func=lambda i: (
            f"Rank {i+1} | {top_segs.loc[i, "highway_type"]}"
            f" | Risk: {top_segs.loc[i, "cat_risk_score"]:.4f}"
            f" | {'🔴 Black Spot' if top_segs.loc[i, "is_blackspot"] else '🟢 Safe'}"
        )
    )

    seg = top_segs.loc[seg_idx]

    # Segment info cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{seg["cat_risk_score"]:.4f}")
    c2.metric("Road Type", seg["highway_type"])
    c3.metric("Length", f"{seg["length_m"]:.0f} m")
    c4.metric(
        "Label",
        "🔴 Black Spot" if seg["is_blackspot"] else "🟢 Safe"
    )

    st.markdown("---")

    # SHAP waterfall
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("SHAP Feature Contributions")
        try:
            seg_input = seg[ALL_FEATURES_CAT].to_frame().T
            for col in FEATURE_COLS:
                seg_input[col] = pd.to_numeric(seg_input[col], errors="coerce").fillna(0)

            explainer   = shap.TreeExplainer(cat_model)
            shap_values = explainer.shap_values(seg_input)

            shap_pairs = sorted(
                zip(ALL_FEATURES_CAT, shap_values[0]),
                key=lambda x: abs(x[1]), reverse=True
            )[:10]

            features = [p[0].replace("_", " ").title() for p in shap_pairs]
            values   = [p[1] for p in shap_pairs]
            colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

            fig = go.Figure(go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker_color=colors
            ))
            fig.update_layout(
                title="Top 10 SHAP Values (red = increases risk)",
                xaxis_title="SHAP Value",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP computation error: {e}")

    with col_r:
        st.subheader("Segment Feature Values")
        feat_df = pd.DataFrame({
            "Feature": [f.replace("_", " ").title() for f in FEATURE_COLS],
            "Value":   [seg[f] for f in FEATURE_COLS]
        })
        st.dataframe(feat_df, use_container_width=True, height=420)

    # Mini map of selected segment
    if pd.notna(seg["lat"]) and pd.notna(seg["lon"]):
        st.subheader("Segment Location")
        m2 = folium.Map(
            location=[seg["lat"], seg["lon"]],
            zoom_start=15,
            tiles="CartoDB positron"
        )
        folium.CircleMarker(
            location=[seg["lat"], seg["lon"]],
            radius=10,
            color="red",
            fill=True,
            popup=f"{seg["highway_type"]} | Risk: {seg["cat_risk_score"]:.4f}"
        ).add_to(m2)
        st_folium(m2, width=1100, height=350)

# ════════════════════════════════════════════════════════════════
# PANEL 3 — BERTOPIC ARCHETYPE CLUSTERS
# ════════════════════════════════════════════════════════════════
elif panel == "🧩 Risk Archetypes":
    st.title("🧩 BERTopic Risk Archetype Clusters")
    st.markdown(
        "Risk archetypes are discovered by applying BERTopic to SHAP value "
        "patterns across all flagged segments. Each archetype represents a "
        "distinct risk profile — a combination of features that makes a road "
        "segment dangerous."
    )

    if arch_df.empty:
        st.warning(
            "flagged_segments_archetypes_v2.csv not found. "
            "Run Notebook 05 Cell 23 first."
        )
    else:
        # Archetype summary bar chart
        arch_counts = arch_df["archetype_name"].value_counts().reset_index()
        arch_counts.columns = ["Archetype", "Count"]

        fig_bar = px.bar(
            arch_counts,
            x="Count",
            y="Archetype",
            orientation="h",
            color="Archetype",
            title="Risk Archetypes — Number of Black Spot Segments",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Archetype by city
        col_a, col_b = st.columns(2)
        for city, col in zip(["Delhi", "Pune"], [col_a, col_b]):
            city_arch = arch_df[
                arch_df["city"] == city
            ]["archetype_name"].value_counts()
            with col:
                st.subheader(f"{city} Archetypes")
                if len(city_arch) > 0:
                    fig_pie = px.pie(
                        values=city_arch.values,
                        names=city_arch.index,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_pie.update_layout(height=320)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info(f"No archetype data for {city}.")

        # Archetype map
        st.subheader("Archetype Map")
        arch_geo = arch_df.dropna(subset=["lat", "lon"])             if "lat" in arch_df.columns             else arch_df.merge(
                df[["osmid_str", "lat", "lon"]],
                left_on=arch_df["osmid"].astype(str),
                right_on="osmid_str",
                how="left"
            ).dropna(subset=["lat", "lon"])

        arch_sample = arch_geo.sample(
            min(3000, len(arch_geo)), random_state=42
        )

        archetype_names = arch_sample["archetype_name"].unique()
        palette = px.colors.qualitative.Set2
        color_map = {
            name: palette[i % len(palette)]
            for i, name in enumerate(archetype_names)
        }

        city_arch_sel = st.selectbox(
            "City for archetype map", ["Both", "Delhi", "Pune"],
            key="arch_city"
        )
        if city_arch_sel != "Both":
            arch_sample = arch_sample[
                arch_sample["city"] == city_arch_sel
            ]
            center_arch = CITY_CENTERS[city_arch_sel]
            zoom_arch   = 11
        else:
            center_arch = [23.5, 77.5]
            zoom_arch   = 5

        m3 = folium.Map(
            location=center_arch,
            zoom_start=zoom_arch,
            tiles="CartoDB positron"
        )
        for _, row in arch_sample.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=color_map.get(row["archetype_name"], "grey"),
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row["archetype_name"]}</b><br>"
                    f"{row["highway_type"]}",
                    max_width=200
                )
            ).add_to(m3)
        st_folium(m3, width=1100, height=500)

# ════════════════════════════════════════════════════════════════
# PANEL 4 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════
elif panel == "⚙️ What-If Simulator":
    st.title("⚙️ What-If Infrastructure Simulator")
    st.markdown(
        "Adjust road features below and see how the predicted risk score "
        "changes. This simulates the effect of infrastructure interventions "
        "such as speed limit reduction, junction redesign, or land use changes."
    )

    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.subheader("Road Segment Parameters")

        highway_type   = st.selectbox(
            "Road Type",
            ["residential", "tertiary", "secondary",
             "primary", "trunk", "motorway",
             "service", "unclassified"],
            index=3
        )
        length_m       = st.slider("Segment Length (m)", 10, 2000, 250)
        lanes          = st.slider("Number of Lanes", 1, 8, 2)
        speed_limit    = st.slider("Speed Limit (km/h)", 10, 120, 50)
        is_oneway      = st.checkbox("One-way road", value=False)
        int_density    = st.slider("Intersection Density", 0, 30, 5,
                                   help="Junctions within 200m")

        st.markdown("**POI Density (count within 500m)**")
        poi_schools    = st.slider("Schools",   0, 20, 2)
        poi_hospitals  = st.slider("Hospitals", 0, 20, 1)
        poi_markets    = st.slider("Markets",   0, 20, 1)
        poi_bus_stops  = st.slider("Bus Stops", 0, 30, 3)
        poi_fuel       = st.slider("Fuel Stations", 0, 10, 0)

        st.markdown("**Weather & Demographics**")
        rainfall       = st.slider("Annual Rainfall (mm)", 0, 1500, 650)
        fog_days       = st.slider("Fog Days per Year", 0, 60, 28)
        avg_temp       = st.slider("Avg Max Temperature (°C)", 15.0, 45.0, 25.0)
        pop_density    = st.slider("Population Density (per km²)",
                                   100, 30000, 11000)
        urban_share    = st.slider("Urban Share (%)", 0, 100, 97)

    with col_out:
        st.subheader("Predicted Risk")

        # Build input row — same order as FEATURE_COLS
        input_dict = {
            "length_m":           length_m,
            "lanes":              lanes,
            "speed_limit":        speed_limit,
            "is_oneway":          int(is_oneway),
            "intersection_density": int_density,
            "poi_schools":        poi_schools,
            "poi_hospitals":      poi_hospitals,
            "poi_markets":        poi_markets,
            "poi_bus_stops":      poi_bus_stops,
            "poi_fuel":           poi_fuel,
            "annual_rainfall_mm": rainfall,
            "fog_days_per_year":  fog_days,
            "avg_temp_c":         avg_temp,
            "pop_density_km2":    pop_density,
            "urban_share_pct":    urban_share,
            "highway_type":       highway_type,
        }

        input_df = pd.DataFrame([input_dict])
        for col in FEATURE_COLS:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        risk_score = cat_model.predict_proba(
            input_df[ALL_FEATURES_CAT]
        )[0][1]

        # Risk gauge
        gauge_color = (
            "#2ecc71" if risk_score < 0.3
            else "#f39c12" if risk_score < 0.6
            else "#e74c3c"
        )
        verdict = (
            "🟢 LOW RISK" if risk_score < 0.3
            else "🟡 MODERATE RISK" if risk_score < 0.6
            else "🔴 HIGH RISK — BLACK SPOT"
        )

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(risk_score * 100, 1),
            title={"text": "Black Spot Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": gauge_color},
                "steps": [
                    {"range": [0, 30],  "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fef9e7"},
                    {"range": [60, 100],"color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        ))
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"<h2 style="text-align:center;color:{gauge_color}">"
            f"{verdict}</h2>",
            unsafe_allow_html=True
        )

        # SHAP for this input
        st.markdown("---")
        st.subheader("What is driving this risk?")
        try:
            explainer   = shap.TreeExplainer(cat_model)
            shap_vals   = explainer.shap_values(input_df[ALL_FEATURES_CAT])
            shap_pairs  = sorted(
                zip(ALL_FEATURES_CAT, shap_vals[0]),
                key=lambda x: abs(x[1]), reverse=True
            )[:8]

            feats  = [p[0].replace("_", " ").title() for p in shap_pairs]
            svals  = [p[1] for p in shap_pairs]
            cols_c = ["#e74c3c" if v > 0 else "#2ecc71" for v in svals]

            fig_shap = go.Figure(go.Bar(
                x=svals, y=feats, orientation="h",
                marker_color=cols_c
            ))
            fig_shap.update_layout(
                xaxis_title="SHAP Value",
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP error: {e}")

        # Intervention suggestions
        st.subheader("💡 Suggested Interventions")
        suggestions = []
        if speed_limit > 60:
            suggestions.append("Reduce speed limit — high speed is a major risk factor")
        if int_density > 8:
            suggestions.append("Install junction controls — intersection density is elevated")
        if poi_schools > 3:
            suggestions.append("Add school zone signage and speed humps")
        if poi_bus_stops > 5:
            suggestions.append("Designate bus bays to reduce conflict with moving traffic")
        if fog_days > 20:
            suggestions.append("Install fog warning signs and rumble strips")
        if not suggestions:
            suggestions.append("No critical interventions identified for current parameters")
        for s in suggestions:
            st.markdown(f"• {s}")

# ── Footer ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Study:** Predicting Road Accident Black Spots

"
    "**Cities:** Delhi + Pune

"
    "**Model:** CatBoost (AUC 1.000, F1 0.994)

"
    "**Data:** MoRTH · OSM · Open-Meteo · Census 2011"
)

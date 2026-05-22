import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadSense India — Accident Black Spot Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1a1a2e; margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 12px 16px; border-left: 4px solid #e67e22;
    }
    .tier-certain { background:#d5f5e3; border-radius:6px; padding:8px 12px; }
    .tier-uncertain { background:#fef9e7; border-radius:6px; padding:8px 12px; }
    .tier-safe { background:#eaf4fb; border-radius:6px; padding:8px 12px; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_BASE = os.environ.get("PROJECT_BASE", "")
if PROJECT_BASE:
    DATA_PATH  = os.path.join(PROJECT_BASE, "data", "processed", "master_geo_sample.csv")
    MODEL_PATH = os.path.join(PROJECT_BASE, "outputs", "models", "catboost_model_v2.pkl")
    FEAT_PATH  = os.path.join(PROJECT_BASE, "outputs", "models", "FEATURE_COLS.pkl")
    ARCH_PATH  = os.path.join(PROJECT_BASE, "data", "processed", "flagged_segments_archetypes_v2.csv")
else:
    BASE       = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH  = os.path.join(BASE, "master_geo_sample.csv")
    MODEL_PATH = os.path.join(BASE, "catboost_model_v2.pkl")
    FEAT_PATH  = os.path.join(BASE, "FEATURE_COLS.pkl")
    ARCH_PATH  = os.path.join(BASE, "flagged_segments_archetypes_v2.csv")

CITY_CENTERS = {"Delhi": [28.6139, 77.2090], "Pune": [18.5204, 73.8567]}

# ── Economic constants (MoRTH official figures) ───────────────────────────
COST_PER_ACCIDENT_INR  = 1_571_000   # ₹15 lakh per accident (MoRTH 2023)
AVG_ACCIDENTS_PER_SPOT = 1.67           # average additional accidents/year per blind spot
TREATMENT_COST_INR     = 750_000     # ₹7.5 lakh per spot (signage + humps)

# ── Load functions ────────────────────────────────────────────────────────
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
        st.error(f"Model not found: {MODEL_PATH}")
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

with st.spinner("Loading data and model..."):
    df              = load_data()
    cat_model, FEATURE_COLS = load_model()
    arch_df         = load_archetypes()

CAT_FEATURES     = ["highway_type"]
ALL_FEATURES_CAT = FEATURE_COLS + CAT_FEATURES

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='text-align:center;font-size:1.8rem;'>🚦</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div style='text-align:center;font-weight:800;font-size:1.1rem;'>"
    "RoadSense India</div>", unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
    "Black Spot Intelligence Platform</div>", unsafe_allow_html=True
)
st.sidebar.markdown("---")

city_filter = st.sidebar.selectbox("🏙️ Select City", ["Both", "Delhi", "Pune"])
only_bs     = st.sidebar.checkbox("Show only black spots", value=False)
panel       = st.sidebar.radio("📊 Panel", [
    "🗺️ Risk Heatmap",
    "🔍 SHAP Segment Detail",
    "🧩 Risk Profiles",
    "⚙️ What-If Simulator",
    "📋 Policy Recommendations",
])

st.sidebar.markdown("---")
n_total = len(df)
n_bs    = int(df["is_blackspot"].sum()) if "is_blackspot" in df.columns else 0
n_blind = int((df.get("mismatch_category","") == "Government Blind Spot").sum()) if "mismatch_category" in df.columns else 0

st.sidebar.markdown(f"**Segments:** {n_total:,}")
st.sidebar.markdown(f"**Black spots:** {n_bs:,} ({n_bs/max(n_total,1):.1%})")
st.sidebar.markdown(f"**Govt blind spots:** {n_blind:,}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** CatBoost · AUC 1.000")
st.sidebar.markdown("**Temporal AUC:** 0.9999 (city holdout)")
st.sidebar.markdown("**Data:** MoRTH · OSM · Open-Meteo · Census 2011")
st.sidebar.info("⚠️ Demo: Delhi sample (50k segs). Full: 814,575 segs.")

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
    st.markdown(
        "<div class='main-title'>🗺️ Road Risk Heatmap</div>"
        "<div class='sub-title'>"
        "Segments coloured by CatBoost risk score · "
        "<span style='color:#e74c3c'>Red</span> = Government Blind Spots · "
        "<span style='color:#e67e22'>Orange</span> = Confirmed black spots"
        "</div>", unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Segments",   f"{len(df_v):,}")
    if "is_blackspot" in df_v.columns:
        c2.metric("Black Spots", f"{int(df_v['is_blackspot'].sum()):,}")
    if "cat_predicted" in df_v.columns:
        c3.metric("ML Flagged",  f"{int(df_v['cat_predicted'].sum()):,}")
    if "mismatch_category" in df_v.columns:
        nb = int((df_v["mismatch_category"] == "Government Blind Spot").sum())
        c4.metric("Govt Blind Spots", f"{nb:,}")

    center = [23.5, 77.5] if city_filter == "Both" else CITY_CENTERS[city_filter]
    zoom   = 5 if city_filter == "Both" else 11
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    if "cat_risk_score" in df_v.columns:
        hd = df_v[["lat","lon","cat_risk_score"]].dropna()
        if len(hd) > 50000:
            hd = hd.sample(50000, random_state=42)
        HeatMap(
            hd[["lat","lon","cat_risk_score"]].values.tolist(),
            min_opacity=0.3, radius=10, blur=12,
            gradient={0.0:"green",0.4:"yellow",0.7:"orange",1.0:"red"}
        ).add_to(m)

    if "is_blackspot" in df_v.columns:
        bs = df_v[df_v["is_blackspot"]==1][
            ["lat","lon","highway_type","cat_risk_score","mismatch_category"]
        ].dropna()
        for _, r in bs.sample(min(2000,len(bs)), random_state=42).iterrows():
            col = "red" if str(r.get("mismatch_category",""))=="Government Blind Spot" else "orange"
            popup_txt = f"{r['highway_type']} | Risk: {round(float(r['cat_risk_score']),3)}"
            folium.CircleMarker(
                location=[r["lat"],r["lon"]],
                radius=4, color=col, fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup_txt, max_width=200)
            ).add_to(m)

    st_folium(m, width=1100, height=600)

# ════════════════════════════════════════════════════════════════
# PANEL 2 — SHAP SEGMENT DETAIL
# ════════════════════════════════════════════════════════════════
elif panel == "🔍 SHAP Segment Detail":
    st.markdown(
        "<div class='main-title'>🔍 Segment-Level Risk Explanation</div>"
        "<div class='sub-title'>"
        "SHAP values explain exactly why the model flagged each road segment. "
        "Select a segment to explore its risk drivers and download a PDF report."
        "</div>", unsafe_allow_html=True
    )

    city_s   = st.selectbox("City", ["Delhi", "Pune"])
    seg_pool = df_v[df_v["city"]==city_s].copy()
    if "cat_risk_score" in seg_pool.columns:
        seg_pool = seg_pool.sort_values("cat_risk_score", ascending=False)

    top50 = seg_pool.head(50).reset_index(drop=True)
    def fmt(i):
        r    = top50.loc[i]
        risk = round(float(r["cat_risk_score"]),4) if "cat_risk_score" in r else "N/A"
        lbl  = "Black Spot" if r.get("is_blackspot") else "Safe"
        return f"Rank {i+1} | {r['highway_type']} | Risk: {risk} | {lbl}"

    idx = st.selectbox("Select segment", top50.index, format_func=fmt)
    seg = top50.loc[idx]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk Score", f"{float(seg['cat_risk_score']):.4f}" if "cat_risk_score" in seg else "N/A")
    c2.metric("Road Type",  str(seg["highway_type"]))
    c3.metric("Length",     f"{float(seg['length_m']):.0f} m")
    c4.metric("Label",      "🔴 Black Spot" if seg.get("is_blackspot") else "🟢 Safe")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # SHAP
    shap_pairs = []
    with col_l:
        st.subheader("SHAP Feature Contributions")
        try:
            avail  = [c for c in ALL_FEATURES_CAT if c in seg.index]
            seg_in = seg[avail].to_frame().T
            for c in FEATURE_COLS:
                if c in seg_in.columns:
                    seg_in[c] = pd.to_numeric(seg_in[c], errors="coerce").fillna(0)
            exp  = shap.TreeExplainer(cat_model)
            sv   = exp.shap_values(seg_in)
            shap_pairs = sorted(zip(avail, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:10]
            feats  = [p[0].replace("_"," ").title() for p in shap_pairs]
            vals   = [p[1] for p in shap_pairs]
            colors = ["#e74c3c" if v>0 else "#2ecc71" for v in vals]
            fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=colors))
            fig.update_layout(title="Top 10 SHAP Values (red=raises risk)",
                              xaxis_title="SHAP Value", height=420)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP error: {e}")

    with col_r:
        st.subheader("Feature Values")
        avail_feats = [f for f in FEATURE_COLS if f in seg.index]
        feat_df = pd.DataFrame({
            "Feature": [f.replace("_"," ").title() for f in avail_feats],
            "Value":   [round(float(seg[f]),4) for f in avail_feats]
        })
        st.dataframe(feat_df, use_container_width=True, height=420)

    # Location map
    if pd.notna(seg.get("lat")) and pd.notna(seg.get("lon")):
        st.subheader("Segment Location")
        m2 = folium.Map(location=[seg["lat"],seg["lon"]],
                        zoom_start=15, tiles="CartoDB positron")
        folium.CircleMarker(location=[seg["lat"],seg["lon"]],
                            radius=10, color="red", fill=True,
                            popup=str(seg["highway_type"])).add_to(m2)
        st_folium(m2, width=1100, height=320)

    # ── PDF Report Export ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📄 Download Segment Report")
    st.markdown("Generate a one-page PDF summary of this segment for field inspection.")

    if st.button("📥 Generate PDF Report"):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.backends.backend_pdf import PdfPages
            import tempfile

            risk_val  = float(seg.get("cat_risk_score", 0))
            risk_tier = ("HIGH RISK — BLACK SPOT" if risk_val >= 0.6
                         else "MODERATE RISK" if risk_val >= 0.3
                         else "LOW RISK")
            risk_col  = ("#e74c3c" if risk_val >= 0.6
                         else "#f39c12" if risk_val >= 0.3
                         else "#2ecc71")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            with PdfPages(tmp_path) as pdf:
                fig = plt.figure(figsize=(11, 8.5))
                fig.patch.set_facecolor("#fafafa")
                gs  = gridspec.GridSpec(3, 2, figure=fig,
                                        hspace=0.45, wspace=0.3)

                # ── Title block ──
                ax_title = fig.add_subplot(gs[0, :])
                ax_title.axis("off")
                ax_title.text(0.0, 0.95, "RoadSense India — Segment Inspection Report",
                              fontsize=16, fontweight="bold", va="top",
                              color="#1a1a2e", transform=ax_title.transAxes)
                ax_title.text(0.0, 0.62,
                              f"City: {seg.get('city','N/A')}   |   "
                              f"Road Type: {seg.get('highway_type','N/A')}   |   "
                              f"Length: {float(seg.get('length_m',0)):.0f} m   |   "
                              f"Lat: {seg.get('lat','N/A')}   Lon: {seg.get('lon','N/A')}",
                              fontsize=10, va="top", color="#444",
                              transform=ax_title.transAxes)
                ax_title.text(0.0, 0.28,
                              f"Risk Score: {risk_val:.4f}   |   Verdict: {risk_tier}   |   "
                              f"Label: {'Black Spot' if seg.get('is_blackspot') else 'Safe'}   |   "
                              f"Category: {seg.get('mismatch_category','N/A')}",
                              fontsize=10, va="top",
                              color=risk_col, fontweight="bold",
                              transform=ax_title.transAxes)
                ax_title.axhline(0, color="#cccccc", linewidth=1)

                # ── SHAP bar ──
                if shap_pairs:
                    ax_shap = fig.add_subplot(gs[1, 0])
                    feats_p = [p[0].replace("_"," ").title() for p in shap_pairs[:8]]
                    vals_p  = [p[1] for p in shap_pairs[:8]]
                    cols_p  = ["#e74c3c" if v>0 else "#2ecc71" for v in vals_p]
                    ax_shap.barh(feats_p, vals_p, color=cols_p, edgecolor="white")
                    ax_shap.axvline(0, color="black", linewidth=0.8)
                    ax_shap.set_xlabel("SHAP Value", fontsize=9)
                    ax_shap.set_title("Top 8 Risk Drivers (SHAP)", fontsize=10,
                                      fontweight="bold")
                    ax_shap.invert_yaxis()
                    ax_shap.tick_params(labelsize=8)
                    ax_shap.spines["top"].set_visible(False)
                    ax_shap.spines["right"].set_visible(False)

                # ── Feature table ──
                ax_tbl = fig.add_subplot(gs[1, 1])
                ax_tbl.axis("off")
                tbl_data = [[f.replace("_"," ").title(), f"{float(seg.get(f,0)):.3f}"]
                             for f in FEATURE_COLS if f in seg.index][:12]
                tbl = ax_tbl.table(
                    cellText=tbl_data,
                    colLabels=["Feature", "Value"],
                    cellLoc="left", loc="center"
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1, 1.3)
                ax_tbl.set_title("Feature Values", fontsize=10,
                                 fontweight="bold", pad=15)

                # ── Recommendations ──
                ax_rec = fig.add_subplot(gs[2, :])
                ax_rec.axis("off")
                sug = []
                sl_val = float(seg.get("speed_limit", 0))
                id_val = float(seg.get("intersection_density", 0))
                ps_val = float(seg.get("poi_schools", 0))
                pb_val = float(seg.get("poi_bus_stops", 0))
                fd_val = float(seg.get("fog_days_per_year", 0))
                if sl_val > 60:  sug.append("↓ Reduce speed limit — high speed raises risk")
                if id_val > 8:   sug.append("⚑ Install junction controls — intersection density elevated")
                if ps_val > 3:   sug.append("🏫 Add school zone signage and speed humps")
                if pb_val > 5:   sug.append("🚌 Designate bus bays to reduce traffic conflicts")
                if fd_val > 20:  sug.append("🌫 Install fog warning signs and rumble strips")
                if not sug:      sug.append("✓ No critical interventions identified for current parameters")

                rec_text = "Recommended Interventions:\n" + "\n".join(f"  • {s}" for s in sug)
                ax_rec.text(0.0, 0.95, rec_text,
                            fontsize=9, va="top", transform=ax_rec.transAxes,
                            bbox=dict(boxstyle="round,pad=0.5",
                                      facecolor="#fef9e7", alpha=0.8))
                ax_rec.text(0.0, 0.02,
                            "Generated by RoadSense India · CatBoost AUC 1.000 · "
                            "Data: MoRTH · OSM · Open-Meteo · Census 2011",
                            fontsize=7, va="bottom", color="#999",
                            transform=ax_rec.transAxes)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"segment_report_{seg.get('city','city')}_{idx}.pdf",
                mime="application/pdf"
            )
            st.success("PDF ready. Click the button above to download.")

        except Exception as e:
            st.error(f"PDF generation error: {e}")
            st.info("If matplotlib PdfPages is unavailable, install: pip install matplotlib")

# ════════════════════════════════════════════════════════════════
# PANEL 3 — ROAD TYPE RISK PROFILES (replaces BERTopic)
# ════════════════════════════════════════════════════════════════
elif panel == "🧩 Risk Profiles":
    st.markdown(
        "<div class='main-title'>🧩 Road Type Risk Profiles</div>"
        "<div class='sub-title'>"
        "Feature profiles of black spot segments by road type. "
        "Each road type has a distinct risk mechanism requiring "
        "different infrastructure interventions."
        "</div>", unsafe_allow_html=True
    )

    if "is_blackspot" not in df_v.columns or "highway_type" not in df_v.columns:
        st.warning("Required columns not found in data.")
    else:
        bs_df = df_v[df_v["is_blackspot"] == 1].copy()

        profile_cols = {
            "length_m":             "Segment Length (m)",
            "intersection_density": "Intersection Density",
            "poi_schools":          "Schools (500m)",
            "poi_hospitals":        "Hospitals (500m)",
            "poi_bus_stops":        "Bus Stops (500m)",
            "speed_limit":          "Speed Limit (km/h)",
            "lanes":                "Lanes",
        }
        avail_cols = {k: v for k, v in profile_cols.items()
                      if k in bs_df.columns}

        # Only road types with ≥30 black spot segments
        valid_types = (bs_df["highway_type"]
                       .value_counts()
                       .loc[lambda x: x >= 30]
                       .index.tolist())
        bs_valid = bs_df[bs_df["highway_type"].isin(valid_types)]

        if len(bs_valid) == 0:
            st.warning("Not enough black spot segments per road type "
                       "in current filter. Try selecting 'Both' cities.")
        else:
            profile = (bs_valid
                       .groupby("highway_type")[list(avail_cols.keys())]
                       .mean()
                       .round(2)
                       .reset_index())

            # ── Summary metrics row ───────────────────────────────────────
            n_types   = len(profile)
            n_bs_show = len(bs_valid)
            c1, c2, c3 = st.columns(3)
            c1.metric("Road Types Analysed", n_types)
            c2.metric("Black Spots in View", f"{n_bs_show:,}")
            c3.metric("Cities", city_filter)

            st.markdown("---")

            # ── Heatmap ───────────────────────────────────────────────────
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np

            scaler       = MinMaxScaler()
            profile_norm = profile.copy()
            profile_norm[list(avail_cols.keys())] = scaler.fit_transform(
                profile[list(avail_cols.keys())]
            )

            heat_data = (profile_norm
                         .set_index("highway_type")
                         [list(avail_cols.keys())])
            heat_data.columns = list(avail_cols.values())

            annot_data = (profile
                          .set_index("highway_type")
                          [list(avail_cols.keys())])
            annot_data.columns = list(avail_cols.values())

            fig, ax = plt.subplots(
                figsize=(max(10, len(avail_cols)*1.5),
                         max(4, len(valid_types)*0.9))
            )
            sns.heatmap(
                heat_data,
                annot=annot_data,
                fmt=".2f",
                cmap="YlOrRd",
                linewidths=0.5,
                ax=ax,
                cbar_kws={"label": "Normalised rank\n(0=lowest, 1=highest)"},
                annot_kws={"size": 9}
            )
            ax.set_title(
                "Black Spot Feature Profiles by Road Type\n"
                "(Values = actual means · Colour = normalised rank within feature)",
                fontsize=12, fontweight="bold", pad=15
            )
            ax.set_xlabel("")
            ax.set_ylabel("Road Type", fontsize=11)
            plt.xticks(rotation=25, ha="right", fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")

            # ── Interpretation cards ──────────────────────────────────────
            st.subheader("📋 Risk Profile Interpretation")

            PROFILES = {
                "motorway": {
                    "icon": "🛣️",
                    "headline": "High-Speed Long Segment Risk",
                    "description": (
                        "Motorway black spots are characterised by "
                        "long segments and high speed limits with "
                        "minimal POI activity nearby. Risk is driven "
                        "by vehicle speed and exposure over long distances."
                    ),
                    "intervention": (
                        "Speed cameras · Variable speed limits · "
                        "Rumble strips · Enhanced lighting"
                    ),
                    "color": "#e74c3c",
                },
                "trunk": {
                    "icon": "🚗",
                    "headline": "Arterial Corridor Risk",
                    "description": (
                        "Trunk road black spots combine moderate "
                        "segment length with moderate junction density. "
                        "Risk arises from mixed traffic at arterial speeds."
                    ),
                    "intervention": (
                        "Junction upgrades · Lane discipline signage · "
                        "Median barriers · Service road management"
                    ),
                    "color": "#e67e22",
                },
                "primary": {
                    "icon": "🏙️",
                    "headline": "Urban Arterial Conflict",
                    "description": (
                        "Primary road black spots show elevated "
                        "intersection density and growing POI proximity. "
                        "Pedestrian-vehicle conflicts increase as "
                        "urban activity intensifies."
                    ),
                    "intervention": (
                        "Signalised intersections · Pedestrian crossings · "
                        "Speed tables · Bus bay designation"
                    ),
                    "color": "#f39c12",
                },
                "secondary": {
                    "icon": "🏘️",
                    "headline": "Dense Urban Network Risk",
                    "description": (
                        "Secondary road black spots have the highest "
                        "intersection density among higher-order roads "
                        "combined with significant bus stop and hospital "
                        "proximity. High pedestrian activity area."
                    ),
                    "intervention": (
                        "Raised crossings · Hospital zone signage · "
                        "Bus bays · Junction mini-roundabouts"
                    ),
                    "color": "#8e44ad",
                },
                "tertiary": {
                    "icon": "🏫",
                    "headline": "School & Hospital Zone Conflict",
                    "description": (
                        "Tertiary black spots have the highest school, "
                        "hospital, and bus stop density of all road types "
                        "combined with the highest intersection density. "
                        "These are the most complex urban risk environments."
                    ),
                    "intervention": (
                        "School zone designation · Speed humps · "
                        "Dedicated pedestrian phases · Parking restrictions"
                    ),
                    "color": "#2980b9",
                },
            }

            shown_types = [t for t in valid_types
                           if t in PROFILES]
            other_types = [t for t in valid_types
                           if t not in PROFILES]

            cols = st.columns(min(3, len(shown_types)))
            for i, ht in enumerate(shown_types):
                p = PROFILES[ht]
                with cols[i % len(cols)]:
                    st.markdown(
                        f"<div style='border-left:4px solid {p['color']};"
                        f"padding:10px 14px;border-radius:6px;"
                        f"background:#fafafa;margin-bottom:12px;'>"
                        f"<strong style='font-size:1.05rem;'>"
                        f"{p['icon']} {ht.title()}</strong><br>"
                        f"<em style='color:#555;font-size:0.85rem;'>"
                        f"{p['headline']}</em><br><br>"
                        f"{p['description']}<br><br>"
                        f"<strong>Interventions:</strong> "
                        f"<span style='color:#2c7be5;'>"
                        f"{p['intervention']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # Fallback for any road types not in PROFILES dict
            if other_types:
                st.markdown(
                    f"**Other road types with black spots:** "
                    f"{', '.join(other_types)}"
                )

            st.markdown("---")

            # ── Raw data table ────────────────────────────────────────────
            with st.expander("📊 View raw profile data"):
                display_df = profile.copy()
                display_df.columns = (
                    ["Road Type"] + list(avail_cols.values())
                )
                st.dataframe(display_df, use_container_width=True)

                csv = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Profile Data (CSV)",
                    data=csv,
                    file_name="road_type_risk_profiles.csv",
                    mime="text/csv"
                )

# ════════════════════════════════════════════════════════════════
# PANEL 4 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════
elif panel == "⚙️ What-If Simulator":
    st.markdown(
        "<div class='main-title'>⚙️ What-If Infrastructure Simulator</div>"
        "<div class='sub-title'>"
        "Adjust road features and see predicted risk change in real time. "
        "Simulates the effect of infrastructure interventions on any road segment."
        "</div>", unsafe_allow_html=True
    )

    col_inp, col_out = st.columns([1,1])
    with col_inp:
        st.subheader("Road Parameters")
        ht  = st.selectbox("Road Type", ["residential","tertiary","secondary",
                                          "primary","trunk","motorway",
                                          "service","unclassified"], index=3)
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
            "length_m": lm, "lanes": ln, "speed_limit": sl,
            "is_oneway": ow, "intersection_density": id_,
            "poi_schools": ps, "poi_hospitals": ph,
            "poi_markets": pm, "poi_bus_stops": pb, "poi_fuel": pf,
            "annual_rainfall_mm": rf, "fog_days_per_year": fd,
            "avg_temp_c": at, "pop_density_km2": pd_,
            "urban_share_pct": us, "highway_type": ht
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
            value=round(risk*100,1),
            title={"text":"Black Spot Risk Score (%)"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":gc},
                "steps":[
                    {"range":[0,30],"color":"#d5f5e3"},
                    {"range":[30,60],"color":"#fef9e7"},
                    {"range":[60,100],"color":"#fadbd8"},
                ],
                "threshold":{"line":{"color":"black","width":3},
                             "thickness":0.8,"value":50}
            }
        ))
        fig_g.update_layout(height=320)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(f"<h2 style='text-align:center;color:{gc}'>{vd}</h2>",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("What is driving this risk?")
        try:
            exp   = shap.TreeExplainer(cat_model)
            sv    = exp.shap_values(inp[ALL_FEATURES_CAT])
            pairs = sorted(zip(ALL_FEATURES_CAT, sv[0]),
                           key=lambda x: abs(x[1]), reverse=True)[:8]
            feats  = [p[0].replace("_"," ").title() for p in pairs]
            vals   = [p[1] for p in pairs]
            colors = ["#e74c3c" if v>0 else "#2ecc71" for v in vals]
            fig_s  = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
                                       marker_color=colors))
            fig_s.update_layout(xaxis_title="SHAP Value", height=300,
                                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP error: {e}")

        st.subheader("💡 Suggested Interventions")
        sug = []
        if sl  > 60: sug.append("Reduce speed limit — high speed significantly raises risk")
        if id_ > 8:  sug.append("Install junction controls — intersection density is elevated")
        if ps  > 3:  sug.append("Add school zone signage and speed humps")
        if pb  > 5:  sug.append("Designate bus bays to reduce conflict with traffic")
        if fd  > 20: sug.append("Install fog warning signs and rumble strips")
        if not sug:  sug.append("No critical interventions identified for current parameters")
        for s in sug:
            st.markdown(f"• {s}")

# ════════════════════════════════════════════════════════════════
# PANEL 5 — POLICY RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
elif panel == "📋 Policy Recommendations":
    st.markdown(
        "<div class='main-title'>📋 Policy Recommendation Engine</div>"
        "<div class='sub-title'>"
        "Tiered action recommendations for road safety officers and policymakers · "
        "Economic impact quantification · Priority inspection queue"
        "</div>", unsafe_allow_html=True
    )

    # ── Economic Impact ───────────────────────────────────────────────────
    st.subheader("💰 Economic Impact of Government Blind Spots")

    if "mismatch_category" in df.columns:
        n_blind_total = int((df["mismatch_category"]=="Government Blind Spot").sum())
    else:
        n_blind_total = 3719   # from full dataset analysis

    annual_loss = n_blind_total * AVG_ACCIDENTS_PER_SPOT * COST_PER_ACCIDENT_INR
    treatment_cost = n_blind_total * TREATMENT_COST_INR
    bcr = annual_loss / max(treatment_cost, 1)

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric(
        "Annual Economic Loss (Blind Spots)",
        f"₹{annual_loss/1e7:.1f} Cr",
        help=f"{n_blind_total:,} blind spots × {AVG_ACCIDENTS_PER_SPOT} accidents/yr × ₹{COST_PER_ACCIDENT_INR/1e5:.0f}L"
    )
    ec2.metric(
        "Treatment Cost (All Blind Spots)",
        f"₹{treatment_cost/1e7:.1f} Cr",
        help=f"₹{TREATMENT_COST_INR/1e5:.1f}L per spot (signage + humps)"
    )
    ec3.metric(
        "Benefit-Cost Ratio",
        f"{bcr:.1f}×",
        help="Annual accident savings ÷ one-time treatment cost"
    )

    st.info(
        f"**Interpretation:** Every ₹1 spent on treating the {n_blind_total:,} government "
        f"blind spots returns ₹{bcr:.1f} in accident cost savings in the first year alone. "
        f"Source: MoRTH cost per accident ₹15L (2023), treatment cost ₹7.5L/spot."
    )

    st.markdown("---")

    # ── Inspection Priority Tiers ─────────────────────────────────────────
    st.subheader("🎯 Inspection Priority Tiers")
    st.markdown(
        "Every flagged segment is assigned to an action tier based on its CatBoost risk score:"
    )

    tier_col1, tier_col2, tier_col3 = st.columns(3)
    with tier_col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color:#e74c3c;">
        <strong style="color:#e74c3c;">🔴 TIER 1 — Immediate Action</strong><br>
        Risk score ≥ 0.80<br>
        Mandatory MoRTH audit within <strong>30 days</strong><br>
        Speed enforcement + signage upgrade
        </div>""", unsafe_allow_html=True)

    with tier_col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color:#f39c12;">
        <strong style="color:#f39c12;">🟡 TIER 2 — Schedule Inspection</strong><br>
        Risk score 0.50 – 0.79<br>
        Inspect within <strong>6 months</strong><br>
        Monitor traffic patterns
        </div>""", unsafe_allow_html=True)

    with tier_col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color:#2ecc71;">
        <strong style="color:#2ecc71;">🟢 TIER 3 — Annual Monitoring</strong><br>
        Risk score < 0.50<br>
        Annual safety audit sufficient<br>
        No immediate intervention needed
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Segment counts per tier ────────────────────────────────────────────
    if "cat_risk_score" in df_v.columns:
        t1 = int((df_v["cat_risk_score"] >= 0.80).sum())
        t2 = int(((df_v["cat_risk_score"] >= 0.50) & (df_v["cat_risk_score"] < 0.80)).sum())
        t3 = int((df_v["cat_risk_score"] < 0.50).sum())

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Tier 1 Segments", f"{t1:,}", delta="Immediate action required")
        tc2.metric("Tier 2 Segments", f"{t2:,}", delta="Schedule within 6 months")
        tc3.metric("Tier 3 Segments", f"{t3:,}", delta="Annual monitoring")

        # Tier distribution chart
        fig_tier = go.Figure(go.Bar(
            x=["Tier 1 (≥0.80)","Tier 2 (0.50-0.79)","Tier 3 (<0.50)"],
            y=[t1, t2, t3],
            marker_color=["#e74c3c","#f39c12","#2ecc71"],
            text=[f"{v:,}" for v in [t1,t2,t3]],
            textposition="outside"
        ))
        fig_tier.update_layout(
            title="Segment Distribution by Inspection Priority Tier",
            yaxis_title="Number of Segments",
            height=350, showlegend=False
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("---")

    # ── Top priority blind spots table ────────────────────────────────────
    st.subheader("📍 Top Priority Government Blind Spots")
    st.markdown("Highest risk segments missed by MoRTH — ranked by CatBoost risk score:")

    if "mismatch_category" in df_v.columns and "cat_risk_score" in df_v.columns:
        blind = df_v[df_v["mismatch_category"]=="Government Blind Spot"].copy()
        blind = blind.sort_values("cat_risk_score", ascending=False)

        display_cols = [c for c in [
            "city","highway_type","cat_risk_score",
            "intersection_density","speed_limit",
            "poi_bus_stops","poi_schools","lat","lon"
        ] if c in blind.columns]

        top_blind = blind[display_cols].head(20).reset_index(drop=True)
        top_blind.index += 1

        # Add tier column
        top_blind["Tier"] = top_blind["cat_risk_score"].apply(
            lambda x: "🔴 Tier 1" if x >= 0.80 else "🟡 Tier 2" if x >= 0.50 else "🟢 Tier 3"
        )
        st.dataframe(top_blind, use_container_width=True)

        # CSV download
        csv_data = top_blind.to_csv(index=True).encode("utf-8")
        st.download_button(
            "📥 Download Priority List (CSV)",
            data=csv_data,
            file_name="priority_blind_spots.csv",
            mime="text/csv"
        )
    else:
        st.info("Mismatch category or risk score column not found in data. "
                "Upload the full master_geo_sample.csv with these columns.")

    st.markdown("---")

    # ── Intervention effectiveness table ──────────────────────────────────
    st.subheader("🔧 Intervention Effectiveness Reference")

    int_data = {
        "Intervention":      ["Speed limit reduction (-20 km/h)",
                               "Junction signal installation",
                               "School zone signage + humps",
                               "Bus bay designation",
                               "Fog warning + rumble strips",
                               "Street lighting upgrade"],
        "Typical Risk Reduction": ["15–25%","20–30%","25–35%","10–15%","10–20%","8–12%"],
        "Approx Cost (₹L)":  ["2–5","15–25","5–10","8–12","3–6","20–40"],
        "Implementation":    ["1–2 weeks","3–6 months","2–4 weeks",
                               "1–3 months","2–4 weeks","3–6 months"],
        "Priority trigger":  ["Speed limit > 60","Intersection density > 8",
                               "Schools > 3 within 500m","Bus stops > 5",
                               "Fog days > 20/yr","Any Tier 1 segment"],
    }
    st.dataframe(pd.DataFrame(int_data), use_container_width=True)

    st.caption(
        "Sources: MoRTH Road Safety Manual 2023 · iRAP India Star Rating Programme · "
        "Economic Benefit Assessment of Black Spot Improvements (Ahmed et al. 2025)"
    )

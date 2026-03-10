import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring"))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False

from scoring_engine import score_homeowner, RiskScoreResult

ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "homeowners_risk_dataset.csv"
MODEL_PKL = ROOT / "trained_models.pkl"

st.set_page_config(page_title="Underwriting Intelligence Platform", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1a1f2e;}
.main>div{padding-top:0!important;}
.block-container{padding:2rem 2.5rem!important;max-width:1400px;}
[data-testid="stSidebar"]{background:#f8f9fc;border-right:1px solid #e8ecf3;}
.page-hero{background:linear-gradient(135deg,#1a1f2e 0%,#2d3561 60%,#1a1f2e 100%);border-radius:16px;padding:32px 36px;margin-bottom:24px;position:relative;overflow:hidden;}
.page-hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,rgba(99,179,237,.15) 0%,transparent 70%);border-radius:50%;}
.hero-tag{font-size:11px;text-transform:uppercase;letter-spacing:3px;color:#90cdf4;font-weight:600;margin-bottom:8px;}
.hero-title{font-size:22px;font-weight:700;color:#fff;line-height:1.3;margin-bottom:8px;}
.hero-sub{font-size:13px;color:#a0aec0;max-width:600px;line-height:1.6;}
.section-title{font-size:18px;font-weight:700;color:#1a1f2e;margin-bottom:4px;line-height:1.3;}
.section-divider{height:2px;background:linear-gradient(90deg,#3b82f6,#8b5cf6,transparent);border:none;margin:8px 0 22px 0;border-radius:2px;}
.kpi-card{background:#fff;border-radius:10px;padding:14px 16px;border:1px solid #e8ecf3;box-shadow:0 1px 6px rgba(0,0,0,.05);position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.kpi-blue::before{background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.kpi-green::before{background:linear-gradient(90deg,#10b981,#34d399);}
.kpi-amber::before{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.kpi-red::before{background:linear-gradient(90deg,#ef4444,#f87171);}
.kpi-purple::before{background:linear-gradient(90deg,#8b5cf6,#a78bfa);}
.kpi-slate::before{background:linear-gradient(90deg,#475569,#64748b);}
.kpi-label{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#718096;font-weight:600;margin-bottom:4px;}
.kpi-value{font-size:20px;font-weight:700;color:#1a1f2e;line-height:1.2;font-variant-numeric:tabular-nums;}
.kpi-sub{font-size:11px;color:#a0aec0;margin-top:3px;}
.flag-card{padding:10px 14px;border-radius:8px;margin-bottom:6px;border-left:3px solid;font-size:12px;line-height:1.5;}
.flag-HIGH{background:#fff5f5;border-color:#fc8181;color:#742a2a;}
.flag-WARN{background:#fffbeb;border-color:#f6ad55;color:#744210;}
.flag-GOOD{background:#f0fff4;border-color:#68d391;color:#22543d;}
.flag-TIER3{background:#faf5ff;border-color:#b794f4;color:#44337a;}
.insight-box{background:#f7faff;border:1px solid #bee3f8;border-radius:10px;padding:14px 16px;margin-bottom:10px;}
.insight-title{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#3182ce;font-weight:700;margin-bottom:4px;}
.insight-text{font-size:12px;color:#2d3748;line-height:1.6;}
.stTabs [data-baseweb="tab"]{font-size:12px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.8px!important;}
[data-testid="stFormSubmitButton"] button{background:linear-gradient(135deg,#1a1f2e,#2d3561)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;letter-spacing:.5px!important;font-size:13px!important;padding:12px!important;width:100%!important;}
.stSlider label,.stSelectbox label,.stNumberInput label{font-size:11px!important;font-weight:600!important;color:#4a5568!important;text-transform:uppercase!important;letter-spacing:.4px!important;}
hr{border-color:#e8ecf3!important;margin:16px 0!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DEC_PALETTE = {
    "Preferred":{"bg":"#f0fff4","border":"#68d391","text":"#22543d","accent":"#38a169"},
    "Standard": {"bg":"#ebf8ff","border":"#63b3ed","text":"#2a4365","accent":"#3182ce"},
    "Rated":    {"bg":"#fffbeb","border":"#f6ad55","text":"#744210","accent":"#d69e2e"},
    "Decline":  {"bg":"#fff5f5","border":"#fc8181","text":"#742a2a","accent":"#e53e3e"},
}
TIER1_MAX  = {"Roof Age":18,"Roof Vulnerability":17,"Dwelling Construction":15,
              "Water Loss Recency":13,"Prior Claims (5yr)":11,"Coverage A Amount":8,
              "Fire Station Distance":7,"Home Age":5,"Square Footage":4,"ISO Class":2}
TIER1_R    = {"Roof Age":.42,"Roof Vulnerability":.38,"Dwelling Construction":.35,
              "Water Loss Recency":.29,"Prior Claims (5yr)":.26,"Coverage A Amount":.18,
              "Fire Station Distance":.15,"Home Age":.12,"Square Footage":.08,"ISO Class":.06}
TIER2_MAX  = {"Insurance Lapses":9,"Pet Ownership":8,"Swimming Pool":6,"Trampoline":5,
              "Wood-Burning Stove":4,"Home Business":4,"Recent Renovations":-6,
              "Monitored Alarm":-5,"Fire Sprinklers":-7,"Gated Community":-2}

T3_INTERACTIONS = [
    ("Steep Slope x Burn History",         "slope_burn_triggered",        1.08, 0.5194, "CONFIRMED"),
    ("Flood Zone x Stone/Dirt Foundation", "flood_foundation_triggered",  1.07, 0.4511, "CONFIRMED"),
    ("Aged Roof x Wildfire Zone",          "roof_age_wildfire_triggered", 1.06, 0.4254, "CONFIRMED"),
    ("Trampoline x Hail Zone",             "trampoline_hail_triggered",   1.03, 0.1754, "PARTIAL"),
    ("Prior Claims x Wildfire Zone",       "claims_wildfire_triggered",   1.03, 0.1652, "PARTIAL"),
    ("Remote Fire Station x Steep Slope",  "firestation_slope_triggered", 1.03, 0.1393, "PARTIAL"),
]

# ── Session state defaults ─────────────────────────────────────────────────────
# _snap holds the confirmed copy of inputs used for the last score.
# Widget keys hold the current displayed values.
# On submit: we build snap from widget locals → write to session_state → score from snap.
# This guarantees score always matches what was visible in the form.
_DEFAULTS = dict(
    roof_age=15, water_loss_recency=0, home_age=25,
    roof_vulnerability="asphalt", prior_claims_5yr=0, square_footage=2000,
    dwelling_construction="wood_frame", coverage_a_amount=350000, iso_class=3,
    fire_station_distance=3.0,
    insurance_lapses=0, swimming_pool="none", recent_renovations=0,
    pet_ownership="none", trampoline=0, monitored_alarm=0,
    wood_burning_stove="none", home_business="none", fire_sprinklers="none",
    gated_community=0,
    wildfire_zone=0, flood_zone=0, canopy_density="sparse",
    foundation_type="concrete_slab", slope_pct=5.0, hail_zone=0, burn_history=0,
    _last_result=None,
    _last_premium=None,
    _snap=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data
def load_portfolio():
    if DATA_PATH.exists(): return pd.read_csv(DATA_PATH)
    return None

@st.cache_resource
def load_artifacts(_mtime=None):
    if MODEL_PKL.exists():
        with open(MODEL_PKL, "rb") as f:
            return pickle.load(f)
    return None

def get_artifacts():
    mtime = MODEL_PKL.stat().st_mtime if MODEL_PKL.exists() else 0
    return load_artifacts(_mtime=mtime)

def compute_premium(expected_loss, decision):
    """Returns None for Decline — no standard-market premium exists."""
    loads = {"Preferred":1.25,"Standard":1.55,"Rated":2.00}
    load  = loads.get(decision)
    return round(expected_loss * load / 0.70, 2) if load else None

HOVER_STYLE = dict(bgcolor="#1e293b", font_color="#f8fafc", font_size=12, bordercolor="#334155")

def _base(fig, h=300, margin=None):
    m = margin or dict(l=10, r=60, t=30, b=10)
    fig.update_layout(
        height=h, paper_bgcolor="white", plot_bgcolor="white", margin=m,
        font=dict(family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif", size=11, color="#4a5568"),
        hoverlabel=HOVER_STYLE,
    )
    fig.update_xaxes(gridcolor="#f0f4f8", zeroline=False, tickfont=dict(size=10, color="#718096"))
    fig.update_yaxes(gridcolor="#f0f4f8", zeroline=False, tickfont=dict(size=10, color="#2d3748"))
    return fig


# ── Sensitivity analysis ───────────────────────────────────────────────────────

def _score_with(snap, **overrides):
    """Re-score the snapshotted policy with specific overrides."""
    inputs = dict(snap)
    inputs.update(overrides)
    r = score_homeowner(**inputs)
    p = compute_premium(r.expected_loss, r.decision)
    return r.final_score, r.decision, p

def build_sensitivity(snap, current_score, current_premium):
    """
    Return up to 7 actionable what-if scenarios sorted by biggest score drop.
    Only includes improvements that are:
      1. Feasible given the current property inputs (e.g. won't suggest removing
         a pool that doesn't exist)
      2. Reduce the score by at least 0.4 points
    Each entry: (action_label, new_score, new_decision, new_premium, delta)
    """
    scenarios = []

    def try_action(label, **kwargs):
        ns, nd, np_ = _score_with(snap, **kwargs)
        delta = ns - current_score
        if delta < -0.4:
            scenarios.append((label, round(ns, 1), nd, np_, round(delta, 1)))

    # ── Roof ──────────────────────────────────────────────────────────────────
    if snap["roof_age"] > 15:
        try_action("Replace roof (new, 0 yrs)", roof_age=0)
    if snap["roof_vulnerability"] == "asphalt":
        try_action("Upgrade roof material: asphalt → metal", roof_vulnerability="metal")
    elif snap["roof_vulnerability"] == "tile":
        try_action("Upgrade roof material: tile → metal", roof_vulnerability="metal")
    elif snap["roof_vulnerability"] == "slate":
        try_action("Upgrade roof material: slate → metal", roof_vulnerability="metal")

    # ── Fire sprinklers ────────────────────────────────────────────────────────
    if snap["fire_sprinklers"] == "none":
        try_action("Install full fire sprinkler system", fire_sprinklers="full")
        try_action("Install partial fire sprinkler system", fire_sprinklers="partial")
    elif snap["fire_sprinklers"] == "partial":
        try_action("Upgrade sprinklers: partial → full system", fire_sprinklers="full")

    # ── Monitored alarm ────────────────────────────────────────────────────────
    if snap["monitored_alarm"] == 0:
        try_action("Install monitored alarm system", monitored_alarm=1)

    # ── Recent renovations ─────────────────────────────────────────────────────
    if snap["recent_renovations"] == 0:
        try_action("Complete major renovation", recent_renovations=1)

    # ── Construction upgrade ───────────────────────────────────────────────────
    dc_upgrades = {"wood_frame":"brick_veneer","brick_veneer":"masonry","masonry":"superior"}
    if snap["dwelling_construction"] in dc_upgrades:
        nxt = dc_upgrades[snap["dwelling_construction"]]
        try_action(f"Upgrade construction: {snap['dwelling_construction'].replace('_',' ')} → {nxt.replace('_',' ')}",
                   dwelling_construction=nxt)

    # ── Trampoline ─────────────────────────────────────────────────────────────
    if snap["trampoline"] == 1:
        try_action("Remove trampoline from property", trampoline=0)

    # ── Home business ──────────────────────────────────────────────────────────
    if snap["home_business"] == "active_business":
        try_action("Reduce home business: active → home office only", home_business="home_office")
        try_action("Eliminate home business use entirely", home_business="none")
    elif snap["home_business"] == "home_office":
        try_action("Eliminate home office use", home_business="none")

    # ── Gated community ────────────────────────────────────────────────────────
    if snap["gated_community"] == 0:
        try_action("Move to / qualify as gated community", gated_community=1)

    # ── Swimming pool ──────────────────────────────────────────────────────────
    if snap["swimming_pool"] == "above_ground":
        try_action("Remove above-ground pool", swimming_pool="none")

    # ── Wood stove ─────────────────────────────────────────────────────────────
    if snap["wood_burning_stove"] == "wood_burning":
        try_action("Replace wood stove with gas fireplace", wood_burning_stove="gas_fireplace")
        try_action("Remove wood stove / open fireplace", wood_burning_stove="none")
    elif snap["wood_burning_stove"] == "gas_fireplace":
        try_action("Remove gas fireplace", wood_burning_stove="none")

    # Sort by delta ascending (largest improvement first), return top 7
    scenarios.sort(key=lambda x: x[4])
    return scenarios[:7]


# ── Chart builders ─────────────────────────────────────────────────────────────
def make_waterfall(result):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","relative","subtotal","relative","total"],
        x=["Base (0)","Tier 1","Tier 2","Pre-T3","T3 Effect","Final Score"],
        y=[0, result.tier1_score, result.tier2_adjustment, result.base_score,
           result.final_score - result.base_score, result.final_score],
        text=["0", f"+{result.tier1_score:.1f}", f"{result.tier2_adjustment:+.1f}",
              f"{result.base_score:.1f}", f"x{result.tier3_multiplier:.4f}", f"<b>{result.final_score:.1f}</b>"],
        textposition="outside",
        textfont=dict(size=10, color="#2d3748"),
        connector=dict(line=dict(color="#e2e8f0", width=1.5, dash="dot")),
        increasing=dict(marker=dict(color="#ef4444", line=dict(width=0))),
        decreasing=dict(marker=dict(color="#10b981", line=dict(width=0))),
        totals=dict(marker=dict(color="#3b82f6", line=dict(width=0))),
        customdata=[
            ["Baseline - zero risk points","All risk accumulated by the three tiers above"],
            [f"Tier 1 structural foundation: +{result.tier1_score:.1f} pts","10 property variables (roof, claims, construction, location)"],
            [f"Tier 2 behavioral adjustment: {result.tier2_adjustment:+.1f} pts","Lifestyle factors and mitigant discounts"],
            [f"Base score (T1 + T2): {result.base_score:.1f} pts","Combined before Tier 3 interaction multipliers"],
            [f"Tier 3 multiplier: x{result.tier3_multiplier:.4f}",f"{len(result.tier3.triggered_keys)} co-occurring interaction(s) triggered"],
            [f"Final risk score: {result.final_score:.1f} / 100",f"Decision: {result.decision} - {result.decision_description}"],
        ],
        hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br><span style='color:#94a3b8'>%{customdata[1]}</span><extra></extra>",
    ))
    fig.update_layout(showlegend=False,
                      yaxis=dict(title="Score Points", title_font=dict(size=10, color="#718096"),
                                 range=[0, max(result.final_score * 1.3, 30)]))
    return _base(fig, 290, dict(l=10, r=20, t=30, b=10))

def make_tier_bars(result):
    df  = result.score_components_df()
    t1  = df[df["Tier"]=="Tier 1"]
    t2  = df[(df["Tier"]=="Tier 2") & (df["Points"]!=0)]
    fig = make_subplots(1, 2,
                        subplot_titles=["Tier 1 - Foundation Variables", "Tier 2 - Behavioral Adjustments"],
                        horizontal_spacing=.10)
    if len(t1):
        max_map = {v: TIER1_MAX.get(v, 18) for v in t1["Variable"]}
        r_map   = {v: TIER1_R.get(v, 0)   for v in t1["Variable"]}
        fig.add_trace(go.Bar(
            x=t1["Points"], y=t1["Variable"], orientation="h",
            marker=dict(color=["#ef4444" if v>8 else "#f97316" if v>4 else "#3b82f6"
                               for v in t1["Points"]], line=dict(width=0)),
            text=[f"{v:.0f} pts" for v in t1["Points"]], textposition="outside",
            textfont=dict(size=9, color="#4a5568"),
            customdata=[[max_map.get(v, 18), r_map.get(v, 0)] for v in t1["Variable"]],
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f} pts<br>Max possible: %{customdata[0]} pts<br>Correlation with loss: r = %{customdata[1]:.2f}<extra></extra>",
        ), 1, 1)
    if len(t2):
        fig.add_trace(go.Bar(
            x=t2["Points"], y=t2["Variable"], orientation="h",
            marker=dict(color=["#ef4444" if v>0 else "#10b981" for v in t2["Points"]],
                        line=dict(width=0)),
            text=[f"{v:+.0f}" for v in t2["Points"]], textposition="outside",
            textfont=dict(size=9, color="#4a5568"),
            customdata=[["Risk factor - raises score" if v>0 else "Mitigant - reduces score"] for v in t2["Points"]],
            hovertemplate="<b>%{y}</b><br>Adjustment: %{x:+.1f} pts<br>%{customdata[0]}<extra></extra>",
        ), 1, 2)
        fig.add_vline(x=0, line_color="#e2e8f0", line_width=1.5, row=1, col=2)
    fig.update_annotations(font=dict(size=11, color="#2d3748"))
    fig.update_layout(showlegend=False)
    return _base(fig, 330, dict(l=10, r=70, t=42, b=10))

def make_radar(result):
    t1d   = result.tier1.to_dict()
    cats  = list(t1d.keys())
    short = [c.replace(" (5yr)","").replace(" Amount","").replace(" Distance","") for c in cats]
    norms = [min(100, (t1d[c] / max(TIER1_MAX.get(c, .1), .1)) * 100) for c in cats]
    raw   = [t1d[c] for c in cats]
    mx    = [TIER1_MAX.get(c, 18) for c in cats]
    sc, sn, sr, smx = short+[short[0]], norms+[norms[0]], raw+[raw[0]], mx+[mx[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[50]*len(sc), theta=sc,
        line=dict(color="#e2e8f0", width=1, dash="dot"),
        mode="lines", showlegend=True, name="50% baseline", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=sn, theta=sc, fill="toself",
        fillcolor="rgba(59,130,246,.12)", line=dict(color="#3b82f6", width=2.5),
        mode="lines+markers", marker=dict(size=7, color="#3b82f6"),
        showlegend=True, name="Risk profile",
        customdata=[[sc[i], sr[i], sn[i], smx[i]] for i in range(len(sc))],
        hovertemplate="<b>%{customdata[0]}</b><br>Score: %{customdata[1]:.1f} / %{customdata[3]} pts<br>%{customdata[2]:.0f}% of maximum<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0,100],
                            tickvals=[25,50,75,100], ticktext=["25%","50%","75%","Max"],
                            tickfont=dict(size=9, color="#a0aec0"), gridcolor="#f0f4f8"),
            angularaxis=dict(tickfont=dict(size=11, color="#2d3748"), gridcolor="#f0f4f8"),
        ),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.10,
                    font=dict(size=11, color="#718096")),
        paper_bgcolor="white", height=370,
        margin=dict(l=60, r=60, t=40, b=50),
        font=dict(family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"),
        hoverlabel=HOVER_STYLE,
    )
    return fig

def make_shap(result):
    t1d = result.tier1.to_dict()
    t2d = {k:v for k,v in result.tier2.to_dict().items() if v!=0}
    vars_, vals, tips = [], [], []
    for k, v in t1d.items():
        vars_.append(f"[T1] {k}"); vals.append(v)
        tips.append(f"Tier 1 structural variable - correlation with loss: r={TIER1_R.get(k,0):.2f}<br>Max possible contribution: {TIER1_MAX.get(k,18)} pts")
    for k, v in t2d.items():
        vars_.append(f"[T2] {k}"); vals.append(v)
        tips.append("Behavioral risk factor - increases score" if v>0 else "Behavioral mitigant - discount applied to score")
    if len(result.tier3.triggered_keys) > 0:
        vars_.append("[T3] Interaction Multiplier"); vals.append(result.final_score - result.base_score)
        tips.append(f"{len(result.tier3.triggered_keys)} interaction(s) triggered - multiplier x{result.tier3_multiplier:.4f}<br>Base {result.base_score:.1f} to Final {result.final_score:.1f} pts")
    df = pd.DataFrame({"Variable":vars_,"Value":vals,"Tip":tips}).sort_values("Value", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["Value"], y=df["Variable"], orientation="h",
        marker=dict(color=["#ef4444" if v>0 else "#10b981" for v in df["Value"]], line=dict(width=0)),
        text=[f"{v:+.1f}" for v in df["Value"]], textposition="outside",
        textfont=dict(size=9, color="#2d3748"),
        customdata=list(zip(df["Value"], df["Tip"])),
        hovertemplate="<b>%{y}</b><br>Score contribution: %{customdata[0]:+.2f} pts<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#cbd5e0", line_width=1.5)
    fig.update_layout(showlegend=False,
                      xaxis=dict(title="Score Contribution (pts)",
                                 title_font=dict(size=10, color="#718096")))
    return _base(fig, max(280, len(df)*26+50), dict(l=10, r=75, t=16, b=10))

def make_portfolio_ctx(score, df):
    scores = df["final_risk_score"].values
    pct    = (scores < score).mean() * 100
    fig    = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=60,
        marker=dict(color="#e2e8f0", line=dict(width=0)),
        hovertemplate="Score: %{x:.0f}<br>Policies in this bin: %{y}<extra></extra>",
        name="Portfolio",
    ))
    for lo, hi, c, lbl in [(0,30,"rgba(240,255,244,.6)","Preferred"),
                            (30,60,"rgba(235,248,255,.6)","Standard"),
                            (60,80,"rgba(255,251,235,.6)","Rated"),
                            (80,100,"rgba(255,245,245,.6)","Decline")]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=c, line_width=0,
                      annotation_text=lbl, annotation_position="top left",
                      annotation_font=dict(size=9, color="#718096"))
    fig.add_vline(x=score, line_color="#ef4444", line_width=2.5,
                  annotation_text=f"  This policy ({score:.0f})",
                  annotation_font=dict(size=11, color="#ef4444"),
                  annotation_position="top right")
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Risk Score"),
        yaxis=dict(title="Number of Policies"),
        title=dict(text=f"Portfolio Distribution - this policy is at the {pct:.0f}th percentile",
                   font=dict(size=12, color="#2d3748"), x=0),
    )
    return _base(fig, 250, dict(l=10, r=10, t=36, b=10)), pct

def make_fw_tier1_chart():
    vars_  = list(TIER1_MAX.keys())
    maxpts = list(TIER1_MAX.values())
    rvals  = [TIER1_R[v] for v in vars_]
    fig = make_subplots(1, 2,
        subplot_titles=["Max Score Contribution (pts)", "Correlation with Loss (r)"],
        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(
        x=maxpts, y=vars_, orientation="h",
        marker=dict(color=maxpts, colorscale=[[0,"#bfdbfe"],[0.5,"#3b82f6"],[1,"#1d4ed8"]], line=dict(width=0)),
        text=[f"{v} pts" for v in maxpts], textposition="outside", textfont=dict(size=9),
        customdata=list(zip(vars_, maxpts, rvals)),
        hovertemplate="<b>%{customdata[0]}</b><br>Max contribution: %{customdata[1]} pts<br>Correlation with loss: r = %{customdata[2]:.2f}<extra></extra>",
    ), 1, 1)
    fig.add_trace(go.Bar(
        x=rvals, y=vars_, orientation="h",
        marker=dict(color=rvals, colorscale=[[0,"#d1fae5"],[0.5,"#10b981"],[1,"#065f46"]], line=dict(width=0)),
        text=[f"r={v:.2f}" for v in rvals], textposition="outside", textfont=dict(size=9),
        customdata=list(zip(vars_, maxpts, rvals)),
        hovertemplate="<b>%{customdata[0]}</b><br>Pearson r with expected loss: %{customdata[2]:.2f}<br>Max score contribution: %{customdata[1]} pts<extra></extra>",
    ), 1, 2)
    fig.update_annotations(font=dict(size=11, color="#2d3748"))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 310, dict(l=10, r=70, t=42, b=10))

def make_fw_tier2_chart():
    items = list(TIER2_MAX.items())
    vars_ = [i[0] for i in items]
    vals  = [i[1] for i in items]
    descs = [
        "Coverage gaps signal financial stress - 3yr history",
        "Dangerous breeds carry high liability exposure",
        "Drowning liability and premises injury risk",
        "High-frequency injury liability",
        "Elevated fire ignition risk",
        "Commercial exposure in residential policy",
        "Updated systems reduce loss probability",
        "Theft deterrence and rapid fire response",
        "Most effective fire severity mitigant - full system",
        "Controlled access reduces theft exposure",
    ]
    fig = go.Figure(go.Bar(
        x=vals, y=vars_, orientation="h",
        marker=dict(color=["#ef4444" if v>0 else "#10b981" for v in vals], opacity=0.82, line=dict(width=0)),
        text=[f"{v:+d} pts" for v in vals], textposition="outside", textfont=dict(size=9),
        customdata=list(zip(vars_, vals, descs)),
        hovertemplate="<b>%{customdata[0]}</b><br>Max adjustment: %{customdata[1]:+d} pts<br>%{customdata[2]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#e2e8f0", line_width=1.5)
    fig.update_layout(showlegend=False,
                      xaxis=dict(title="Max score adjustment (pts)", title_font=dict(size=10, color="#718096")),
                      yaxis=dict(autorange="reversed"))
    return _base(fig, 300, dict(l=10, r=70, t=16, b=10))

def make_fw_tier3_chart():
    labels = [t[0] for t in T3_INTERACTIONS]
    mults  = [t[2] for t in T3_INTERACTIONS]
    h_vals = [t[3] for t in T3_INTERACTIONS]
    descs  = [
        "Post-fire erosion on steep grade - mudslide/debris structural damage",
        "Permeable stone/dirt foundation in FEMA flood zone - water ingress virtually certain",
        "Aged roof loses fire resistance; compounded ignition/spread risk in wildfire zone",
        "Trampoline on property in active hail zone - impact liability compound",
        "Repeat-loss policyholder in wildfire corridor - escalated severity",
        "Remote fire station on steep terrain - slow response, higher structural loss",
    ]
    status_raw = [t[4] for t in T3_INTERACTIONS]
    status = ["GBM Confirmed" if s=="CONFIRMED" else "GBM Partial" if s=="PARTIAL" else "Actuarial" for s in status_raw]
    colors = ["#d97706" if s=="GBM Confirmed" else "#f59e0b" if s=="GBM Partial" else "#6366f1" for s in status]
    fig = go.Figure(go.Bar(
        x=h_vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"H={h:.3f}" for h in h_vals], textposition="outside", textfont=dict(size=9),
        customdata=list(zip(labels, mults, status, descs, h_vals)),
        hovertemplate="<b>%{customdata[0]}</b><br>H-Statistic: %{customdata[4]:.4f}<br>Multiplier: x%{customdata[1]:.2f}<br>Status: %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
    ))
    fig.update_layout(showlegend=False,
                      xaxis=dict(title="Friedman H-Statistic (interaction strength; H>=0.30 = Confirmed)",
                                 title_font=dict(size=10, color="#718096"),
                                 range=[0, max(h_vals)*1.20]),
                      yaxis=dict(autorange="reversed"))
    return _base(fig, 280, dict(l=10, r=80, t=16, b=10))

def make_portfolio_dist(df):
    fig = make_subplots(1, 3,
        subplot_titles=["Tier 1 Score Distribution","Tier 2 Adjustment Distribution","Tier 3 Multiplier Distribution"],
        horizontal_spacing=0.09)
    for i, (col, color, tip_prefix) in enumerate([
        ("tier1_score","#3b82f6","Tier 1 foundation score"),
        ("tier2_adjustment","#8b5cf6","Tier 2 behavioral adjustment"),
        ("tier3_multiplier","#f59e0b","Tier 3 interaction multiplier"),
    ], 1):
        if col not in df.columns: continue
        vals = df[col].dropna()
        fig.add_trace(go.Histogram(x=vals, nbinsx=35,
            marker=dict(color=color, opacity=0.75, line=dict(width=0)),
            hovertemplate=f"{tip_prefix}: %{{x:.2f}}<br>Policies in bin: %{{y}}<extra></extra>",
        ), 1, i)
        if i == 2: fig.add_vline(x=0, line_color="#e2e8f0", line_width=1.5, row=1, col=2)
        if i == 3: fig.add_vline(x=1.0, line_color="#e2e8f0", line_width=1.5, row=1, col=3)
    fig.update_annotations(font=dict(size=11, color="#2d3748"))
    fig.update_layout(showlegend=False)
    return _base(fig, 250, dict(l=10, r=10, t=42, b=10))

def make_score_vs_loss(df):
    if "final_risk_score" not in df.columns or "expected_loss" not in df.columns:
        return None
    sample = df.sample(min(1500, len(df)), random_state=42)
    dec_colors = {"Preferred":"#38a169","Standard":"#3182ce","Rated":"#d69e2e","Decline":"#e53e3e"}
    fig = go.Figure()
    for dec in ["Preferred","Standard","Rated","Decline"]:
        if "decision" not in sample.columns: continue
        sub = sample[sample["decision"]==dec]
        if not len(sub): continue
        fig.add_trace(go.Scatter(
            x=sub["final_risk_score"], y=sub["expected_loss"], mode="markers",
            marker=dict(size=4, color=dec_colors[dec], opacity=0.55, line=dict(width=0)),
            name=dec,
            hovertemplate=f"<b>{dec}</b><br>Risk Score: %{{x:.1f}}<br>Expected Loss: $%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, font=dict(size=11)),
                      xaxis=dict(title="Final Risk Score", title_font=dict(size=10, color="#718096")),
                      yaxis=dict(title="Expected Annual Loss ($)", title_font=dict(size=10, color="#718096")),
                      title=dict(text="Risk Score vs Expected Loss - colour by decision",
                                 font=dict(size=12, color="#2d3748"), x=0))
    return _base(fig, 280, dict(l=10, r=10, t=36, b=10))

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
df_portfolio = load_portfolio()

with st.sidebar:
    st.markdown("""
    <div style='padding:20px 6px 14px 6px'>
        <div style='font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#9aa5b4;
                    font-weight:600;margin-bottom:5px'>Homeowners</div>
        <div style='font-size:17px;font-weight:700;color:#1a1f2e;line-height:1.3'>Risk Scoring</div>
        <div style='height:2px;background:linear-gradient(90deg,#3b82f6,transparent);
                    margin:10px 0;border-radius:2px'></div>
    </div>""", unsafe_allow_html=True)

    section = st.radio("NAV", ["🏠  Policy Risk Scorer", "📖  Framework Guide"], label_visibility="collapsed")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if df_portfolio is not None:
        st.markdown("""<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                    color:#9aa5b4;font-weight:600;margin-bottom:8px'>Portfolio</div>""",
                    unsafe_allow_html=True)
        col_name = "final_risk_score" if "final_risk_score" in df_portfolio.columns else "final_score"
        for lbl, val, color in [
            ("Policies",     f"{len(df_portfolio):,}", "#3b82f6"),
            ("Avg Score",    f"{df_portfolio[col_name].mean():.1f}" if col_name in df_portfolio.columns else "-", "#f59e0b"),
            ("Avg Expected Loss", f"${df_portfolio['expected_loss'].mean():,.0f}" if "expected_loss" in df_portfolio.columns else "-", "#ef4444"),
            ("Decline Rate", f"{(df_portfolio['decision']=='Decline').mean()*100:.1f}%" if "decision" in df_portfolio.columns else "-", "#8b5cf6"),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:6px 10px;background:#f8f9fc;border-radius:7px;margin-bottom:4px;
                        border-left:3px solid {color}'>
                <span style='font-size:12px;color:#4a5568'>{lbl}</span>
                <span style='font-size:12px;font-variant-numeric:tabular-nums;color:#1a1f2e;
                             font-weight:600'>{val}</span>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — POLICY RISK SCORER
# ═══════════════════════════════════════════════════════════════════════════════
if "Policy" in section:
    st.markdown("""
    <div class="page-hero">
        <div class="hero-tag">Homeowners Risk Intelligence Platform</div>
        <div class="hero-title">Policy Risk Assessment</div>
    </div>""", unsafe_allow_html=True)

    fc = st.columns(4)
    for col, lbl, fml, color in [
        (fc[1], "Tier 1", "Structural Foundation", "#10b981"),
        (fc[2], "Tier 2", "Behavioural Adjustments", "#8b5cf6"),
        (fc[3], "Tier 3", "Interaction Effect", "#f59e0b"),
    ]:
        col.markdown(f"""<div style='background:#fff;border:1px solid #e8ecf3;border-radius:8px;
            padding:12px 14px;border-top:3px solid {color}'>
            <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                        color:{color};font-weight:700;margin-bottom:3px'>{lbl}</div>
            <div style='font-size:12px;font-variant-numeric:tabular-nums;color:#2d3748'>{fml}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── INPUT FORM ─────────────────────────────────────────────────────────────
    # FIX for stale-input bug:
    #   Streamlit re-renders the whole script on every interaction. If we write
    #   session_state BEFORE the form re-renders (i.e. in the same run where
    #   submitted=True), the new session_state values are picked up by the next
    #   render. The old bug was scoring from session_state instead of from the
    #   widget locals, meaning there was always a one-run lag.
    #
    #   Fix: after the form block we capture all widget locals into `snap`, write
    #   snap to session_state, then score directly from snap. Score and displayed
    #   values are always in sync.
    with st.form("risk_form"):

        # ── Tier 1 ────────────────────────────────────────────────────────────
        st.markdown("""<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>
            <div style='width:28px;height:28px;background:#3b82f6;border-radius:7px;display:flex;
                        align-items:center;justify-content:center;color:white;font-weight:700;
                        font-size:12px;flex-shrink:0'>1</div>
            <div><div style='font-size:14px;font-weight:700;color:#1a1f2e'>Tier 1</div>
        </div></div>""", unsafe_allow_html=True)

        _rv = ["asphalt","tile","slate","metal"]
        _dc = ["wood_frame","brick_veneer","masonry","superior"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            roof_age = st.slider("Roof Age (years)", 0, 50,
                                 value=st.session_state.roof_age)
            water_loss_recency = st.selectbox("Prior Water Claim", [0, 1],
                                    index=[0,1].index(st.session_state.water_loss_recency),
                                    format_func=lambda x: "Yes" if x else "No")
            home_age = st.slider("Home Age (years)", 0, 150,
                                 value=st.session_state.home_age)
        with c2:
            roof_vulnerability = st.selectbox("Roof Material", _rv,
                                    index=_rv.index(st.session_state.roof_vulnerability))
            prior_claims_5yr = st.slider("Prior Claims (5yr)", 0, 8,
                                         value=st.session_state.prior_claims_5yr)
            square_footage = st.slider("Square Footage", 500, 8000,
                                       value=st.session_state.square_footage, step=100)
        with c3:
            dwelling_construction = st.selectbox("Construction Type", _dc,
                                        index=_dc.index(st.session_state.dwelling_construction))
            coverage_a_amount = st.number_input("Coverage A ($)", 100_000, 2_000_000,
                                        value=st.session_state.coverage_a_amount, step=25_000)
            iso_class = st.slider("ISO Class (1=best)", 1, 10,
                                  value=st.session_state.iso_class)
        with c4:
            fire_station_distance = st.slider("Fire Station (miles)", 0.5, 40.0,
                                        value=st.session_state.fire_station_distance, step=0.5)
            st.markdown("""<div style='background:#f7faff;border:1px solid #bee3f8;border-radius:8px;
                padding:11px;font-size:11px;color:#2a4365;line-height:1.6;margin-top:6px'>
                <b>Tier 1</b> captures physical property characteristics.
                Contributes up to <b>100 foundation points</b>.</div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Tier 2 ────────────────────────────────────────────────────────────
        st.markdown("""<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>
            <div style='width:28px;height:28px;background:#8b5cf6;border-radius:7px;display:flex;
                        align-items:center;justify-content:center;color:white;font-weight:700;
                        font-size:12px;flex-shrink:0'>2</div>
            <div><div style='font-size:14px;font-weight:700;color:#1a1f2e'>Tier 2</div>
        </div></div>""", unsafe_allow_html=True)

        _sp = ["none","above_ground","in_ground"]
        _po = ["none","standard_pets","dangerous_breed"]
        _wb = ["none","gas_fireplace","wood_burning"]
        _hb = ["none","home_office","active_business"]
        _fs = ["none","partial","full"]
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            insurance_lapses = st.slider("Insurance Lapses (3yr)", 0, 5,
                                         value=st.session_state.insurance_lapses)
            swimming_pool = st.selectbox("Swimming Pool", _sp,
                                index=_sp.index(st.session_state.swimming_pool))
            recent_renovations = st.selectbox("Recent Renovations", [0, 1],
                                    index=[0,1].index(st.session_state.recent_renovations),
                                    format_func=lambda x: "Yes" if x else "No")
        with c6:
            pet_ownership = st.selectbox("Pet Ownership", _po,
                                index=_po.index(st.session_state.pet_ownership))
            trampoline = st.selectbox("Trampoline", [0, 1],
                                index=[0,1].index(st.session_state.trampoline),
                                format_func=lambda x: "Yes" if x else "No")
            monitored_alarm = st.selectbox("Monitored Alarm", [0, 1],
                                index=[0,1].index(st.session_state.monitored_alarm),
                                format_func=lambda x: "Yes" if x else "No")
        with c7:
            wood_burning_stove = st.selectbox("Heating", _wb,
                                    index=_wb.index(st.session_state.wood_burning_stove))
            home_business = st.selectbox("Home Business", _hb,
                                index=_hb.index(st.session_state.home_business))
            fire_sprinklers = st.selectbox("Fire Sprinklers", _fs,
                                index=_fs.index(st.session_state.fire_sprinklers))
        with c8:
            gated_community = st.selectbox("Gated Community", [0, 1],
                                index=[0,1].index(st.session_state.gated_community),
                                format_func=lambda x: "Yes" if x else "No")
            st.markdown("""<div style='background:#faf5ff;border:1px solid #ddd6fe;border-radius:8px;
                padding:11px;font-size:11px;color:#44337a;line-height:1.6;margin-top:6px'>
                <b>Tier 2</b> captures lifestyle risk. Mitigants reduce score;
                hazards increase it.</div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Tier 3 ────────────────────────────────────────────────────────────
        st.markdown("""<div style='display:flex;align-items:center;gap:12px;margin-bottom:14px'>
            <div style='width:28px;height:28px;background:#f59e0b;border-radius:7px;display:flex;
                        align-items:center;justify-content:center;color:white;font-weight:700;
                        font-size:12px;flex-shrink:0'>3</div>
            <div><div style='font-size:14px;font-weight:700;color:#1a1f2e'>Tier 3</div>
        </div></div>""", unsafe_allow_html=True)

        _cd = ["sparse","moderate","dense"]
        _ft = ["concrete_slab","poured_concrete","block","stone_dirt"]
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            wildfire_zone = st.selectbox("Wildfire Zone", [0, 1],
                                index=[0,1].index(st.session_state.wildfire_zone),
                                format_func=lambda x: "Yes" if x else "No")
            flood_zone = st.selectbox("FEMA Flood Zone", [0, 1],
                                index=[0,1].index(st.session_state.flood_zone),
                                format_func=lambda x: "Yes" if x else "No")
        with c10:
            canopy_density = st.selectbox("Canopy Density", _cd,
                                index=_cd.index(st.session_state.canopy_density))
            foundation_type = st.selectbox("Foundation", _ft,
                                index=_ft.index(st.session_state.foundation_type))
        with c11:
            slope_pct = st.slider("Terrain Slope %", 0.0, 60.0,
                                  value=st.session_state.slope_pct, step=0.5)
            hail_zone = st.selectbox("Hail Zone", [0, 1],
                                index=[0,1].index(st.session_state.hail_zone),
                                format_func=lambda x: "Yes" if x else "No")
        with c12:
            burn_history = st.selectbox("Burn Scar (5mi)", [0, 1],
                                index=[0,1].index(st.session_state.burn_history),
                                format_func=lambda x: "Yes" if x else "No")
            st.markdown("""<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
                padding:11px;font-size:11px;color:#744210;line-height:1.6;margin-top:6px'>
                <b>Tier 3</b> interactions compound risk when two hazards
                co-occur in the same property.</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Evaluate Risk")

    # ── On submit: build snapshot from widget locals, THEN write to session_state,
    #    THEN score from snapshot — never from session_state. This is the core fix.
    if submitted:
        snap = dict(
            roof_age=int(roof_age),
            roof_vulnerability=roof_vulnerability,
            dwelling_construction=dwelling_construction,
            water_loss_recency=int(water_loss_recency),
            prior_claims_5yr=int(prior_claims_5yr),
            coverage_a_amount=int(coverage_a_amount),
            fire_station_distance=float(fire_station_distance),
            home_age=int(home_age),
            square_footage=int(square_footage),
            iso_class=int(iso_class),
            insurance_lapses=int(insurance_lapses),
            pet_ownership=pet_ownership,
            swimming_pool=swimming_pool,
            trampoline=int(trampoline),
            wood_burning_stove=wood_burning_stove,
            home_business=home_business,
            recent_renovations=int(recent_renovations),
            monitored_alarm=int(monitored_alarm),
            fire_sprinklers=fire_sprinklers,
            gated_community=int(gated_community),
            wildfire_zone=int(wildfire_zone),
            canopy_density=canopy_density,
            flood_zone=int(flood_zone),
            foundation_type=foundation_type,
            slope_pct=float(slope_pct),
            burn_history=int(burn_history),
            hail_zone=int(hail_zone),
        )
        # Persist so widgets restore correctly on next render
        for k, v in snap.items():
            st.session_state[k] = v

        # Score from snapshot — guaranteed to match what user saw
        result = score_homeowner(**snap)
        st.session_state._last_result  = result
        st.session_state._last_premium = compute_premium(result.expected_loss, result.decision)
        st.session_state._snap         = snap

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state._last_result is not None:
        result  = st.session_state._last_result
        premium = st.session_state._last_premium
        snap    = st.session_state._snap
        pal     = DEC_PALETTE[result.decision]

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Decision banner ────────────────────────────────────────────────────
        _bg       = pal["bg"]
        _border   = pal["border"]
        _text     = pal["text"]
        _accent   = pal["accent"]
        _decision = result.decision
        _desc     = result.decision_description
        _score    = f"{result.final_score:.0f}"
        _loss     = f"${result.expected_loss:,.0f}"

        if premium is not None:
            _prem_block = (
                "<div style='text-align:center'>"
                f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
                f"color:{_accent};font-weight:600;margin-bottom:3px'>Indicative Premium</div>"
                f"<div style='font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;"
                f"color:{_text};line-height:1.2'>${premium:,.0f}</div>"
                f"<div style='font-size:11px;color:{_text};opacity:.7'>annual estimate</div>"
                "</div>"
            )
        else:
            _prem_block = (
                "<div style='text-align:center'>"
                f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
                f"color:{_accent};font-weight:600;margin-bottom:3px'>Indicative Premium</div>"
                f"<div style='font-size:15px;font-weight:700;color:{_text};line-height:1.4'>"
                "Not Available</div>"
                f"<div style='font-size:11px;color:{_text};opacity:.7'>Refer to E&amp;S market</div>"
                "</div>"
            )

        _banner = (
            f"<div style='background:{_bg};border:1.5px solid {_border};border-radius:14px;"
            "padding:22px 28px;margin-bottom:18px'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px'>"
            "<div>"
            f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:2px;"
            f"color:{_accent};font-weight:700;margin-bottom:5px'>Underwriting Decision</div>"
            f"<div style='font-size:26px;font-weight:700;color:{_text};line-height:1.2'>{_decision}</div>"
            f"<div style='font-size:12px;color:{_text};margin-top:4px;opacity:.8'>{_desc}</div>"
            "</div>"
            "<div style='display:flex;gap:32px;flex-wrap:wrap;align-items:center'>"
            "<div style='text-align:center'>"
            f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
            f"color:{_accent};font-weight:600;margin-bottom:3px'>Risk Score</div>"
            f"<div style='font-size:36px;font-weight:700;font-variant-numeric:tabular-nums;"
            f"color:{_text};line-height:1.1'>{_score}</div>"
            f"<div style='font-size:11px;color:{_text};opacity:.7'>out of 100</div>"
            "</div>"
            "<div style='text-align:center'>"
            f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;"
            f"color:{_accent};font-weight:600;margin-bottom:3px'>Expected Loss</div>"
            f"<div style='font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;"
            f"color:{_text};line-height:1.2'>{_loss}</div>"
            f"<div style='font-size:11px;color:{_text};opacity:.7'>annual</div>"
            "</div>"
            + _prem_block +
            "</div>"
            "</div>"
            "</div>"
        )
        st.markdown(_banner, unsafe_allow_html=True)

        # ── KPI row ────────────────────────────────────────────────────────────
        kc = st.columns(6)
        for col, css, lbl, val, sub in [
            (kc[0], "kpi-blue",   "Tier 1 Score",      f"{result.tier1_score:.1f} pts",      "Foundation"),
            (kc[1], "kpi-purple", "Tier 2 Adjustment",  f"{result.tier2_adjustment:+.1f} pts", "Behavioral"),
            (kc[2], "kpi-amber",  "Tier 3 Multiplier",  f"x{result.tier3_multiplier:.4f}",     f"{len(result.tier3.triggered_keys)} triggered"),
            (kc[3], "kpi-slate",  "Base Score",          f"{result.base_score:.1f}",            "Before T3"),
            (kc[4], "kpi-red",    "Claim Frequency",    f"{result.claim_frequency:.1%}",       "Annual prob"),
            (kc[5], "kpi-green",  "Claim Severity",     f"${result.claim_severity:,.0f}",     "If claim occurs"),
        ]:
            col.markdown(f"""<div class="kpi-card {css}">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # RISK SENSITIVITY ANALYSIS  — compact standalone card
        # ══════════════════════════════════════════════════════════════════════
        if snap:
            scenarios = build_sensitivity(snap, result.final_score, premium)
            if scenarios:
                dec_order = ["Preferred","Standard","Rated","Decline"]
                dec_colors_map = {
                    "Preferred": ("#16a34a","#dcfce7"),
                    "Standard":  ("#2563eb","#dbeafe"),
                    "Rated":     ("#d97706","#fef9c3"),
                    "Decline":   ("#dc2626","#fee2e2"),
                }

                # Build every row as a plain string — no nested f-strings
                all_rows = []
                for action, new_score, new_dec, new_prem, delta in scenarios:
                    dc, dbg = dec_colors_map.get(new_dec, ("#64748b","#f1f5f9"))

                    # decision upgrade badge
                    if dec_order.index(new_dec) < dec_order.index(result.decision):
                        badge = ("<span style='background:" + dbg + ";color:" + dc +
                                 ";border-radius:4px;padding:1px 6px;font-size:10px;"
                                 "font-weight:700;margin-left:4px'>" + new_dec + "</span>")
                    else:
                        badge = ""

                    # premium saving
                    if premium is not None and new_prem is not None and (premium - new_prem) > 0:
                        saving_str = ("<span style='color:#15803d;font-size:11px;margin-left:6px'>"
                                      "save $" + f"{premium - new_prem:,.0f}" + "/yr</span>")
                    else:
                        saving_str = ""

                    row = (
                        "<div style='display:flex;align-items:center;justify-content:space-between;"
                        "padding:8px 12px;border-radius:8px;margin-bottom:5px;"
                        "background:white;border:1px solid #e2e8f0;gap:8px;flex-wrap:wrap'>"
                        "<span style='font-size:12px;color:#1e293b;font-weight:500;flex:1;"
                        "min-width:160px'>" + action + "</span>"
                        "<span style='display:flex;align-items:center;gap:5px;flex-shrink:0'>"
                        "<span style='font-size:12px;color:#94a3b8'>" + str(int(result.final_score)) + "</span>"
                        "<span style='color:#16a34a;font-weight:700'>&#8594;</span>"
                        "<span style='font-size:13px;font-weight:700;color:#14532d'>" + str(int(new_score)) + "</span>"
                        "<span style='font-size:11px;color:#16a34a;font-weight:700;"
                        "background:#dcfce7;border-radius:4px;padding:1px 5px'>"
                        + str(int(delta)) + "</span>"
                        + badge + saving_str +
                        "</span></div>"
                    )
                    all_rows.append(row)

                prem_line = (" &nbsp;·&nbsp; premium <b>$" + f"{premium:,.0f}" + "/yr</b>") if premium else ""
                n = len(scenarios)
                act_word = "action" if n == 1 else "actions"

                card = (
                    "<div style='background:#f8fffe;border:1.5px solid #bbf7d0;"
                    "border-radius:12px;padding:18px 20px;margin-bottom:16px'>"
                    "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                    "<div style='width:26px;height:26px;background:#16a34a;border-radius:7px;"
                    "display:flex;align-items:center;justify-content:center;"
                    "color:white;font-size:14px;font-weight:700;flex-shrink:0'>&#8595;</div>"
                    "<div>"
                    "<div style='font-size:13px;font-weight:700;color:#14532d'>Risk Sensitivity Analysis</div>"
                    "<div style='font-size:11px;color:#4b7c59;margin-top:1px'>"
                    "Score <b style='color:#14532d'>" + str(int(result.final_score)) + "</b>"
                    + prem_line +
                    " &nbsp;&mdash;&nbsp; " + str(n) + " improvement " + act_word + " available"
                    "</div></div></div>"
                    "<div style='height:1px;background:#d1fae5;margin-bottom:10px'></div>"
                    + "".join(all_rows) +
                    "</div>"
                )
                st.markdown(card, unsafe_allow_html=True)

        # ── Analysis tabs ──────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈  Score Breakdown", "🎯  Risk Radar", "🔬  SHAP Analysis",
            "🚦  Flags & Actions", "📍  Portfolio Context"])

        with tab1:
            st.caption("How each tier adds to the final score - hover each bar for a detailed breakdown.")
            col_wf, col_bars = st.columns([1, 2])
            with col_wf:
                st.markdown("**Score construction waterfall**")
                if PLOTLY: st.plotly_chart(make_waterfall(result), use_container_width=True)
            with col_bars:
                st.markdown("**Variable contributions per tier**")
                st.caption("Red = increases risk - Green = discount/mitigant. Hover for max pts and correlation.")
                if PLOTLY: st.plotly_chart(make_tier_bars(result), use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            col_t1, col_t2, col_t3 = st.columns(3)

            with col_t1:
                st.markdown(f"""<div style='background:#eff6ff;border:1px solid #bfdbfe;
                    border-radius:10px;padding:14px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:1px;
                                color:#1d4ed8;font-weight:700;margin-bottom:10px'>
                        Tier 1 - {result.tier1_score:.1f} / 100 pts</div>""", unsafe_allow_html=True)
                for var, pts in sorted(result.tier1.to_dict().items(), key=lambda x: -x[1]):
                    mx  = TIER1_MAX.get(var, 18)
                    pct = min(100, (pts/mx)*100) if mx else 0
                    bc  = "#ef4444" if pct > 75 else "#f97316" if pct > 40 else "#3b82f6"
                    st.markdown(f"""<div style='margin-bottom:8px'>
                        <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                            <span style='font-size:11px;color:#374151;font-weight:500'>{var}</span>
                            <span style='font-size:11px;font-variant-numeric:tabular-nums;
                                         color:{bc};font-weight:700'>{pts:.1f}</span>
                        </div>
                        <div style='height:3px;background:#dbeafe;border-radius:2px'>
                            <div style='height:3px;width:{pct:.0f}%;background:{bc};border-radius:2px'></div>
                        </div>
                        <div style='font-size:10px;color:#9ca3af;margin-top:1px'>r={TIER1_R.get(var,0):.2f} - max {mx} pts</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_t2:
                st.markdown(f"""<div style='background:#faf5ff;border:1px solid #ddd6fe;
                    border-radius:10px;padding:14px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:1px;
                                color:#6d28d9;font-weight:700;margin-bottom:10px'>
                        Tier 2 - {result.tier2_adjustment:+.1f} pts</div>""", unsafe_allow_html=True)
                t2d = {k:v for k,v in result.tier2.to_dict().items() if v != 0}
                if t2d:
                    for var, pts in sorted(t2d.items(), key=lambda x: -abs(x[1])):
                        mx  = abs(TIER2_MAX.get(var, 9)) or 1
                        pct = min(100, (abs(pts)/mx)*100)
                        bc  = "#ef4444" if pts > 0 else "#10b981"
                        lbl_ = "Risk up" if pts > 0 else "Discount"
                        st.markdown(f"""<div style='margin-bottom:8px'>
                            <div style='display:flex;justify-content:space-between;margin-bottom:2px'>
                                <span style='font-size:11px;color:#374151;font-weight:500'>{var}</span>
                                <span style='font-size:11px;font-variant-numeric:tabular-nums;
                                             color:{bc};font-weight:700'>{pts:+.1f}</span>
                            </div>
                            <div style='height:3px;background:#ede9fe;border-radius:2px'>
                                <div style='height:3px;width:{pct:.0f}%;background:{bc};border-radius:2px'></div>
                            </div>
                            <div style='font-size:10px;color:#9ca3af;margin-top:1px'>{lbl_}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size:12px;color:#94a3b8;padding:8px 0">No active adjustments</div>',
                                unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_t3:
                st.markdown(f"""<div style='background:#fffbeb;border:1px solid #fde68a;
                    border-radius:10px;padding:14px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:1px;
                                color:#b45309;font-weight:700;margin-bottom:10px'>
                        Tier 3 - x{result.tier3_multiplier:.4f}</div>""", unsafe_allow_html=True)
                for name, attr, mult, h_stat, ml_status in T3_INTERACTIONS:
                    trig = getattr(result.tier3, attr)
                    bc   = "#f59e0b" if trig else "#d1d5db"
                    bg_  = "#fef3c7" if trig else "#f9fafb"
                    icon = "Active" if trig else "-"
                    st.markdown(f"""<div style='background:{bg_};border-radius:6px;padding:8px 10px;
                        margin-bottom:5px;border:1px solid {"#fde68a" if trig else "#e5e7eb"}'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                            <span style='font-size:11px;color:{"#92400e" if trig else "#9ca3af"};
                                         font-weight:{"700" if trig else "400"}'>{icon} {name}</span>
                            <span style='font-size:11px;font-variant-numeric:tabular-nums;
                                         color:{bc};font-weight:700'>x{mult:.2f} - H={h_stat:.3f}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                if len(result.tier3.triggered_keys) == 0:
                    st.markdown("""<div style='text-align:center;padding:12px;color:#6b7280;font-size:12px'>
                        No interactions triggered - Multiplier x1.0000</div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            if PLOTLY:
                rc1, rc2 = st.columns([2, 1])
                with rc1:
                    st.plotly_chart(make_radar(result), use_container_width=True)
                with rc2:
                    t1d  = result.tier1.to_dict()
                    top3 = sorted(t1d.items(), key=lambda x: -x[1])[:3]
                    st.markdown("""<div class="insight-box">
                        <div class="insight-title">How to Read</div>
                        <div class="insight-text">Each axis represents a Tier 1 variable scaled to
                        its <b>maximum possible score</b>. The dotted ring at 50% is the average-risk
                        reference. Area beyond the ring = elevated structural risk.</div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("""<div class="insight-box" style='background:#fff5f5;border-color:#feb2b2'>
                        <div class="insight-title" style='color:#c53030'>Top Structural Drivers</div>""",
                        unsafe_allow_html=True)
                    for var, pts in top3:
                        mx  = TIER1_MAX.get(var, 18)
                        pct = (pts/mx*100) if mx else 0
                        st.markdown(f"""<div style='font-size:12px;padding:4px 0;border-bottom:1px solid #fee2e2'>
                            <b>{var}</b> - {pts:.1f} pts ({pct:.0f}% of max)</div>""",
                            unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(f"""<div class="insight-box" style='background:#f0fff4;border-color:#9ae6b4'>
                        <div class="insight-title" style='color:#276749'>Score Summary</div>
                        <div class="insight-text">
                            T1: <b>{result.tier1_score:.1f} pts</b><br>
                            T2: <b>{result.tier2_adjustment:+.1f} pts</b><br>
                            T3: <b>x{result.tier3_multiplier:.4f}</b><br>
                            Final: <b>{result.final_score:.1f} / 100</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

        with tab3:
            st.caption("Red = increases risk score - Green = reduces score. Hover for tier, correlation, and variable detail.")
            if PLOTLY: st.plotly_chart(make_shap(result), use_container_width=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**Expected Loss & Premium Build-Up**")
            ec1, ec2, ec3 = st.columns(3)
            ec1.markdown(f"""<div class="kpi-card kpi-red">
                <div class="kpi-label">Claim Frequency</div>
                <div class="kpi-value">{result.claim_frequency:.1%}</div>
                <div class="kpi-sub">Annual probability of claim</div></div>""", unsafe_allow_html=True)
            ec2.markdown(f"""<div class="kpi-card kpi-amber">
                <div class="kpi-label">Avg Claim Severity</div>
                <div class="kpi-value">${result.claim_severity:,.0f}</div>
                <div class="kpi-sub">Expected cost if claim occurs</div></div>""", unsafe_allow_html=True)
            ec3.markdown(f"""<div class="kpi-card kpi-blue">
                <div class="kpi-label">Expected Annual Loss</div>
                <div class="kpi-value">${result.expected_loss:,.0f}</div>
                <div class="kpi-sub">Frequency x Severity</div></div>""", unsafe_allow_html=True)

            if premium is not None:
                load_lbl = ("1.25x" if result.decision == "Preferred"
                            else "1.55x" if result.decision == "Standard"
                            else "2.00x")
                st.markdown(f"""
                <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                            padding:16px 20px;margin-top:14px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:1px;
                                color:#1d4ed8;font-weight:700;margin-bottom:10px'>Premium Build-Up</div>
                    <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px'>
                        <div><div style='font-size:11px;color:#718096'>Pure Premium</div>
                             <div style='font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;
                                         color:#1a1f2e'>${result.expected_loss:,.0f}</div>
                             <div style='font-size:10px;color:#9ca3af'>Expected Loss</div></div>
                        <div><div style='font-size:11px;color:#718096'>Load Factor</div>
                             <div style='font-size:16px;font-weight:700;color:#1a1f2e'>{load_lbl}</div>
                             <div style='font-size:10px;color:#9ca3af'>{result.decision} tier</div></div>
                        <div><div style='font-size:11px;color:#718096'>Expense Loading</div>
                             <div style='font-size:16px;font-weight:700;color:#1a1f2e'>30%</div></div>
                        <div><div style='font-size:11px;color:#1d4ed8;font-weight:700'>Indicative Premium</div>
                             <div style='font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;
                                         color:#1d4ed8'>${premium:,.0f}</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#fff5f5;border:1px solid #feb2b2;border-radius:10px;
                            padding:16px 20px;margin-top:14px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:1px;
                                color:#c53030;font-weight:700;margin-bottom:6px'>Premium Not Available</div>
                    <div style='font-size:12px;color:#742a2a;line-height:1.6'>
                        This policy has been declined for standard market placement.
                        No indicative premium can be generated. Refer to the
                        E&amp;S (Excess &amp; Surplus) lines market for potential placement options.
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab4:
            st.markdown("**Underwriting Flags**")
            flags = result.all_flags
            if not flags:
                st.markdown('<div class="flag-card flag-GOOD">Clean risk profile - no concerns. Auto-bind eligible.</div>',
                            unsafe_allow_html=True)
            else:
                for category, icon, cls, label in [
                    ("HIGH",  "🔴", "flag-HIGH",  "High Risk - Immediate Action Required"),
                    ("TIER3", "🔷", "flag-TIER3", "Tier 3 Interactions - Amplified Signals"),
                    ("WARN",  "⚠️", "flag-WARN",  "Warnings - Elevated Factors"),
                    ("GOOD",  "✅", "flag-GOOD",  "Risk Mitigants - Credits Applied"),
                ]:
                    subset = [f for f in flags if
                              ("[T3]" in f.upper() if category == "TIER3"
                               else "HIGH RISK" in f.upper() if category == "HIGH"
                               else "WARNING" in f.upper() and "[T3]" not in f.upper() if category == "WARN"
                               else "GOOD" in f.upper())]
                    if subset:
                        st.markdown(f"**{icon} {label}**")
                        for f in subset:
                            st.markdown(f'<div class="flag-card {cls}">{f}</div>', unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**Underwriting Recommendation**")
            rec_map = {
                "Preferred": ("Auto-Bind Eligible","#f0fff4","#22543d","#38a169",
                    "Qualifies for automated binding at preferred rates. No manual review required. Apply preferred tier discounts and confirm coverage limits within bind authority."),
                "Standard":  ("Standard Issue","#ebf8ff","#2a4365","#3182ce",
                    "Issue at standard rates with +/-15% band. Desktop review recommended. Confirm all discounts are properly applied."),
                "Rated":     ("Manual Review Required","#fffbeb","#744210","#d69e2e",
                    "Senior underwriter review required before binding. Apply 15-50% surcharge. Mandatory property inspection. Evaluate coverage exclusions."),
                "Decline":   ("Refer to Specialty Market","#fff5f5","#742a2a","#e53e3e",
                    "Standard market binding not recommended. Refer to E&S lines. Provide written explanation of underwriting concerns. Document all Tier 3 interaction triggers."),
            }
            badge, bg_, text_, accent_, desc_ = rec_map[result.decision]
            st.markdown(f"""<div style='background:{bg_};border:1.5px solid {accent_}44;border-radius:10px;
                padding:18px 20px;border-left:4px solid {accent_}'>
                <div style='font-size:14px;font-weight:700;color:{text_};margin-bottom:7px'>{badge}</div>
                <div style='font-size:12px;color:{text_};line-height:1.7;opacity:.9'>{desc_}</div>
            </div>""", unsafe_allow_html=True)

        with tab5:
            st.caption("Where this policy sits relative to the full portfolio. Hover for policy counts per score bin.")
            if df_portfolio is not None and PLOTLY:
                col_name = "final_risk_score" if "final_risk_score" in df_portfolio.columns else "final_score"
                if col_name in df_portfolio.columns:
                    fig_ctx, pct = make_portfolio_ctx(result.final_score, df_portfolio)
                    st.plotly_chart(fig_ctx, use_container_width=True)
                    p1, p2, p3 = st.columns(3)
                    p1.markdown(f"""<div class="kpi-card kpi-blue">
                        <div class="kpi-label">Risk Percentile</div>
                        <div class="kpi-value">{pct:.0f}th</div>
                        <div class="kpi-sub">Riskier than {pct:.0f}% of portfolio</div></div>""",
                        unsafe_allow_html=True)
                    p2.markdown(f"""<div class="kpi-card kpi-slate">
                        <div class="kpi-label">Portfolio Avg Score</div>
                        <div class="kpi-value">{df_portfolio[col_name].mean():.1f}</div>
                        <div class="kpi-sub">This policy: {result.final_score:.1f}</div></div>""",
                        unsafe_allow_html=True)
                    if "decision" in df_portfolio.columns:
                        p3.markdown(f"""<div class="kpi-card kpi-purple">
                            <div class="kpi-label">Same Decision</div>
                            <div class="kpi-value">{(df_portfolio["decision"]==result.decision).mean()*100:.1f}%</div>
                            <div class="kpi-sub">of portfolio is {result.decision}</div></div>""",
                            unsafe_allow_html=True)
            else:
                st.info("Run `python eda.py` to generate portfolio data for context comparison.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FRAMEWORK GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Framework" in section:
    st.markdown("""
    <div class="page-hero">
        <div class="hero-tag">📖 Documentation</div>
        <div class="hero-title">3-Tier Scoring Framework</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Decision Thresholds</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    tc = st.columns(4)
    for col, rng, lbl, color, bg_, bord_, desc_ in [
        (tc[0],"0-30",   "Preferred","#38a169","#f0fff4","#9ae6b4","Auto-bind. No manual review."),
        (tc[1],"31-60",  "Standard", "#3182ce","#ebf8ff","#90cdf4","Standard rate +/-15%. Desktop review."),
        (tc[2],"61-80",  "Rated",    "#d69e2e","#fffbeb","#fbd38d","15-50% surcharge. Sr. UW review."),
        (tc[3],"81-100", "Decline",  "#e53e3e","#fff5f5","#feb2b2","Refer E&S. Do not bind."),
    ]:
        col.markdown(f"""<div style='background:{bg_};border:1px solid {bord_};border-radius:10px;
            padding:16px;text-align:center'>
            <div style='font-size:18px;font-weight:700;color:{color};
                        font-variant-numeric:tabular-nums;margin-bottom:3px'>{rng}</div>
            <div style='font-size:13px;font-weight:700;color:#1a1f2e;margin-bottom:5px'>{lbl}</div>
            <div style='font-size:11px;color:#4a5568;line-height:1.5'>{desc_}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Variable Scoring Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Hover over each bar for the variable name, max contribution, and actuarial correlation.")

    fw_tab1, fw_tab2, fw_tab3 = st.tabs([
        "Tier 1 - Foundation Variables",
        "Tier 2 - Behavioral Adjustments",
        "Tier 3 - Interaction Multipliers",
    ])

    with fw_tab1:
        st.markdown("""<div class="insight-box" style='margin-bottom:12px'>
            <div class="insight-title">Tier 1 - Structural Foundation</div>
            <div class="insight-text">10 structural variables contributing up to 100 points.
            Left chart shows maximum score contribution per variable; right chart shows Pearson correlation
            with expected annual loss. Higher correlation = stronger actuarial signal.
            Hover any bar for full detail.</div>
        </div>""", unsafe_allow_html=True)
        if PLOTLY: st.plotly_chart(make_fw_tier1_chart(), use_container_width=True)
        t1_df = pd.DataFrame({
            "Variable":    list(TIER1_MAX.keys()),
            "Max Pts":     list(TIER1_MAX.values()),
            "Corr (r)":    [f"{v:.2f}" for v in TIER1_R.values()],
            "Significance":["Very High" if v>=0.35 else "High" if v>=0.20 else "Moderate" if v>=0.10 else "Low"
                            for v in TIER1_R.values()],
        })
        st.dataframe(t1_df.sort_values("Max Pts", ascending=False).reset_index(drop=True),
                     hide_index=True, use_container_width=True)

    with fw_tab2:
        st.markdown("""<div class="insight-box" style='margin-bottom:12px'>
            <div class="insight-title">Tier 2 - Behavioral Adjustment</div>
            <div class="insight-text">10 lifestyle variables adjusting the base score by up to +/-30 points.
            Red bars increase risk; green bars are mitigants that reduce the score.
            Hover for the rationale behind each adjustment.</div>
        </div>""", unsafe_allow_html=True)
        if PLOTLY: st.plotly_chart(make_fw_tier2_chart(), use_container_width=True)
        t2_rows = [
            ("Insurance Lapses",   +9, "Risk Factor", "Coverage gaps signal financial stress - 3yr history"),
            ("Pet Ownership",      +8, "Risk Factor", "Dangerous breeds carry high liability exposure"),
            ("Swimming Pool",      +6, "Risk Factor", "Drowning liability and premises injury risk"),
            ("Trampoline",         +5, "Risk Factor", "High-frequency injury liability"),
            ("Wood-Burning Stove", +4, "Risk Factor", "Elevated fire ignition risk"),
            ("Home Business",      +4, "Risk Factor", "Commercial exposure in residential policy"),
            ("Fire Sprinklers",    -7, "Mitigant",    "Most effective fire severity mitigant - full system"),
            ("Recent Renovations", -6, "Mitigant",    "Updated systems reduce loss probability"),
            ("Monitored Alarm",    -5, "Mitigant",    "Theft deterrence and rapid fire response"),
            ("Gated Community",    -2, "Mitigant",    "Controlled access reduces theft exposure"),
        ]
        t2_df = pd.DataFrame(t2_rows, columns=["Variable","Max Pts","Type","Rationale"])
        st.dataframe(t2_df.sort_values("Max Pts", key=abs, ascending=False).reset_index(drop=True),
                     hide_index=True, use_container_width=True)

    with fw_tab3:
        if PLOTLY: st.plotly_chart(make_fw_tier3_chart(), use_container_width=True)
        _t3art = get_artifacts()
        if _t3art is not None and "interaction_df" in _t3art:
            pkl_s = list(_t3art["interaction_df"]["Status"])
        else:
            pkl_s = []
        _t3descs = [
            "Post-fire erosion on steep grade - mudslide/debris structural damage",
            "Permeable stone/dirt foundation in FEMA flood zone - water ingress virtually certain",
            "Aged roof loses fire resistance; compounded ignition/spread risk in wildfire zone",
            "Trampoline on property in active hail zone - impact liability compound",
            "Repeat-loss policyholder in wildfire corridor - escalated severity",
            "Remote fire station on steep terrain - slow response, higher structural loss",
        ]
        t3_rows = [
            (lbl,
             f"x{mult:.2f}",
             f"H={h:.4f}",
             "GBM Confirmed" if st=="CONFIRMED" else "GBM Partial" if st=="PARTIAL" else "Actuarial",
             _t3descs[i])
            for i, (lbl, _, mult, h, st) in enumerate(T3_INTERACTIONS)
        ]
        t3_df = pd.DataFrame(t3_rows, columns=["Interaction","Multiplier","H-Statistic","ML Status","Rationale"])
        st.dataframe(t3_df, hide_index=True, use_container_width=True)

    if df_portfolio is not None and PLOTLY:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Portfolio Score Distributions</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.caption("Distribution of Tier 1, 2 and 3 scores across the 10,000-policy portfolio. Hover for bin count.")
        st.plotly_chart(make_portfolio_dist(df_portfolio), use_container_width=True)

        col_sl, col_blank = st.columns([2, 1])
        with col_sl:
            fig_sl = make_score_vs_loss(df_portfolio)
            if fig_sl:
                st.caption("Risk score vs expected loss - each dot is a sampled policy, colour by decision segment.")
                st.plotly_chart(fig_sl, use_container_width=True)
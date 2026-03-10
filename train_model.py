"""
train_model.py — Model Training Pipeline (ROOT LEVEL)
=======================================================
Place this file at: HOMEOWNERS_RISK_SCORING/train_model.py

Step 1: Tweedie GLM  — actuarial baseline pricing model
Step 2: GBM          — GradientBoostingRegressor for interaction discovery
Step 3: Feature Importance — validates Tier 1/2/3 signal ordering
Step 4: Interaction Detection — confirms Tier 3 pairs

Run from VS Code terminal:
    python train_model.py
"""

import sys, os
# Add scoring/ to path so score_engine etc. can be imported if needed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring"))

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ── Paths (all relative to this file = project root) ─────────────────────────
ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "homeowners_risk_dataset.csv"
EDA_DIR   = ROOT / "eda_outputs"
MODEL_DIR = ROOT                          # save trained_models.pkl at root
EDA_DIR.mkdir(exist_ok=True)

COLORS = {
    "tier1":  "#2c3e50",
    "tier2":  "#8e44ad",
    "tier3":  "#e67e22",
    "good":   "#2ecc71",
    "warn":   "#f39c12",
    "bad":    "#e74c3c",
}

# ── Feature Groups ────────────────────────────────────────────────────────────
TIER1_FEATURES = [
    "roof_age", "roof_vulnerability", "dwelling_construction",
    "water_loss_recency", "prior_claims_5yr", "coverage_a_amount",
    "fire_station_distance", "home_age", "square_footage", "iso_class",
]
TIER2_FEATURES = [
    "insurance_lapses", "pet_ownership", "swimming_pool", "trampoline",
    "wood_burning_stove", "home_business", "recent_renovations",
    "monitored_alarm", "fire_sprinklers", "gated_community",
]
ENV_FEATURES = [
    "wildfire_zone", "canopy_density", "flood_zone", "foundation_type",
    "slope_pct", "burn_history", "hail_zone",
]
ALL_FEATURES = TIER1_FEATURES + TIER2_FEATURES + ENV_FEATURES

CATEGORICAL_COLS = [
    "roof_vulnerability", "dwelling_construction", "pet_ownership",
    "canopy_density", "foundation_type",
]
# Ordinal encode all categorical features where alphabetical order would invert risk direction.
# Rule: higher integer = MORE risk (consistent with numeric features).
#   fire_sprinklers: none(0) < partial(1) < full(2) → INVERTED for GBM (more sprinklers = less risk)
#     but GBM handles monotone constraints internally; we just need consistent ordering.
#     Use none=0, partial=1, full=2 so GBM correctly learns full → lowest prediction.
#   swimming_pool:   none=0 < above_ground=1 < in_ground=2 (none=least risk)
#   home_business:   none=0 < home_office=1 < active_business=2 (none=least risk)
#   wood_burning_stove: none=0 < gas_fireplace=1 < wood_burning=2
ORDINAL_COLS = {
    "fire_sprinklers":   {"none": 0, "partial": 1, "full": 2},
    "swimming_pool":     {"none": 0, "above_ground": 1, "in_ground": 2},
    "home_business":     {"none": 0, "home_office": 1, "active_business": 2},
    "wood_burning_stove":{"none": 0, "gas_fireplace": 1, "wood_burning": 2},
}
TARGET = "expected_loss"


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df):
    X = df[ALL_FEATURES].copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    # Ordinal encode fire_sprinklers: none=0, partial=1, full=2
    # LabelEncoder sorts alphabetically (full=0, none=1, partial=2) which inverts
    # the risk direction — ordinal encoding fixes this so GBM sees correct ordering.
    for col, mapping in ORDINAL_COLS.items():
        X[col] = X[col].astype(str).str.lower().map(mapping).fillna(0).astype(int)
        encoders[col] = mapping
    return X, encoders


def load_and_split(test_size=0.2):
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")
    X, encoders = prepare_features(df)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, encoders, df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TWEEDIE GLM
# ─────────────────────────────────────────────────────────────────────────────

def train_tweedie(X_train, X_test, y_train, y_test):
    print("\n  [STEP 1] Tweedie GLM (Baseline Pricing Model)")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = TweedieRegressor(power=1.5, alpha=0.1, max_iter=500)
    model.fit(X_tr_s, y_train)

    tr_pred = model.predict(X_tr_s)
    te_pred = model.predict(X_te_s)

    metrics = {
        "train_r2":   r2_score(y_train, tr_pred),
        "test_r2":    r2_score(y_test,  te_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, tr_pred)),
        "test_rmse":  np.sqrt(mean_squared_error(y_test,  te_pred)),
    }
    print(f"    Train R2: {metrics['train_r2']:.4f}  |  Test R2: {metrics['test_r2']:.4f}")
    print(f"    Train RMSE: ${metrics['train_rmse']:,.0f}  |  Test RMSE: ${metrics['test_rmse']:,.0f}")

    # Top coefficients
    coef_df = pd.DataFrame({
        "Feature": ALL_FEATURES, "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)
    print("    Top 5 coefficients:")
    for _, row in coef_df.head(5).iterrows():
        sign = "+" if row["Coefficient"] > 0 else ""
        print(f"      {row['Feature']:<30} {sign}{row['Coefficient']:.4f}")

    return model, scaler, metrics, te_pred


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GBM
# ─────────────────────────────────────────────────────────────────────────────

def train_gbm(X_train, X_test, y_train, y_test):
    print("\n  [STEP 2] Gradient Boosting Model (GBM)")
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, subsample=0.8, random_state=42, loss="huber",
    )
    model.fit(X_train, y_train)

    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)

    metrics = {
        "train_r2":   r2_score(y_train, tr_pred),
        "test_r2":    r2_score(y_test,  te_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, tr_pred)),
        "test_rmse":  np.sqrt(mean_squared_error(y_test,  te_pred)),
    }
    print(f"    Train R2: {metrics['train_r2']:.4f}  |  Test R2: {metrics['test_r2']:.4f}")
    print(f"    Train RMSE: ${metrics['train_rmse']:,.0f}  |  Test RMSE: ${metrics['test_rmse']:,.0f}")

    return model, metrics, te_pred


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance(gbm, X_test, y_test):
    print("\n  [STEP 3] Feature Importance")

    imp_df = pd.DataFrame({
        "Feature": ALL_FEATURES, "Importance": gbm.feature_importances_,
    }).sort_values("Importance", ascending=False)

    perm = permutation_importance(gbm, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Perm_Mean": perm.importances_mean,
        "Perm_Std":  perm.importances_std,
    }).sort_values("Perm_Mean", ascending=False)

    print("    Top 10 by GBM impurity importance:")
    for _, row in imp_df.head(10).iterrows():
        tier = "T1" if row["Feature"] in TIER1_FEATURES else \
               "T2" if row["Feature"] in TIER2_FEATURES else "ENV"
        bar = "█" * max(1, int(row["Importance"] * 150))
        print(f"    [{tier}] {row['Feature']:<28} {row['Importance']:.4f}  {bar}")

    return imp_df, perm_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TIER 3 INTERACTION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _h_statistic(gbm, X_sample, feat1_idx, feat2_idx, grid_points=15):
    """
    Friedman H-statistic — measures genuine non-linear interaction strength.

    H^2 = Var[PD_12 - PD_1 - PD_2] / Var[PD_12]

    H=0   → purely additive, no interaction
    H=0.1 → weak but detectable interaction
    H=0.3 → strong interaction (used as confirmation threshold)
    H→1   → joint effect entirely non-additive
    """
    from sklearn.inspection import partial_dependence
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pd_joint = partial_dependence(
            gbm, X_sample, features=[feat1_idx, feat2_idx],
            kind="average", grid_resolution=grid_points,
        )
        pd1 = partial_dependence(gbm, X_sample, features=[feat1_idx],
                                 kind="average", grid_resolution=grid_points)
        pd2 = partial_dependence(gbm, X_sample, features=[feat2_idx],
                                 kind="average", grid_resolution=grid_points)

    v1  = pd1["average"][0]           # (grid,)
    v2  = pd2["average"][0]           # (grid,)
    v12 = pd_joint["average"][0]      # (grid, grid)

    # Centre all PDPs before differencing (removes intercept noise)
    v12c = v12 - v12.mean()
    v1c  = (v1 - v1.mean())[:, np.newaxis]
    v2c  = (v2 - v2.mean())[np.newaxis, :]

    num = np.var(v12c - v1c - v2c)
    den = np.var(v12c)
    return float(np.sqrt(num / den)) if den > 1e-10 else 0.0


def discover_tier3_interactions(gbm, X_train, imp_df):
    """
    STEP 4 — Tier 3 Interaction Discovery.

    This is ML-first: we scan all scientifically plausible feature pairs,
    rank them by Friedman H-statistic, and the TOP pairs BECOME Tier 3.
    We do NOT pre-define the interactions and then validate them.

    Candidate pool: all pairs where at least one feature is an ENV variable
    (the natural amplifiers in the framework). This gives ~17×27/2 ≈ 170 pairs,
    scanned in order of (imp1 × imp2) to prioritise signal-rich pairs.

    Confirmation criteria:
        H >= 0.30  → CONFIRMED  (strong non-additive interaction)
        H >= 0.10  → PARTIAL    (weak interaction, monitor)
        H <  0.10  → additive   (not included in Tier 3)

    Returns the top 6 confirmed/partial pairs sorted by H descending,
    which define the Tier 3 multipliers in tier3.py.
    """
    print("\n  [STEP 4] Tier 3 Interaction Discovery — Exhaustive H-Statistic Scan")

    feat_names = list(X_train.columns)
    feat_idx   = {f: i for i, f in enumerate(feat_names)}

    # ── Candidate pairs ───────────────────────────────────────────────────────
    # All pairs where at least one feature is an ENV variable
    env_set   = set(ENV_FEATURES)
    all_feats = feat_names  # 27 features

    candidates = []
    for i, f1 in enumerate(all_feats):
        for f2 in all_feats[i+1:]:
            if f1 in env_set or f2 in env_set:   # at least one ENV feature
                imp1 = imp_df[imp_df["Feature"] == f1]["Importance"].values
                imp2 = imp_df[imp_df["Feature"] == f2]["Importance"].values
                if len(imp1) and len(imp2):
                    priority = float(imp1[0]) * float(imp2[0])
                    candidates.append((priority, f1, f2))

    # Sort by joint importance descending — compute H for highest-signal pairs first
    candidates.sort(reverse=True)
    print(f"    {len(candidates)} candidate pairs identified")

    # Subsample rows for speed (H-stat computes PD grids of size grid^2 per pair)
    np.random.seed(42)
    idx      = np.random.choice(len(X_train), min(600, len(X_train)), replace=False)
    X_sample = X_train.iloc[idx].reset_index(drop=True).astype(float)

    print(f"    Scanning top pairs on {len(X_sample)}-row sample ...")

    H_CONFIRM = 0.30   # strong non-additive interaction
    H_PARTIAL = 0.10   # weak but detectable

    all_results = []
    for rank, (priority, f1, f2) in enumerate(candidates, 1):
        h = _h_statistic(gbm, X_sample, feat_idx[f1], feat_idx[f2])
        imp1 = imp_df[imp_df["Feature"] == f1]["Importance"].values[0]
        imp2 = imp_df[imp_df["Feature"] == f2]["Importance"].values[0]
        status = "CONFIRMED" if h >= H_CONFIRM else "PARTIAL" if h >= H_PARTIAL else "additive"
        all_results.append({
            "Feat1": f1, "Feat2": f2,
            "H_Statistic": round(h, 4),
            "Feat1_Imp": round(imp1, 4),
            "Feat2_Imp": round(imp2, 4),
            "Priority": round(priority, 6),
            "Status": status,
        })
        if h >= H_PARTIAL:
            tier1 = "T1" if f1 in TIER1_FEATURES else "T2" if f1 in TIER2_FEATURES else "ENV"
            tier2 = "T1" if f2 in TIER1_FEATURES else "T2" if f2 in TIER2_FEATURES else "ENV"
            print(f"    [{status:<9}] [{tier1}]{f1} × [{tier2}]{f2}  H={h:.4f}")

    all_df = pd.DataFrame(all_results).sort_values("H_Statistic", ascending=False)

    # Top 6 significant pairs become Tier 3
    top6 = all_df[all_df["Status"] != "additive"].head(6).reset_index(drop=True)

    # Build canonical interaction labels
    def _label(r):
        return f"{r['Feat1'].replace('_',' ').title()} × {r['Feat2'].replace('_',' ').title()}"

    top6["Interaction"] = top6.apply(_label, axis=1)

    confirmed_n = (top6["Status"] == "CONFIRMED").sum()
    print(f"\n    ── Tier 3 Interactions (ML-Discovered) ──")
    print(f"    {confirmed_n} CONFIRMED  |  {len(top6)-confirmed_n} PARTIAL  "
          f"(threshold: CONFIRMED H≥{H_CONFIRM}, PARTIAL H≥{H_PARTIAL})")
    for _, row in top6.iterrows():
        print(f"    [{row['Status']:<9}] {row['Interaction']:<45} H={row['H_Statistic']:.4f}")

    # Also return the full scan for the chart
    top6["All_Scan"] = False  # marker
    all_df["All_Scan"] = True
    return top6, all_df


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def tier_color(feat):
    if feat in TIER1_FEATURES: return COLORS["tier1"]
    if feat in TIER2_FEATURES: return COLORS["tier2"]
    return COLORS["tier3"]


def plot_actual_vs_predicted(y_test, tw_preds, gbm_preds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Actual vs Predicted Expected Loss", fontsize=13, fontweight="bold")
    for ax, preds, title, color in [
        (axes[0], tw_preds,  "Tweedie GLM", COLORS["tier2"]),
        (axes[1], gbm_preds, "GBM",         COLORS["tier1"]),
    ]:
        ax.scatter(y_test, preds, alpha=0.2, s=8, color=color)
        lim = [0, max(float(y_test.max()), float(max(preds))) * 1.05]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
        r2   = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        ax.set_title(f"{title}  |  R2={r2:.4f}  RMSE=${rmse:,.0f}")
        ax.set_xlabel("Actual ($)")
        ax.set_ylabel("Predicted ($)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    plt.tight_layout()
    p = EDA_DIR / "actual_vs_predicted.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: actual_vs_predicted.png")


def plot_feature_importance(imp_df, perm_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Feature Importance — GBM Validation of Tier Signal Ordering",
                 fontsize=13, fontweight="bold")
    top_n = 20
    patches = [
        mpatches.Patch(color=COLORS["tier1"], label="Tier 1", alpha=0.85),
        mpatches.Patch(color=COLORS["tier2"], label="Tier 2", alpha=0.85),
        mpatches.Patch(color=COLORS["tier3"], label="Environment", alpha=0.85),
    ]

    # Impurity importance
    ax = axes[0]
    top = imp_df.head(top_n)
    ax.barh(range(len(top)), top["Importance"],
            color=[tier_color(f) for f in top["Feature"]],
            edgecolor="white", alpha=0.88)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Impurity Importance (MDI)")
    ax.set_title("GBM Impurity Importance")
    ax.legend(handles=patches, fontsize=9)

    # Permutation importance
    ax2 = axes[1]
    top_p = perm_df.head(top_n)
    ax2.barh(range(len(top_p)), top_p["Perm_Mean"],
             xerr=top_p["Perm_Std"],
             color=[tier_color(f) for f in top_p["Feature"]],
             edgecolor="white", alpha=0.88, capsize=3)
    ax2.set_yticks(range(len(top_p)))
    ax2.set_yticklabels(top_p["Feature"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Permutation Importance")
    ax2.set_title("Permutation Importance (test set)")
    ax2.legend(handles=patches, fontsize=9)

    plt.tight_layout()
    p = EDA_DIR / "feature_importance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: feature_importance.png")


def plot_tier3_validation(interaction_df):
    """
    Two-panel chart:
      Left  — Top 6 ML-discovered Tier 3 pairs by H-statistic (bar chart)
      Right — All pairs with H >= 0.05 ranked (horizontal lollipop)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Tier 3 — ML-Discovered Interactions (Friedman H-Statistic)",
        fontsize=13, fontweight="bold",
    )

    # ── Left: top-6 Tier 3 pairs ─────────────────────────────────────────────
    ax = axes[0]
    n  = len(interaction_df)
    x  = np.arange(n)

    color_map = {"CONFIRMED": COLORS["tier1"], "PARTIAL": COLORS["tier3"]}
    colors    = [color_map.get(s, "#aaaaaa") for s in interaction_df["Status"]]

    bars = ax.bar(x, interaction_df["H_Statistic"], color=colors, alpha=0.88,
                  edgecolor="white", width=0.60)

    ax.axhline(0.30, color=COLORS["tier1"], linestyle="--", linewidth=1.3,
               label="CONFIRMED threshold (H=0.30)")
    ax.axhline(0.10, color=COLORS["tier3"], linestyle=":", linewidth=1.2,
               label="PARTIAL threshold (H=0.10)")

    ax.set_xticks(x)
    short_labels = [r["Interaction"].replace(" × ", "\n×\n")
                    for _, r in interaction_df.iterrows()]
    ax.set_xticklabels(short_labels, fontsize=8, rotation=0)
    ax.set_ylabel("H-Statistic")
    ax.set_title("Top 6 ML-Discovered Tier 3 Interactions")
    ax.set_ylim(0, max(0.8, interaction_df["H_Statistic"].max() * 1.30))

    for bar, (_, row) in zip(bars, interaction_df.iterrows()):
        icon = "✓" if row["Status"] == "CONFIRMED" else "~"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{icon} H={row['H_Statistic']:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    confirmed_p = mpatches.Patch(color=COLORS["tier1"], alpha=0.88, label="CONFIRMED")
    partial_p   = mpatches.Patch(color=COLORS["tier3"], alpha=0.88, label="PARTIAL")
    ax.legend(handles=[confirmed_p, partial_p], fontsize=8, loc="upper right")

    # ── Right: feature-importance context for the top-6 pairs ────────────────
    ax2 = axes[1]
    feats1 = interaction_df["Feat1"].tolist()
    feats2 = interaction_df["Feat2"].tolist()
    imps1  = interaction_df["Feat1_Imp"].tolist()
    imps2  = interaction_df["Feat2_Imp"].tolist()
    labels = [r["Interaction"].split(" × ") for _, r in interaction_df.iterrows()]

    y_pos  = np.arange(n)
    ax2.barh(y_pos - 0.20, imps1, height=0.35,
             color=COLORS["tier1"], alpha=0.80, label="Feature 1 importance")
    ax2.barh(y_pos + 0.20, imps2, height=0.35,
             color=COLORS["tier3"], alpha=0.80, label="Feature 2 importance")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{a[0]}\n× {a[1]}" for a in labels], fontsize=8)
    ax2.set_xlabel("GBM Feature Importance")
    ax2.set_title("Individual Feature Importance of Each Pair")
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

    plt.tight_layout()
    p = EDA_DIR / "tier3_validation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: tier3_validation.png")


def plot_model_comparison(tw_metrics, gbm_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Tweedie GLM vs GBM — Performance Comparison", fontsize=13, fontweight="bold")

    models  = ["Tweedie\n(GLM)", "GBM"]
    r2s     = [tw_metrics["test_r2"],   gbm_metrics["test_r2"]]
    rmses   = [tw_metrics["test_rmse"], gbm_metrics["test_rmse"]]
    colors  = [COLORS["tier2"], COLORS["tier1"]]

    for ax, vals, ylabel, title, fmt in [
        (axes[0], r2s,   "Test R2",   "R2 Comparison",   ".4f"),
        (axes[1], rmses, "Test RMSE ($)", "RMSE Comparison", ",.0f"),
    ]:
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5, alpha=0.9)
        for bar, v in zip(bars, vals):
            label = f"${v:{fmt}}" if "RMSE" in ylabel else f"R2={v:{fmt}}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    label, ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout()
    p = EDA_DIR / "model_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def save_models(gbm, tweedie, scaler, encoders, imp_df, interaction_df,
                gbm_metrics, tw_metrics):
    artifacts = {
        "gbm":             gbm,
        "tweedie":         tweedie,
        "scaler":          scaler,
        "encoders":        encoders,
        "feature_names":   ALL_FEATURES,
        "tier1_features":  TIER1_FEATURES,
        "tier2_features":  TIER2_FEATURES,
        "env_features":    ENV_FEATURES,
        "imp_df":          imp_df,
        "interaction_df":  interaction_df,
        "gbm_metrics":     gbm_metrics,
        "tw_metrics":      tw_metrics,
    }
    path = MODEL_DIR / "trained_models.pkl"
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Saved: trained_models.pkl  ({path.stat().st_size / 1024:.0f} KB)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  HOMEOWNERS RISK MODEL — TRAINING PIPELINE")
    print("=" * 55)

    X_train, X_test, y_train, y_test, encoders, df = load_and_split()

    tweedie, scaler, tw_metrics, tw_preds  = train_tweedie(X_train, X_test, y_train, y_test)
    gbm, gbm_metrics, gbm_preds            = train_gbm(X_train, X_test, y_train, y_test)
    imp_df, perm_df                        = compute_feature_importance(gbm, X_test, y_test)
    interaction_df, _full_scan             = discover_tier3_interactions(gbm, X_train, imp_df)

    print("\n  Generating charts ...")
    plot_actual_vs_predicted(y_test, tw_preds, gbm_preds)
    plot_feature_importance(imp_df, perm_df)
    plot_tier3_validation(interaction_df)
    plot_model_comparison(tw_metrics, gbm_metrics)

    save_models(gbm, tweedie, scaler, encoders, imp_df, interaction_df,
                gbm_metrics, tw_metrics)

    print("\n" + "=" * 55)
    print("  COMPLETE")
    print("=" * 55)
    print(f"  Tweedie  Test R2: {tw_metrics['test_r2']:.4f}  RMSE: ${tw_metrics['test_rmse']:,.0f}")
    print(f"  GBM      Test R2: {gbm_metrics['test_r2']:.4f}  RMSE: ${gbm_metrics['test_rmse']:,.0f}")
    print(f"  GBM R2 lift over Tweedie: +{gbm_metrics['test_r2'] - tw_metrics['test_r2']:.4f}")

    # Variance split by tier (framework targets: T1~60%, T2~+20%, ENV~+12%)
    t1_imp  = imp_df[imp_df["Feature"].isin(TIER1_FEATURES)]["Importance"].sum()
    t2_imp  = imp_df[imp_df["Feature"].isin(TIER2_FEATURES)]["Importance"].sum()
    env_imp = imp_df[imp_df["Feature"].isin(ENV_FEATURES)]["Importance"].sum()
    print(f"\n  Variance split (GBM feature importance):")
    print(f"    Tier 1 (target ~60%): {t1_imp*100:.1f}%")
    print(f"    Tier 2 (target ~20%): {t2_imp*100:.1f}%")
    print(f"    ENV/T3 (target ~12%): {env_imp*100:.1f}%")
    print(f"\n  Tier 3 validation summary:")
    for _, row in interaction_df.iterrows():
        h_val = row.get("H_Statistic", "?")
        print(f"    [{row['Status']:<9}] {row['Interaction']:<45} H={h_val:.4f}")
    print(f"\n  Charts saved to: eda_outputs/")
    print(f"  Model saved to:  trained_models.pkl")
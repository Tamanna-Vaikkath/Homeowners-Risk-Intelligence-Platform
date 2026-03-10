"""
scoring/scoring_engine.py  —  Master Scoring Engine
====================================================
Final Score = clip( (T1 + T2) × T3 , 0, 100 )

PROTOTYPE NOTICE
  Trained on SYNTHETIC data. Real-data expectations:
    GBM R²    ≈ 0.35–0.55    (synthetic ≈ 0.32)
    Tweedie R² ≈ 0.20–0.35   (synthetic ≈ 0.25)
  Replace homeowners_risk_dataset.csv with real CLUE data and re-run
  python train_model.py to recalibrate all coefficients.

Framework compliance (synthetic data)
  ✓ Variance split  T1=60%  T2=26%  ENV=14%  (target 60/20/12 — T2 slightly high)
  ✓ fire_sprinklers ordinal-encoded  none=0 · partial=1 · full=2
  ✓ Tier 3 interactions ML-discovered via Friedman H-statistic exhaustive scan
      3 CONFIRMED (H≥0.30) + 3 PARTIAL (H≥0.10)
  ✓ Realistic R² via lognormal noise injection in generate_dataset.py

Compound interaction encoding — must match tier3.py INTERACTION_ORDER exactly:
  slope_burn         slope_pct>15 AND burn_history=1       H=0.5194  CONFIRMED
  flood_foundation   flood_zone=1 AND stone_dirt            H=0.4511  CONFIRMED
  roof_age_wildfire  roof_age>20 AND wildfire_zone=1        H=0.4254  CONFIRMED
  trampoline_hail    trampoline=1 AND hail_zone=1           H=0.1754  PARTIAL
  claims_wildfire    prior_claims≥1 AND wildfire_zone=1     H=0.1652  PARTIAL
  firestation_slope  fire_station≥10mi AND slope_pct>15     H=0.1393  PARTIAL
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from dataclasses import dataclass, field
from tier1 import compute_tier1_score, get_tier1_flags, Tier1Breakdown
from tier2 import compute_tier2_adjustment, get_tier2_flags, Tier2Breakdown
from tier3 import score_tier3, Tier3Breakdown, INTERACTION_LABELS, H_STATISTICS

DECISION_COLORS = {
    "Preferred": "#2ecc71",
    "Standard":  "#3498db",
    "Rated":     "#f39c12",
    "Decline":   "#e74c3c",
}
DECISION_DESCRIPTIONS = {
    "Preferred": "Approve at base rate — excellent risk profile.",
    "Standard":  "Approve at standard rate (+/-15%) — normal risk.",
    "Rated":     "Rate up 15-50% or refer to senior underwriter.",
    "Decline":   "Decline application or refer to specialty underwriting.",
}


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED LOSS MODEL
# ─────────────────────────────────────────────────────────────────────────────

def compute_expected_loss(
    prior_claims_5yr, water_loss_recency, roof_age, roof_vulnerability,
    dwelling_construction, coverage_a_amount, fire_station_distance,
    home_age, square_footage, iso_class,
    insurance_lapses, pet_ownership, swimming_pool, trampoline,
    wood_burning_stove, home_business, recent_renovations, monitored_alarm,
    fire_sprinklers, gated_community,
    wildfire_zone, canopy_density, flood_zone, foundation_type,
    slope_pct, burn_history, hail_zone,
    final_score,
):
    """
    Expected loss = claim_frequency × claim_severity.
    Variance allocation mirrors GBM-measured feature importance.
    """
    rv = str(roof_vulnerability).lower()
    dc = str(dwelling_construction).lower()
    sp = str(swimming_pool).lower()
    ws = str(wood_burning_stove).lower()
    hb = str(home_business).lower()
    fs = str(fire_sprinklers).lower()
    cd = str(canopy_density).lower()
    ft = str(foundation_type).lower()

    # ── FREQUENCY  (base ≈ 8% annual) ────────────────────────────────────────
    freq = 0.08

    # Tier 1
    freq += prior_claims_5yr   * 0.050
    freq += water_loss_recency * 0.065
    freq += 0.055 if roof_age > 20 else 0.022 if roof_age > 15 else 0
    freq += 0.025 if iso_class >= 8 else 0.012 if iso_class >= 5 else 0
    freq += 0.022 if fire_station_distance >= 10 else 0.010 if fire_station_distance >= 5 else 0
    freq += 0.020 if home_age > 80 else 0.010 if home_age > 40 else 0
    freq += 0.040 if rv == "asphalt" else 0.015 if rv == "tile" else 0
    freq += 0.035 if dc == "wood_frame" else 0

    # Tier 2
    freq += insurance_lapses * 0.032
    freq += 0.035 if pet_ownership == "dangerous_breed" else 0.012 if pet_ownership == "standard_pets" else 0
    freq += 0.030 if sp == "in_ground" else 0.016 if sp == "above_ground" else 0
    freq += trampoline * 0.032
    freq += 0.030 if ws == "wood_burning" else 0.010 if ws == "gas_fireplace" else 0
    freq += 0.030 if hb == "active_business" else 0.010 if hb == "home_office" else 0
    freq -= recent_renovations * 0.035
    freq -= monitored_alarm    * 0.045
    freq -= 0.040 if fs == "full" else 0.020 if fs == "partial" else 0
    freq -= gated_community * 0.012

    # ENV — small additive baselines; interactions carry the non-linear lift
    freq += wildfire_zone * 0.007
    freq += 0.002 if cd == "dense" else 0.001 if cd == "moderate" else 0
    freq += flood_zone    * 0.005
    freq += 0.001 if slope_pct > 25 else 0
    freq += burn_history  * 0.002
    freq += hail_zone     * 0.004

    # Compound interactions — mirrors tier3.py INTERACTION_ORDER exactly
    if slope_pct > 15 and burn_history == 1:            # slope_burn        H=0.5194
        freq *= 1.18
    if flood_zone == 1 and ft == "stone_dirt":           # flood_foundation  H=0.4511
        freq *= 1.16
    if roof_age > 20 and wildfire_zone == 1:             # roof_age_wildfire H=0.4254
        freq *= 1.14
    if int(trampoline) == 1 and hail_zone == 1:          # trampoline_hail   H=0.1754
        freq *= 1.06
    if int(prior_claims_5yr) >= 1 and wildfire_zone == 1:# claims_wildfire   H=0.1652
        freq *= 1.06
    if fire_station_distance >= 10 and slope_pct > 15:   # firestation_slope H=0.1393
        freq *= 1.05

    freq = max(0.01, min(0.95, freq))

    # ── SEVERITY  (base ≈ $18 000) ────────────────────────────────────────────
    sev = 18_000.0

    # Tier 1
    sev += coverage_a_amount * 0.010
    sev += home_age          * 180
    sev += square_footage    * 2.0
    sev += 5_000 if rv == "asphalt" else 2_000 if rv == "tile" else -3_500 if rv == "metal" else 0
    sev += 8_000 if dc == "wood_frame" else -6_000 if dc == "masonry" else -12_000 if dc == "superior" else 0

    # Tier 2
    sev += 12_000 if sp == "in_ground" else 3_500 if sp == "above_ground" else 0
    sev += 10_000 if ws == "wood_burning" else 0
    sev += 11_000 if hb == "active_business" else 2_500 if hb == "home_office" else 0
    sev -= 14_000 if fs == "full" else 7_000 if fs == "partial" else 0
    sev -= recent_renovations * 6_000

    # ENV — small baselines
    sev += wildfire_zone * 1_500
    sev += flood_zone    * 1_200
    sev += burn_history  * 500
    sev += hail_zone     * 800
    sev += 250 if slope_pct > 25 else 120 if slope_pct > 15 else 0
    sev += 300 if cd == "dense" else 0
    sev += 0.002 * coverage_a_amount if ft == "stone_dirt" else 0

    # Compound severity effects — same order
    if slope_pct > 15 and burn_history == 1:
        sev *= 1.22
    if flood_zone == 1 and ft == "stone_dirt":
        sev *= 1.24
    if roof_age > 20 and wildfire_zone == 1:
        sev *= 1.18
    if int(trampoline) == 1 and hail_zone == 1:
        sev *= 1.08
    if int(prior_claims_5yr) >= 1 and wildfire_zone == 1:
        sev *= 1.08
    if fire_station_distance >= 10 and slope_pct > 15:
        sev *= 1.12

    sev = max(5_000, min(500_000, sev))

    return {
        "claim_frequency": round(freq, 4),
        "claim_severity":  round(sev, 2),
        "expected_loss":   round(freq * sev, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskScoreResult:
    tier1:                Tier1Breakdown
    tier2:                Tier2Breakdown
    tier3:                Tier3Breakdown
    tier1_score:          float
    tier2_adjustment:     float
    tier3_multiplier:     float
    base_score:           float
    final_score:          float
    decision:             str
    decision_color:       str
    decision_description: str
    claim_frequency:      float
    claim_severity:       float
    expected_loss:        float
    all_flags:            list = field(default_factory=list)

    def score_components_df(self):
        t1 = self.tier1.to_dict()
        t2 = {k: v for k, v in self.tier2.to_dict().items() if v != 0}
        rows = (
            [{"Variable": k, "Points": v, "Tier": "Tier 1"} for k, v in t1.items()] +
            [{"Variable": k, "Points": v, "Tier": "Tier 2"} for k, v in t2.items()]
        )
        return pd.DataFrame(rows)

    def to_flat_dict(self):
        return {
            "tier1_score": self.tier1_score,
            "tier2_adjustment": self.tier2_adjustment,
            "tier3_multiplier": self.tier3_multiplier,
            "base_score": self.base_score,
            "final_score": self.final_score,
            "decision": self.decision,
            "claim_frequency": self.claim_frequency,
            "claim_severity": self.claim_severity,
            "expected_loss": self.expected_loss,
            **self.tier1.to_dict(),
            **self.tier2.to_dict(),
            **self.tier3.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_homeowner(
    roof_age, roof_vulnerability, dwelling_construction, water_loss_recency,
    prior_claims_5yr, coverage_a_amount, fire_station_distance, home_age,
    square_footage, iso_class,
    insurance_lapses, pet_ownership, swimming_pool, trampoline,
    wood_burning_stove, home_business, recent_renovations, monitored_alarm,
    fire_sprinklers, gated_community,
    wildfire_zone, canopy_density, flood_zone, foundation_type,
    slope_pct, burn_history, hail_zone,
) -> RiskScoreResult:
    """Master scoring function.  Formula: Final = clip((T1 + T2) × T3, 0, 100)"""

    t1 = compute_tier1_score(
        roof_age=roof_age, roof_vulnerability=roof_vulnerability,
        dwelling_construction=dwelling_construction, water_loss_recency=water_loss_recency,
        prior_claims_5yr=prior_claims_5yr, coverage_a_amount=coverage_a_amount,
        fire_station_distance=fire_station_distance, home_age=home_age,
        square_footage=square_footage, iso_class=iso_class,
    )
    t2 = compute_tier2_adjustment(
        insurance_lapses=insurance_lapses, pet_ownership=pet_ownership,
        swimming_pool=swimming_pool, trampoline=trampoline,
        wood_burning_stove=wood_burning_stove, home_business=home_business,
        recent_renovations=recent_renovations, monitored_alarm=monitored_alarm,
        fire_sprinklers=fire_sprinklers, gated_community=gated_community,
    )
    t3 = score_tier3(
        roof_age=roof_age, roof_vulnerability=roof_vulnerability,
        water_loss_recency=water_loss_recency,
        prior_claims_5yr=prior_claims_5yr,
        wildfire_zone=wildfire_zone, hail_zone=hail_zone,
        canopy_density=canopy_density, flood_zone=flood_zone,
        foundation_type=foundation_type, slope_pct=slope_pct,
        burn_history=burn_history,
        fire_station_distance=fire_station_distance,
        trampoline=trampoline,
        home_age=home_age, coverage_a_amount=coverage_a_amount,
    )

    base_score  = t1.total + t2.total
    final_score = round(max(0.0, min(100.0, base_score * t3.multiplier)), 2)

    if   final_score <= 30: decision = "Preferred"
    elif final_score <= 60: decision = "Standard"
    elif final_score <= 80: decision = "Rated"
    else:                   decision = "Decline"

    loss = compute_expected_loss(
        prior_claims_5yr=prior_claims_5yr, water_loss_recency=water_loss_recency,
        roof_age=roof_age, roof_vulnerability=roof_vulnerability,
        dwelling_construction=dwelling_construction, coverage_a_amount=coverage_a_amount,
        fire_station_distance=fire_station_distance, home_age=home_age,
        square_footage=square_footage, iso_class=iso_class,
        insurance_lapses=insurance_lapses, pet_ownership=pet_ownership,
        swimming_pool=swimming_pool, trampoline=trampoline,
        wood_burning_stove=wood_burning_stove, home_business=home_business,
        recent_renovations=recent_renovations, monitored_alarm=monitored_alarm,
        fire_sprinklers=fire_sprinklers, gated_community=gated_community,
        wildfire_zone=wildfire_zone, canopy_density=canopy_density,
        flood_zone=flood_zone, foundation_type=foundation_type,
        slope_pct=slope_pct, burn_history=burn_history, hail_zone=hail_zone,
        final_score=final_score,
    )

    all_flags = (
        get_tier1_flags(
            roof_age=roof_age, roof_vulnerability=roof_vulnerability,
            dwelling_construction=dwelling_construction,
            water_loss_recency=water_loss_recency, prior_claims_5yr=prior_claims_5yr,
            home_age=home_age, fire_station_distance=fire_station_distance,
        ) +
        get_tier2_flags(t2, home_business) +
        [
            f"Tier 3 — {INTERACTION_LABELS[k]} (H={H_STATISTICS[k]:.3f})"
            for k in t3.triggered_keys
        ]
    )

    return RiskScoreResult(
        tier1=t1, tier2=t2, tier3=t3,
        tier1_score=t1.total, tier2_adjustment=t2.total,
        tier3_multiplier=t3.multiplier,
        base_score=round(base_score, 2), final_score=final_score,
        decision=decision,
        decision_color=DECISION_COLORS[decision],
        decision_description=DECISION_DESCRIPTIONS[decision],
        claim_frequency=loss["claim_frequency"],
        claim_severity=loss["claim_severity"],
        expected_loss=loss["expected_loss"],
        all_flags=all_flags,
    )


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Batch score every row in a DataFrame."""
    results = []
    for _, row in df.iterrows():
        r = score_homeowner(
            roof_age=row["roof_age"], roof_vulnerability=row["roof_vulnerability"],
            dwelling_construction=row["dwelling_construction"],
            water_loss_recency=row["water_loss_recency"],
            prior_claims_5yr=row["prior_claims_5yr"],
            coverage_a_amount=row["coverage_a_amount"],
            fire_station_distance=row["fire_station_distance"],
            home_age=row["home_age"], square_footage=row["square_footage"],
            iso_class=row["iso_class"], insurance_lapses=row["insurance_lapses"],
            pet_ownership=row["pet_ownership"], swimming_pool=row["swimming_pool"],
            trampoline=row["trampoline"], wood_burning_stove=row["wood_burning_stove"],
            home_business=row["home_business"], recent_renovations=row["recent_renovations"],
            monitored_alarm=row["monitored_alarm"], fire_sprinklers=row["fire_sprinklers"],
            gated_community=row["gated_community"], wildfire_zone=row["wildfire_zone"],
            canopy_density=row["canopy_density"], flood_zone=row["flood_zone"],
            foundation_type=row["foundation_type"], slope_pct=row["slope_pct"],
            burn_history=row["burn_history"], hail_zone=row["hail_zone"],
        )
        results.append(r.to_flat_dict())
    return pd.DataFrame(results)


if __name__ == "__main__":
    r = score_homeowner(
        roof_age=25, roof_vulnerability="asphalt", dwelling_construction="wood_frame",
        water_loss_recency=1, prior_claims_5yr=2, coverage_a_amount=350_000,
        fire_station_distance=14.0, home_age=45, square_footage=2400, iso_class=6,
        insurance_lapses=1, pet_ownership="standard_pets", swimming_pool="in_ground",
        trampoline=1, wood_burning_stove="wood_burning", home_business="none",
        recent_renovations=0, monitored_alarm=0, fire_sprinklers="none", gated_community=0,
        wildfire_zone=1, canopy_density="dense", flood_zone=0,
        foundation_type="poured_concrete", slope_pct=22.0, burn_history=1, hail_zone=1,
    )
    print(f"Tier 1:        {r.tier1_score:.2f} pts")
    print(f"Tier 2:        {r.tier2_adjustment:+.2f} pts")
    print(f"Base (T1+T2):  {r.base_score:.2f} pts")
    print(f"Tier 3 mult:   ×{r.tier3_multiplier:.4f}  triggered={r.tier3.triggered_keys}")
    print(f"Final Score:   {r.final_score:.2f} / 100")
    print(f"Decision:      {r.decision}")
    print(f"Expected Loss: ${r.expected_loss:,.2f}")
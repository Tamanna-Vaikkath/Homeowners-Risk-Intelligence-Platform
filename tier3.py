"""
scoring/tier3.py  —  Tier 3 Amplification Layer
================================================
PROTOTYPE NOTICE
  All H-statistics and multipliers were discovered from SYNTHETIC data.
  On real carrier loss data, re-run  python train_model.py  and update
  this file to match what the GBM actually finds.

ML Discovery — exhaustive Friedman H-statistic scan, 161 ENV-involving pairs
  (train_model.py Step 4, run on 10 000-row synthetic dataset)

  Key    Feature pair                        H       Status
  ─────────────────────────────────────────────────────────
  slope_burn          Slope × Burn History   0.5194  CONFIRMED
  flood_foundation    Flood × Foundation     0.4511  CONFIRMED
  roof_age_wildfire   Roof Age × Wildfire    0.4254  CONFIRMED
  trampoline_hail     Trampoline × Hail      0.1754  PARTIAL
  claims_wildfire     Prior Claims × Wildfire 0.1652  PARTIAL
  firestation_slope   Fire Station × Slope   0.1393  PARTIAL (actuarially motivated)

Thresholds:  CONFIRMED H≥0.30  |  PARTIAL H≥0.10
Multipliers: proportional to H-statistic, capped at ×1.08 (CONFIRMED) / ×1.04 (PARTIAL)

Note on firestation_slope
  fire_station_distance × slope_pct was the 8th highest-H pair (H=0.1393) in the
  scan and is included because it has a direct actuarial mechanism (remote + steep
  terrain = slow response + higher severity). Trampoline pairs are included because
  the GBM confirmed them, but they are flagged PARTIAL pending real-data validation.
"""
from dataclasses import dataclass, field


# ── Multipliers ───────────────────────────────────────────────────────────────
MULTIPLIERS = {
    "slope_burn":       1.08,   # H=0.5194  CONFIRMED  — post-fire erosion / mudslide
    "flood_foundation": 1.07,   # H=0.4511  CONFIRMED  — permeable foundation + flood zone
    "roof_age_wildfire":1.06,   # H=0.4254  CONFIRMED  — aged roof loses fire resistance
    "trampoline_hail":  1.03,   # H=0.1754  PARTIAL    — liability compound in hail zone
    "claims_wildfire":  1.03,   # H=0.1652  PARTIAL    — repeat loss in wildfire corridor
    "firestation_slope":1.03,   # H=0.1393  PARTIAL    — remote steep terrain, slow response
}

INTERACTION_LABELS = {
    "slope_burn":       "Steep Slope × Burn History",
    "flood_foundation": "Flood Zone × Stone/Dirt Foundation",
    "roof_age_wildfire":"Aged Roof × Wildfire Zone",
    "trampoline_hail":  "Trampoline × Hail Zone",
    "claims_wildfire":  "Prior Claims × Wildfire Zone",
    "firestation_slope":"Remote Fire Station × Steep Slope",
}

H_STATISTICS = {
    "slope_burn":       0.5194,
    "flood_foundation": 0.4511,
    "roof_age_wildfire":0.4254,
    "trampoline_hail":  0.1754,
    "claims_wildfire":  0.1652,
    "firestation_slope":0.1393,
}

ML_STATUS = {
    "slope_burn":       "CONFIRMED",
    "flood_foundation": "CONFIRMED",
    "roof_age_wildfire":"CONFIRMED",
    "trampoline_hail":  "PARTIAL",
    "claims_wildfire":  "PARTIAL",
    "firestation_slope":"PARTIAL",
}

# Ordered by descending H-statistic
INTERACTION_ORDER = [
    "slope_burn", "flood_foundation", "roof_age_wildfire",
    "trampoline_hail", "claims_wildfire", "firestation_slope",
]


@dataclass
class Tier3Breakdown:
    slope_burn_triggered:        bool
    flood_foundation_triggered:  bool
    roof_age_wildfire_triggered: bool
    trampoline_hail_triggered:   bool
    claims_wildfire_triggered:   bool
    firestation_slope_triggered: bool
    multiplier:                  float
    triggered_keys:              list = field(default_factory=list)

    def to_dict(self):
        return {
            "t3_slope_burn":        int(self.slope_burn_triggered),
            "t3_flood_foundation":  int(self.flood_foundation_triggered),
            "t3_roof_age_wildfire": int(self.roof_age_wildfire_triggered),
            "t3_trampoline_hail":   int(self.trampoline_hail_triggered),
            "t3_claims_wildfire":   int(self.claims_wildfire_triggered),
            "t3_firestation_slope": int(self.firestation_slope_triggered),
            "t3_multiplier":        self.multiplier,
        }


def score_tier3(
    roof_age, roof_vulnerability, water_loss_recency=0,
    prior_claims_5yr=0,
    wildfire_zone=0, hail_zone=0, canopy_density="sparse",
    flood_zone=0, foundation_type="concrete_slab",
    slope_pct=0.0, burn_history=0,
    fire_station_distance=3.0,
    trampoline=0,
    coverage_a_amount=999_999, home_age=0,
    **kwargs,
) -> Tier3Breakdown:
    """
    Apply ML-discovered multiplicative amplifiers.
    Trigger logic translates GBM-confirmed feature interactions into
    underwriting-interpretable boolean conditions.
    """
    rv        = str(roof_vulnerability).lower()
    soft_roof = rv in ("asphalt", "tile")
    fd_type   = str(foundation_type).lower()

    triggered = {
        "slope_burn":        slope_pct > 15 and burn_history == 1,
        "flood_foundation":  flood_zone == 1 and fd_type == "stone_dirt",
        "roof_age_wildfire": roof_age > 20 and wildfire_zone == 1,
        "trampoline_hail":   int(trampoline) == 1 and hail_zone == 1,
        "claims_wildfire":   int(prior_claims_5yr) >= 1 and wildfire_zone == 1,
        "firestation_slope": fire_station_distance >= 10 and slope_pct > 15,
    }

    multiplier     = 1.0
    triggered_keys = []
    for key in INTERACTION_ORDER:
        if triggered[key]:
            multiplier     *= MULTIPLIERS[key]
            triggered_keys.append(key)

    return Tier3Breakdown(
        slope_burn_triggered        = triggered["slope_burn"],
        flood_foundation_triggered  = triggered["flood_foundation"],
        roof_age_wildfire_triggered = triggered["roof_age_wildfire"],
        trampoline_hail_triggered   = triggered["trampoline_hail"],
        claims_wildfire_triggered   = triggered["claims_wildfire"],
        firestation_slope_triggered = triggered["firestation_slope"],
        multiplier                  = round(multiplier, 4),
        triggered_keys              = triggered_keys,
    )
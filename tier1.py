"""
scoring/tier1.py — Tier 1 Foundation Layer
10 high-correlation variables (r >= 0.29). Max raw score = 100 pts.
"""
import pandas as pd
from dataclasses import dataclass


def score_roof_age(roof_age):
    if   roof_age >= 30: return 18.0
    elif roof_age >= 25: return 15.0
    elif roof_age >= 20: return 12.0
    elif roof_age >= 15: return  8.0
    elif roof_age >= 10: return  5.0
    else:                return  2.0

def score_roof_vulnerability(material):
    return {"asphalt": 17.0, "tile": 10.0, "slate": 6.0, "metal": 3.0}.get(str(material).lower().strip(), 10.0)

def score_dwelling_construction(construction):
    return {"wood_frame": 15.0, "brick_veneer": 9.0, "masonry": 5.0, "superior": 2.0}.get(str(construction).lower().strip(), 9.0)

def score_water_loss_recency(had_water_claim):
    return 13.0 if int(had_water_claim) == 1 else 0.0

def score_prior_claims(claim_count):
    if   claim_count >= 5: return 11.0
    elif claim_count >= 3: return  8.0
    elif claim_count >= 1: return  4.0
    else:                  return  0.0

def score_coverage_a(coverage_a):
    if   coverage_a >= 1_500_000: return 8.0
    elif coverage_a >= 800_000:   return 6.0
    elif coverage_a >= 300_000:   return 4.0
    else:                         return 2.0

def score_fire_station_distance(miles):
    if   miles >= 20: return 7.0
    elif miles >= 10: return 5.0
    elif miles >=  5: return 3.0
    else:             return 1.0

def score_home_age(home_age):
    if   home_age >= 100: return 5.0
    elif home_age >=  80: return 4.0
    elif home_age >=  30: return 2.0
    else:                 return 1.0

def score_square_footage(sqft):
    if   sqft >= 5000: return 4.0
    elif sqft >= 3000: return 3.0
    elif sqft >= 1500: return 2.0
    else:              return 1.0

def score_iso_class(iso_class):
    if   iso_class >= 8: return 2.0
    elif iso_class >= 4: return 1.5
    else:                return 0.5


@dataclass
class Tier1Breakdown:
    roof_age_pts:              float
    roof_vulnerability_pts:    float
    dwelling_construction_pts: float
    water_loss_recency_pts:    float
    prior_claims_pts:          float
    coverage_a_pts:            float
    fire_station_pts:          float
    home_age_pts:              float
    square_footage_pts:        float
    iso_class_pts:             float
    total:                     float

    def to_dict(self):
        return {
            "Roof Age":              self.roof_age_pts,
            "Roof Vulnerability":    self.roof_vulnerability_pts,
            "Dwelling Construction": self.dwelling_construction_pts,
            "Water Loss Recency":    self.water_loss_recency_pts,
            "Prior Claims (5yr)":    self.prior_claims_pts,
            "Coverage A Amount":     self.coverage_a_pts,
            "Fire Station Distance": self.fire_station_pts,
            "Home Age":              self.home_age_pts,
            "Square Footage":        self.square_footage_pts,
            "ISO Class":             self.iso_class_pts,
        }


def compute_tier1_score(roof_age, roof_vulnerability, dwelling_construction,
                        water_loss_recency, prior_claims_5yr, coverage_a_amount,
                        fire_station_distance, home_age, square_footage, iso_class):
    pts = [
        score_roof_age(roof_age),
        score_roof_vulnerability(roof_vulnerability),
        score_dwelling_construction(dwelling_construction),
        score_water_loss_recency(water_loss_recency),
        score_prior_claims(prior_claims_5yr),
        score_coverage_a(coverage_a_amount),
        score_fire_station_distance(fire_station_distance),
        score_home_age(home_age),
        score_square_footage(square_footage),
        score_iso_class(iso_class),
    ]
    return Tier1Breakdown(
        roof_age_pts=pts[0], roof_vulnerability_pts=pts[1],
        dwelling_construction_pts=pts[2], water_loss_recency_pts=pts[3],
        prior_claims_pts=pts[4], coverage_a_pts=pts[5],
        fire_station_pts=pts[6], home_age_pts=pts[7],
        square_footage_pts=pts[8], iso_class_pts=pts[9],
        total=round(sum(pts), 2),
    )


def compute_tier1_batch(df):
    score = pd.Series(0.0, index=df.index)
    score += df["roof_age"].apply(score_roof_age)
    score += df["roof_vulnerability"].apply(score_roof_vulnerability)
    score += df["dwelling_construction"].apply(score_dwelling_construction)
    score += df["water_loss_recency"].apply(score_water_loss_recency)
    score += df["prior_claims_5yr"].apply(score_prior_claims)
    score += df["coverage_a_amount"].apply(score_coverage_a)
    score += df["fire_station_distance"].apply(score_fire_station_distance)
    score += df["home_age"].apply(score_home_age)
    score += df["square_footage"].apply(score_square_footage)
    score += df["iso_class"].apply(score_iso_class)
    return score.clip(0, 100).round(2)


def get_tier1_flags(roof_age, roof_vulnerability, dwelling_construction,
                    water_loss_recency, prior_claims_5yr, home_age, fire_station_distance):
    flags = []
    if roof_age >= 25:
        flags.append(f"WARNING: Roof Age {roof_age}yrs - mandatory inspection required")
    if roof_vulnerability == "asphalt" and roof_age >= 20:
        flags.append("WARNING: Aged asphalt shingle - high weather vulnerability")
    if dwelling_construction == "wood_frame":
        flags.append("WARNING: Wood frame construction - elevated fire/wind risk")
    if water_loss_recency == 1:
        flags.append("HIGH RISK: Prior water claim - mandatory property inspection")
    if prior_claims_5yr >= 5:
        flags.append("HIGH RISK: 5+ claims in 5 years - refer to specialty underwriting")
    elif prior_claims_5yr >= 3:
        flags.append("WARNING: 3-4 claims in 5 years - rate adjustment required")
    if home_age >= 80:
        flags.append(f"WARNING: Home Age {home_age}yrs - comprehensive inspection mandatory")
    if fire_station_distance >= 20:
        flags.append(f"HIGH RISK: Fire station {fire_station_distance}mi - decline or refer")
    return flags
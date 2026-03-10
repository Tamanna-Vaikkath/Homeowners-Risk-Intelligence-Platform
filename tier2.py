"""
scoring/tier2.py — Tier 2 Behavioral Refinement Layer
10 moderate-correlation variables. Returns additive adjustment (can be negative).
"""
import pandas as pd
from dataclasses import dataclass


def adjust_insurance_lapses(lapses):
    if   lapses >= 2: return  9.0
    elif lapses == 1: return  5.0
    else:             return  0.0

def adjust_pet_ownership(pet_type):
    return {"dangerous_breed": 8.0, "standard_pets": 3.0, "none": 0.0}.get(str(pet_type).lower().strip(), 3.0)

def adjust_swimming_pool(pool_type):
    return {"in_ground": 6.0, "above_ground": 3.0, "none": 0.0}.get(str(pool_type).lower().strip(), 0.0)

def adjust_trampoline(has_trampoline):
    return 5.0 if int(has_trampoline) == 1 else 0.0

def adjust_wood_burning_stove(stove_type):
    return {"wood_burning": 4.0, "gas_fireplace": 1.0, "none": 0.0}.get(str(stove_type).lower().strip(), 0.0)

def adjust_home_business(business_type):
    return {"active_business": 4.0, "home_office": 1.0, "none": 0.0}.get(str(business_type).lower().strip(), 0.0)

def adjust_recent_renovations(has_renovations):
    return -6.0 if int(has_renovations) == 1 else 0.0

def adjust_monitored_alarm(has_alarm):
    return -5.0 if int(has_alarm) == 1 else 0.0

def adjust_fire_sprinklers(sprinkler_type):
    return {"full": -7.0, "partial": -4.0, "none": 0.0}.get(str(sprinkler_type).lower().strip(), 0.0)

def adjust_gated_community(is_gated):
    return -2.0 if int(is_gated) == 1 else 0.0


@dataclass
class Tier2Breakdown:
    insurance_lapses_adj:   float
    pet_ownership_adj:      float
    swimming_pool_adj:      float
    trampoline_adj:         float
    wood_burning_stove_adj: float
    home_business_adj:      float
    recent_renovations_adj: float
    monitored_alarm_adj:    float
    fire_sprinklers_adj:    float
    gated_community_adj:    float
    total:                  float

    def to_dict(self):
        return {
            "Insurance Lapses":   self.insurance_lapses_adj,
            "Pet Ownership":      self.pet_ownership_adj,
            "Swimming Pool":      self.swimming_pool_adj,
            "Trampoline":         self.trampoline_adj,
            "Wood-Burning Stove": self.wood_burning_stove_adj,
            "Home Business":      self.home_business_adj,
            "Recent Renovations": self.recent_renovations_adj,
            "Monitored Alarm":    self.monitored_alarm_adj,
            "Fire Sprinklers":    self.fire_sprinklers_adj,
            "Gated Community":    self.gated_community_adj,
        }

    @property
    def risk_increasing(self):
        return {k: v for k, v in self.to_dict().items() if v > 0}

    @property
    def risk_reducing(self):
        return {k: v for k, v in self.to_dict().items() if v < 0}


def compute_tier2_adjustment(insurance_lapses, pet_ownership, swimming_pool,
                             trampoline, wood_burning_stove, home_business,
                             recent_renovations, monitored_alarm, fire_sprinklers,
                             gated_community):
    adjs = [
        adjust_insurance_lapses(insurance_lapses),
        adjust_pet_ownership(pet_ownership),
        adjust_swimming_pool(swimming_pool),
        adjust_trampoline(trampoline),
        adjust_wood_burning_stove(wood_burning_stove),
        adjust_home_business(home_business),
        adjust_recent_renovations(recent_renovations),
        adjust_monitored_alarm(monitored_alarm),
        adjust_fire_sprinklers(fire_sprinklers),
        adjust_gated_community(gated_community),
    ]
    return Tier2Breakdown(
        insurance_lapses_adj=adjs[0], pet_ownership_adj=adjs[1],
        swimming_pool_adj=adjs[2], trampoline_adj=adjs[3],
        wood_burning_stove_adj=adjs[4], home_business_adj=adjs[5],
        recent_renovations_adj=adjs[6], monitored_alarm_adj=adjs[7],
        fire_sprinklers_adj=adjs[8], gated_community_adj=adjs[9],
        total=round(sum(adjs), 2),
    )


def compute_tier2_batch(df):
    adj = pd.Series(0.0, index=df.index)
    adj += df["insurance_lapses"].apply(adjust_insurance_lapses)
    adj += df["pet_ownership"].apply(adjust_pet_ownership)
    adj += df["swimming_pool"].apply(adjust_swimming_pool)
    adj += df["trampoline"].apply(adjust_trampoline)
    adj += df["wood_burning_stove"].apply(adjust_wood_burning_stove)
    adj += df["home_business"].apply(adjust_home_business)
    adj += df["recent_renovations"].apply(adjust_recent_renovations)
    adj += df["monitored_alarm"].apply(adjust_monitored_alarm)
    adj += df["fire_sprinklers"].apply(adjust_fire_sprinklers)
    adj += df["gated_community"].apply(adjust_gated_community)
    return adj.round(2)


def get_tier2_flags(breakdown, home_business):
    flags = []
    if breakdown.insurance_lapses_adj > 0:
        flags.append("WARNING: Prior insurance lapse(s) - possible maintenance neglect")
    if breakdown.pet_ownership_adj >= 8:
        flags.append("HIGH RISK: Dangerous breed on property - high liability exposure")
    elif breakdown.pet_ownership_adj > 0:
        flags.append("INFO: Pet ownership - liability noted")
    if breakdown.swimming_pool_adj > 0:
        flags.append("WARNING: Pool on premises - drowning liability surcharge applied")
    if breakdown.trampoline_adj > 0:
        flags.append("WARNING: Trampoline present - injury liability surcharge applied")
    if breakdown.wood_burning_stove_adj >= 4:
        flags.append("WARNING: Wood-burning stove - fire risk, maintenance check required")
    if home_business == "active_business":
        flags.append("HIGH RISK: Active home business - refer to commercial underwriter")
    if breakdown.recent_renovations_adj < 0:
        flags.append("GOOD: Recent renovations - rate discount applied")
    if breakdown.monitored_alarm_adj < 0:
        flags.append("GOOD: Monitored alarm system - rate discount applied")
    if breakdown.fire_sprinklers_adj < 0:
        flags.append("GOOD: Fire sprinkler system - fire severity discount applied")
    return flags
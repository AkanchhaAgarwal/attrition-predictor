"""
Synthetic Contact Centre Attrition Dataset Generator
====================================================
Based on the white paper: "Predicting Voluntary Attrition in Contact Centres"
(WFM Simplified, 2026)

Design principles from the paper:
  1. Three layers: static profile / behavioural trends / environmental context
  2. Behaviour is driven by a LATENT DISENGAGEMENT factor, so behavioural
     KPIs are naturally correlated with each other AND with the outcome
     (just like real life -- and it creates the multicollinearity the
     paper warns about, which tree models handle and logistic won't).
  3. Target = voluntary exit within 90 days of scoring date (~10% base rate)
  4. Trend features = 30-day window vs 90-day personal baseline
     (no notice-period leakage: features represent T-120 to T-30)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

# ----------------------------------------------------------------------
# LAYER 1: STATIC PROFILE
# ----------------------------------------------------------------------
age = np.clip(rng.normal(27, 5, N), 19, 48).round(0)

# tenure: heavily skewed toward low tenure (real contact centre shape)
tenure_months = np.clip(rng.exponential(16, N), 1, 84).round(0)

commute_minutes = np.clip(rng.gamma(4, 12, N), 10, 150).round(0)

hiring_source = rng.choice(["Referral", "Job Portal", "Walk-in"], N, p=[0.30, 0.55, 0.15])
shift_type    = rng.choice(["Day Fixed", "Night Fixed", "Rotational", "Split"], N, p=[0.30, 0.25, 0.35, 0.10])
queue_type    = rng.choice(["Voice-Support", "Voice-Collections", "Chat", "Email", "Escalations"], N,
                           p=[0.40, 0.15, 0.20, 0.15, 0.10])
education     = rng.choice(["Undergraduate", "Graduate", "Postgraduate"], N, p=[0.25, 0.60, 0.15])

dependents  = rng.choice([0, 1, 2, 3, 4], N, p=[0.40, 0.25, 0.20, 0.10, 0.05])
prior_jobs  = np.clip(rng.poisson(1.5, N), 0, 7)

compa_ratio = np.clip(rng.normal(1.0, 0.10, N), 0.72, 1.30).round(3)
months_since_increment = np.clip(rng.exponential(10, N), 0, 36).round(0)
ijp_rejections = rng.choice([0, 1, 2, 3], N, p=[0.70, 0.18, 0.08, 0.04])

# ----------------------------------------------------------------------
# LAYER 3: ENVIRONMENTAL CONTEXT (drawn before behaviour -- it CAUSES it)
# ----------------------------------------------------------------------
tl_attrition_rate_6m = np.clip(rng.beta(2, 8, N), 0, 0.60).round(3)   # TL trailing team attrition
team_exits_60d       = rng.poisson(0.8 + 3.0 * tl_attrition_rate_6m)  # contagion follows bad TLs
batch_attrition_rate = np.clip(rng.beta(2.5, 6, N), 0, 0.70).round(3)
occupancy_90d        = np.clip(rng.normal(84, 6, N), 65, 98).round(1) # chronic load

# ----------------------------------------------------------------------
# LATENT DISENGAGEMENT SCORE  (the hidden truth the model must recover)
# ----------------------------------------------------------------------
z = lambda x: (x - x.mean()) / x.std()

# tenure risk is NON-LINEAR: spike at 0-3 months and at 11-14 months
tenure_risk = np.where(tenure_months <= 3, 1.0,
              np.where((tenure_months >= 11) & (tenure_months <= 14), 0.8,
              np.where(tenure_months <= 24, 0.3, 0.0)))

disengagement = (
      0.90 * tenure_risk
    + 0.45 * z(commute_minutes)
    + 0.40 * (hiring_source == "Job Portal") + 0.55 * (hiring_source == "Walk-in")
    + 0.35 * (shift_type == "Night Fixed") + 0.30 * (shift_type == "Rotational") + 0.50 * (shift_type == "Split")
    + 0.30 * (queue_type == "Voice-Collections") + 0.35 * (queue_type == "Escalations")
    + 0.35 * (education == "Postgraduate")            # overqualified for agent role
    - 0.30 * z(np.minimum(dependents, 3))             # dependents -> stability
    + 0.25 * z(prior_jobs)                            # job-hoppers keep hopping
    - 0.55 * z(compa_ratio)                           # underpaid vs band -> risk
    + 0.35 * z(months_since_increment)
    + 0.40 * ijp_rejections                           # blocked growth
    + 0.80 * z(tl_attrition_rate_6m)                  # people leave managers
    + 0.45 * z(team_exits_60d)                        # contagion
    + 0.30 * z(batch_attrition_rate)
    + 0.50 * np.maximum(0, (occupancy_90d - 90) / 4)  # burnout above 90% occ
    + rng.normal(0, 1.1, N)                           # everything we can't see
)
D = z(disengagement)  # standardised latent factor

# ----------------------------------------------------------------------
# LAYER 2: BEHAVIOURAL TRENDS (driven by D + noise) -- 30d vs 90d baseline
# ----------------------------------------------------------------------
noise = lambda s: rng.normal(0, s, N)

aht_delta_pct        = np.round( 6.0 * np.maximum(D, -0.5) + noise(6.0), 1)          # AHT drifting up
acw_pct_delta_pp     = np.round( 2.2 * np.maximum(D, 0) + noise(1.6), 2)             # hiding in wrap
avail_time_delta_pct = np.round(-4.5 * np.maximum(D, 0) + noise(4.0), 1)             # withdrawal
adherence_delta_pp   = np.round(-3.8 * np.maximum(D, -0.3) + noise(2.5), 2)          # discipline slide
break_overrun_min_30d = np.round(np.maximum(0, 4 + 6.0 * D + noise(4.0)), 1)
late_logins_30d      = rng.poisson(np.maximum(0.2, 1.2 + 2.2 * D))
avg_minutes_late     = np.round(np.maximum(0, 4 + 5.0 * D + noise(4.0)) * (late_logins_30d > 0), 1)
early_departures_30d = rng.poisson(np.maximum(0.05, 0.5 + 1.4 * D))
unplanned_absences_30d = rng.poisson(np.maximum(0.1, 1.0 + 1.8 * D))
fri_mon_absence_ratio  = np.round(np.clip(0.35 + 0.18 * D + noise(0.18), 0, 1)
                                  * (unplanned_absences_30d > 0), 2)
ncns_90d             = rng.poisson(np.maximum(0.01, 0.10 + 0.55 * np.maximum(D, 0)))
pto_burn_rate        = np.round(np.clip(0.9 + 0.45 * D + noise(0.35), 0, 2.5), 2)    # used/accrued 90d
pto_rejections_180d  = rng.poisson(np.maximum(0.05, 0.5 + 0.8 * np.maximum(D, 0)))
shift_swap_requests_30d = rng.poisson(np.maximum(0.1, 0.8 + 1.6 * np.maximum(D, 0)))
short_calls_rate_delta_pp = np.round(1.5 * np.maximum(D, 0) + noise(1.2), 2)
transfer_rate_delta_pp    = np.round(1.1 * np.maximum(D, 0) + noise(1.0), 2)

# ----------------------------------------------------------------------
# TARGET: voluntary exit within 90 days of scoring date
# ----------------------------------------------------------------------
logit = -3.35 + 1.35 * D + rng.normal(0, 0.55, N)   # residual randomness: life happens
p_exit = 1 / (1 + np.exp(-logit))
attrition_90d = rng.binomial(1, p_exit)

# ----------------------------------------------------------------------
# ASSEMBLE
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "agent_id": [f"AGT{100000+i}" for i in range(N)],
    # Layer 1 -- static profile
    "age": age.astype(int),
    "tenure_months": tenure_months.astype(int),
    "commute_minutes": commute_minutes.astype(int),
    "hiring_source": hiring_source,
    "shift_type": shift_type,
    "queue_type": queue_type,
    "education": education,
    "dependents": dependents,
    "prior_jobs": prior_jobs,
    "compa_ratio": compa_ratio,
    "months_since_increment": months_since_increment.astype(int),
    "ijp_rejections": ijp_rejections,
    # Layer 2 -- behavioural trends (30d vs 90d baseline)
    "aht_delta_pct": aht_delta_pct,
    "acw_pct_delta_pp": acw_pct_delta_pp,
    "avail_time_delta_pct": avail_time_delta_pct,
    "adherence_delta_pp": adherence_delta_pp,
    "break_overrun_min_30d": break_overrun_min_30d,
    "late_logins_30d": late_logins_30d,
    "avg_minutes_late": avg_minutes_late,
    "early_departures_30d": early_departures_30d,
    "unplanned_absences_30d": unplanned_absences_30d,
    "fri_mon_absence_ratio": fri_mon_absence_ratio,
    "ncns_90d": ncns_90d,
    "pto_burn_rate": pto_burn_rate,
    "pto_rejections_180d": pto_rejections_180d,
    "shift_swap_requests_30d": shift_swap_requests_30d,
    "short_calls_rate_delta_pp": short_calls_rate_delta_pp,
    "transfer_rate_delta_pp": transfer_rate_delta_pp,
    # Layer 3 -- environmental context
    "occupancy_90d": occupancy_90d,
    "tl_attrition_rate_6m": tl_attrition_rate_6m,
    "team_exits_60d": team_exits_60d,
    "batch_attrition_rate": batch_attrition_rate,
    # Target
    "attrition_90d": attrition_90d,
})

df.to_csv("/home/claude/attrition/contact_centre_attrition_dataset.csv", index=False)
print(f"Rows: {len(df)}  |  Features: {df.shape[1]-2}  |  "
      f"Attrition rate: {df.attrition_90d.mean():.1%}")

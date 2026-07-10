"""
generate_raw_data.py — simulates the three simple files a real company
would export, for demoing 'train on your own data' mode.

250 agents, ~15 months of daily activity. Agents who eventually resign
develop realistic disengagement patterns in their final ~75 days
(adherence slide, late logins, absences, PTO burn, swap spikes).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
N_AGENTS = 250
START, END = pd.Timestamp("2025-04-01"), pd.Timestamp("2026-06-30")

first = ["Aarav","Priya","Rahul","Ananya","Vikram","Sneha","Karan","Divya","Amit","Neha",
         "Rohan","Kavya","Manish","Ritika","Nikhil","Tanvi","Sahil","Meera","Deepak","Isha"]
last = ["Sharma","Verma","Iyer","Mehta","Reddy","Singh","Gupta","Nair","Kumar","Joshi",
        "Das","Menon","Rao","Bansal","Tiwari","Kapoor","Jain","Desai","Choudhary","Pillai"]
names = [f"{rng.choice(first)} {rng.choice(last)} #{i:03d}" for i in range(N_AGENTS)]

# ---- agent master ----
sup_ids = [f"TL{k:02d}" for k in range(1, 21)]
bad_tls = set(rng.choice(sup_ids, 4, replace=False))     # 4 leaky team leaders
master = pd.DataFrame({
    "agent_id": names,
    "date_of_joining": [START - pd.Timedelta(days=int(rng.exponential(400)))
                        for _ in range(N_AGENTS)],
    "hiring_source": rng.choice(["Referral","Job Portal","Walk-in"], N_AGENTS, p=[.3,.55,.15]),
    "shift_type": rng.choice(["Day Fixed","Night Fixed","Rotational","Split"], N_AGENTS, p=[.3,.25,.35,.1]),
    "queue_type": rng.choice(["Voice-Support","Voice-Collections","Chat","Email"], N_AGENTS, p=[.45,.15,.25,.15]),
    "education": rng.choice(["Undergraduate","Graduate","Postgraduate"], N_AGENTS, p=[.25,.6,.15]),
    "commute_minutes": rng.integers(15, 110, N_AGENTS),
    "supervisor_id": rng.choice(sup_ids, N_AGENTS),
    "last_increment_date": [START + pd.Timedelta(days=int(rng.uniform(-200, 200)))
                            for _ in range(N_AGENTS)],
    "compa_ratio": np.round(rng.normal(1.0, 0.1, N_AGENTS).clip(0.75, 1.25), 2),
})

# ---- decide who resigns, when (risk driven by profile) ----
risk = (0.15
        + 0.10 * (master.shift_type != "Day Fixed")
        + 0.08 * (master.hiring_source != "Referral")
        + 0.10 * (master.commute_minutes > 75)
        + 0.10 * (master.compa_ratio < 0.92)
        + 0.15 * master.supervisor_id.isin(bad_tls)
        + rng.normal(0, 0.05, N_AGENTS)).clip(0.03, 0.75)
leaves = rng.random(N_AGENTS) < risk
resig_date = pd.Series(pd.NaT, index=range(N_AGENTS))
resig_date[leaves] = [START + pd.Timedelta(days=int(rng.uniform(150, 440)))
                      for _ in range(leaves.sum())]

exits = pd.DataFrame({
    "agent_id": master.agent_id[leaves].values,
    "resignation_date": resig_date[leaves].values,
    "exit_type": rng.choice(["Voluntary","Involuntary"], leaves.sum(), p=[.85,.15]),
})

# ---- daily activity ----
dates = pd.bdate_range(START, END)
rows = []
for i in range(N_AGENTS):
    doj = master.date_of_joining.iloc[i]
    r_date = resig_date.iloc[i]
    lwd = r_date + pd.Timedelta(days=30) if pd.notna(r_date) else END  # 30-day notice
    base_adh = rng.normal(93, 2); base_aht = rng.normal(320, 30)
    base_acw = base_aht * rng.uniform(0.10, 0.16); base_occ = rng.normal(84, 3)
    for d in dates:
        if d < doj or d > lwd:
            continue
        # disengagement ramps over the 75 days BEFORE resignation
        dis = 0.0
        if pd.notna(r_date):
            days_to = (r_date - d).days
            if days_to <= 75:
                dis = np.clip((75 - days_to) / 75, 0, 1)
        # absences
        p_unp = 0.015 + 0.10 * dis
        p_pto = 0.010 + 0.05 * dis
        u = rng.random()
        if u < p_unp:
            atype = "uninformed" if rng.random() < (0.1 + 0.3 * dis) else "informed"
            # cluster risky absences on Fri/Mon
            if dis > 0.3 and d.dayofweek not in (0, 4) and rng.random() < 0.5:
                continue  # skip; effectively shifts absences toward Fri/Mon
            rows.append([names[i], d.date(), np.nan, np.nan, 0, np.nan, np.nan,
                         np.nan, np.nan, atype, 0])
            continue
        if u < p_unp + p_pto:
            rows.append([names[i], d.date(), np.nan, np.nan, 0, np.nan, np.nan,
                         np.nan, np.nan, "planned", 0])
            continue
        late = max(0, rng.normal(-4 + 14 * dis, 6))
        rows.append([
            names[i], d.date(),
            round(np.clip(base_adh - 8 * dis + rng.normal(0, 2), 60, 100), 1),  # adherence
            round(late if late > 2 else 0),                                      # minutes_late
            int(rng.random() < 0.02 + 0.10 * dis),                               # early_departure
            round(base_aht * (1 + 0.12 * dis) + rng.normal(0, 15)),              # aht
            round(base_acw * (1 + 0.30 * dis) + rng.normal(0, 6)),               # acw
            round(np.clip(base_occ + rng.normal(0, 3), 60, 99), 1),              # occupancy
            round(max(0, rng.normal(2 + 10 * dis, 3)), 1),                       # break overrun
            "", int(rng.random() < 0.02 + 0.12 * dis),                           # swap request
        ])

daily = pd.DataFrame(rows, columns=["agent_id","date","adherence_pct","minutes_late",
    "early_departure","aht_sec","acw_sec","occupancy_pct","break_overrun_min",
    "absence_type","swap_request"])

import os
os.makedirs("data/raw", exist_ok=True)
daily.to_csv("data/raw/daily_activity.csv", index=False)
master.to_csv("data/raw/agent_master.csv", index=False)
exits.to_csv("data/raw/exit_log.csv", index=False)
print(f"daily: {len(daily):,} rows | master: {len(master)} agents | "
      f"exits: {len(exits)} ({(exits.exit_type=='Voluntary').sum()} voluntary)")

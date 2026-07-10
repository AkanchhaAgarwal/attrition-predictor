"""
feature_engine.py — turns three simple company exports into a
leakage-safe, model-ready snapshot table.

Inputs (CSV):
  A. daily_activity : one row per agent per working day
       agent_id, date, adherence_pct, minutes_late, early_departure,
       aht_sec, acw_sec, occupancy_pct, break_overrun_min,
       absence_type (blank | planned | informed | uninformed),
       swap_request (0/1, optional)
  B. agent_master   : one row per agent
       agent_id, date_of_joining, hiring_source, shift_type, queue_type,
       education, commute_minutes, supervisor_id,
       last_increment_date (optional), compa_ratio (optional)
  C. exit_log       : one row per leaver
       agent_id, resignation_date, exit_type (Voluntary | Involuntary)

Design (per the white paper):
  * Monthly snapshots. Features look BACKWARD from each snapshot date T
    (recent = last 30 days, baseline = the 90 days before that).
  * Label looks FORWARD: voluntary resignation within (T, T+90].
    Agents who already resigned before T are excluded from that snapshot,
    so notice-period behaviour can never leak into features.
  * Snapshots younger than 90 days cannot be labelled -> they become the
    PREDICTION set (current employees to score).
"""
import numpy as np
import pandas as pd

REQ_DAILY = ["agent_id", "date", "adherence_pct", "minutes_late", "early_departure",
             "aht_sec", "acw_sec", "occupancy_pct", "break_overrun_min", "absence_type"]
REQ_MASTER = ["agent_id", "date_of_joining", "hiring_source", "shift_type",
              "queue_type", "education", "commute_minutes", "supervisor_id"]
REQ_EXIT = ["agent_id", "resignation_date", "exit_type"]


def validate(daily, master, exits):
    """Return a list of human-readable schema problems (empty = all good)."""
    problems = []
    for name, df, req in [("Daily activity", daily, REQ_DAILY),
                          ("Agent master", master, REQ_MASTER),
                          ("Exit log", exits, REQ_EXIT)]:
        missing = [c for c in req if c not in df.columns]
        if missing:
            problems.append(f"{name} file is missing column(s): {', '.join(missing)}")
    return problems


def _win(df, start, end):
    return df[(df["date"] >= start) & (df["date"] < end)]


def build_snapshots(daily, master, exits, horizon_days=90):
    """Return (labelled_df, predict_df, info) — model-ready tables."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    master = master.copy()
    master["date_of_joining"] = pd.to_datetime(master["date_of_joining"])
    if "last_increment_date" in master.columns:
        master["last_increment_date"] = pd.to_datetime(master["last_increment_date"])
    exits = exits.copy()
    exits["resignation_date"] = pd.to_datetime(exits["resignation_date"])
    vol_exits = exits[exits["exit_type"].str.lower().str.startswith("vol")]
    resig = vol_exits.set_index("agent_id")["resignation_date"]
    any_exit = exits.set_index("agent_id")["resignation_date"]

    has_swap = "swap_request" in daily.columns
    if not has_swap:
        daily["swap_request"] = 0
    daily["absence_type"] = daily["absence_type"].fillna("").astype(str).str.lower()
    is_work = daily["absence_type"].isin(["", "nan"])

    # snapshot dates: 1st of each month from (first date + 120d) to last date
    d0, d1 = daily["date"].min(), daily["date"].max()
    snaps = pd.date_range((d0 + pd.Timedelta(days=120)).replace(day=1) + pd.offsets.MonthBegin(0),
                          d1, freq="MS")

    rows = []
    for T in snaps:
        rec_lo, base_lo = T - pd.Timedelta(days=30), T - pd.Timedelta(days=120)
        # eligible: joined ≥60d before T, not exited before T
        m = master[master["date_of_joining"] <= T - pd.Timedelta(days=60)].copy()
        exited_before = any_exit.reindex(m["agent_id"]).values < T
        m = m[~pd.Series(exited_before, index=m.index).fillna(False)]

        win = _win(daily[daily["agent_id"].isin(m["agent_id"])], base_lo, T)
        rec = win[win["date"] >= rec_lo]
        base = win[win["date"] < rec_lo]

        g_rec, g_base = rec.groupby("agent_id"), base.groupby("agent_id")
        work_rec = rec[is_work.reindex(rec.index).fillna(True)]
        gw_rec = work_rec.groupby("agent_id")
        work_base = base[is_work.reindex(base.index).fillna(True)]
        gw_base = work_base.groupby("agent_id")

        f = pd.DataFrame(index=m["agent_id"].values)
        # behavioural trends: recent vs personal baseline
        f["adherence_delta_pp"] = gw_rec["adherence_pct"].mean() - gw_base["adherence_pct"].mean()
        f["aht_delta_pct"] = (gw_rec["aht_sec"].mean() / gw_base["aht_sec"].mean() - 1) * 100
        acw_rec = gw_rec["acw_sec"].mean() / gw_rec["aht_sec"].mean() * 100
        acw_base = gw_base["acw_sec"].mean() / gw_base["aht_sec"].mean() * 100
        f["acw_pct_delta_pp"] = acw_rec - acw_base
        f["occupancy_90d"] = gw_rec["occupancy_pct"].mean() * 0.33 + gw_base["occupancy_pct"].mean() * 0.67
        f["break_overrun_min_30d"] = gw_rec["break_overrun_min"].mean()
        f["late_logins_30d"] = gw_rec.apply(lambda x: (x["minutes_late"] > 0).sum())
        f["avg_minutes_late"] = gw_rec.apply(lambda x: x.loc[x["minutes_late"] > 0, "minutes_late"].mean())
        f["early_departures_30d"] = gw_rec["early_departure"].sum()
        unp = rec[rec["absence_type"].isin(["informed", "uninformed"])]
        f["unplanned_absences_30d"] = unp.groupby("agent_id").size()
        unp90 = win[win["absence_type"].isin(["informed", "uninformed"])]
        frimon = unp90[unp90["date"].dt.dayofweek.isin([0, 4])].groupby("agent_id").size()
        f["fri_mon_absence_ratio"] = (frimon / unp90.groupby("agent_id").size())
        f["ncns_90d"] = win[win["absence_type"] == "uninformed"].groupby("agent_id").size()
        f["pto_days_30d"] = rec[rec["absence_type"] == "planned"].groupby("agent_id").size()
        f["shift_swap_requests_30d"] = g_rec["swap_request"].sum()

        # counts/ratios that are NaN when the event never happened -> 0
        for c in ["late_logins_30d", "avg_minutes_late", "early_departures_30d",
                  "unplanned_absences_30d", "fri_mon_absence_ratio", "ncns_90d",
                  "pto_days_30d", "shift_swap_requests_30d", "break_overrun_min_30d"]:
            f[c] = f[c].fillna(0)

        # static profile
        mi = m.set_index("agent_id")
        f["tenure_months"] = ((T - mi["date_of_joining"]).dt.days / 30.4).round()
        f["commute_minutes"] = mi["commute_minutes"]
        for c in ["hiring_source", "shift_type", "queue_type", "education"]:
            f[c] = mi[c]
        if "compa_ratio" in mi.columns:
            f["compa_ratio"] = mi["compa_ratio"]
        if "last_increment_date" in mi.columns:
            f["months_since_increment"] = ((T - mi["last_increment_date"]).dt.days / 30.4)\
                .clip(lower=0).round().fillna(f["tenure_months"])

        # environmental: exits by supervisor / population (trailing windows)
        ex6 = vol_exits[(vol_exits["resignation_date"] >= T - pd.Timedelta(days=180)) &
                        (vol_exits["resignation_date"] < T)]
        sup_of = master.set_index("agent_id")["supervisor_id"]
        team_size = master.groupby("supervisor_id").size()
        ex6_by_sup = ex6["agent_id"].map(sup_of).value_counts()
        tl_rate = (ex6_by_sup / team_size).fillna(0)
        f["tl_attrition_rate_6m"] = sup_of.reindex(f.index).map(tl_rate).fillna(0)
        ex2 = vol_exits[(vol_exits["resignation_date"] >= T - pd.Timedelta(days=60)) &
                        (vol_exits["resignation_date"] < T)]
        ex2_by_sup = ex2["agent_id"].map(sup_of).value_counts()
        f["team_exits_60d"] = sup_of.reindex(f.index).map(ex2_by_sup).fillna(0)

        # drop agents with no baseline activity at all
        f = f.dropna(subset=["adherence_delta_pp", "aht_delta_pct"])

        # label: voluntary resignation within (T, T+90]
        r = resig.reindex(f.index)
        labelable = T + pd.Timedelta(days=horizon_days) <= d1
        f.insert(0, "snapshot_date", T)
        f.insert(0, "agent_id", f.index)
        if labelable:
            f["attrition_90d"] = ((r > T) & (r <= T + pd.Timedelta(days=horizon_days))).astype(int)
        else:
            f["attrition_90d"] = np.nan
        rows.append(f.reset_index(drop=True))

    all_snaps = pd.concat(rows, ignore_index=True)
    labelled = all_snaps.dropna(subset=["attrition_90d"]).copy()
    labelled["attrition_90d"] = labelled["attrition_90d"].astype(int)
    latest_T = all_snaps["snapshot_date"].max()
    predict = all_snaps[all_snaps["snapshot_date"] == latest_T].copy()
    info = {"n_snapshots": len(snaps), "labelled_rows": len(labelled),
            "attrition_rate": labelled["attrition_90d"].mean() if len(labelled) else 0,
            "predict_date": latest_T, "predict_rows": len(predict),
            "train_span": f"{labelled['snapshot_date'].min():%b %Y} – "
                          f"{labelled['snapshot_date'].max():%b %Y}" if len(labelled) else "—"}
    return labelled, predict, info

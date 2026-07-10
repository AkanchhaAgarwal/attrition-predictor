"""
Attrition Predictor — Contact Centre Early-Warning System
==========================================================
An end-to-end ML app that predicts 90-day voluntary attrition risk from
WFM behavioural data, and tells the story behind every flag.

Author : Akanchha Agarwal | WFM Simplified
Paper  : "Predicting Voluntary Attrition in Contact Centres" (docs/)
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import reports
from feature_engine import build_snapshots, validate, REQ_DAILY, REQ_MASTER, REQ_EXIT

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from xgboost import XGBClassifier, DMatrix

# ----------------------------------------------------------------------
# Brand + page config
# ----------------------------------------------------------------------
TEAL, TEAL_DARK, GOLD, INK = "#0F6B6B", "#0B4F4F", "#C9A227", "#22313A"
TIER_COLORS = {"Critical": "#C0504D", "Elevated": "#E8A33D",
               "Watch": "#C9A227", "Baseline": "#0F6B6B"}

st.set_page_config(page_title="Attrition Predictor | WFM Simplified",
                   page_icon="📉", layout="wide")

st.markdown(f"""
<style>
    .main-title {{ color:{TEAL_DARK}; font-size:2rem; font-weight:800; margin-bottom:0; }}
    .sub-title  {{ color:{GOLD}; font-weight:600; margin-top:0; }}
    div[data-testid="stMetricValue"] {{ color:{TEAL_DARK}; }}
</style>""", unsafe_allow_html=True)

CAT_COLS = ["hiring_source", "shift_type", "queue_type", "education"]

# ----------------------------------------------------------------------
# Data & models (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(uploaded=None):
    df = pd.read_csv(uploaded) if uploaded is not None \
         else pd.read_csv("data/contact_centre_attrition_dataset.csv")
    return df

def encode(X):
    """One-hot encode categoricals; trees need no scaling."""
    return pd.get_dummies(X, columns=CAT_COLS, drop_first=True)

@st.cache_resource
def train_all_models(df, temporal=False):
    """Train 5 models; return metrics, ROC data, fitted XGBoost scorer,
    feature columns, and test-set info.
    temporal=True -> hold out the most recent snapshots (company mode);
    temporal=False -> stratified random split (demo mode)."""
    drop = [c for c in ["agent_id", "attrition_90d", "snapshot_date"] if c in df.columns]
    y = df["attrition_90d"]
    X = encode(df.drop(columns=drop))
    if temporal and "snapshot_date" in df.columns:
        cut = sorted(df["snapshot_date"].unique())[-2]
        tr_mask = df["snapshot_date"] < cut
        X_tr, X_te = X[tr_mask.values], X[~tr_mask.values]
        y_tr, y_te = y[tr_mask.values], y[~tr_mask.values]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                  stratify=y, random_state=42)
    pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()

    # Logistic regression gets scaled features via a small wrapper
    scaler = StandardScaler().fit(X_tr)
    models = {
        "Logistic Regression": ("scaled", LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree":       ("raw", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                              class_weight="balanced", random_state=42)),
        "Random Forest":       ("raw", RandomForestClassifier(n_estimators=400, max_depth=10,
                                                              min_samples_leaf=20, class_weight="balanced",
                                                              n_jobs=-1, random_state=42)),
        "Gradient Boosting":   ("raw", GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                                  learning_rate=0.05, subsample=0.8,
                                                                  random_state=42)),
        "XGBoost":             ("raw", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                                     subsample=0.8, colsample_bytree=0.7,
                                                     min_child_weight=10, reg_lambda=5,
                                                     scale_pos_weight=pos_w,
                                                     eval_metric="logloss", random_state=42)),
    }

    results, roc_data, fitted = [], {}, {}
    for name, (mode, clf) in models.items():
        Xtr = scaler.transform(X_tr) if mode == "scaled" else X_tr
        Xte = scaler.transform(X_te) if mode == "scaled" else X_te
        clf.fit(Xtr, y_tr)
        s = clf.predict_proba(Xte)[:, 1]
        k = int(len(s) * 0.10)
        top = y_te.iloc[np.argsort(s)[::-1][:k]]
        results.append({"Model": name,
                        "ROC-AUC": roc_auc_score(y_te, s),
                        "PR-AUC": average_precision_score(y_te, s),
                        "Precision@10%": top.mean(),
                        "Recall@10%": top.sum() / y_te.sum()})
        roc_data[name] = roc_curve(y_te, s)
        fitted[name] = clf

    metrics = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

    # Scorer for the register/story: XGBoost retrained on ALL data
    scorer = XGBClassifier(**fitted["XGBoost"].get_params())
    scorer.fit(X, y)
    test_info = {"n_test": len(y_te), "base": float(y_te.mean()),
                 "split": "temporal (latest snapshots held out)" if temporal
                          else "random stratified 75/25"}
    return metrics, roc_data, scorer, list(X.columns), test_info

@st.cache_data
def score_population(df, _scorer, feat_cols):
    """Score everyone, assign tiers + 0-100 score, compute SHAP contributions."""
    drop = [c for c in ["agent_id", "attrition_90d", "snapshot_date"] if c in df.columns]
    X = encode(df.drop(columns=drop))
    X = X.reindex(columns=feat_cols, fill_value=0)
    proba = _scorer.predict_proba(X)[:, 1]

    # native SHAP values from XGBoost (last column = bias term)
    contribs = _scorer.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]
    contribs = pd.DataFrame(contribs, columns=feat_cols, index=df.index)

    scored = df.copy()
    if "attrition_90d" not in scored.columns:
        scored["attrition_90d"] = np.nan
    scored["attrition_score"] = (proba * 100).round(1)   # 0-100 scale
    q95, q85, q70 = np.quantile(proba, [0.95, 0.85, 0.70])
    scored["risk_tier"] = np.select(
        [proba >= q95, proba >= q85, proba >= q70],
        ["Critical", "Elevated", "Watch"], default="Baseline")
    scored["top_driver"] = contribs.idxmax(axis=1)
    cutoffs = {"Critical": q95 * 100, "Elevated": q85 * 100, "Watch": q70 * 100}
    return scored, contribs, cutoffs

# ----------------------------------------------------------------------
# Storytelling engine: SHAP contribution -> plain-English WFM sentence
# ----------------------------------------------------------------------
STORY = {
    "adherence_delta_pp":    lambda v: f"Schedule adherence has slipped {abs(v):.1f} points against their 90-day baseline",
    "late_logins_30d":       lambda v: f"{int(v)} late logins in the last 30 days",
    "avg_minutes_late":      lambda v: f"averaging {v:.0f} minutes late per late login",
    "early_departures_30d":  lambda v: f"{int(v)} early departures in the last 30 days",
    "break_overrun_min_30d": lambda v: f"breaks overrunning by ~{v:.0f} minutes on average",
    "unplanned_absences_30d":lambda v: f"{int(v)} unplanned absences in the last 30 days",
    "fri_mon_absence_ratio": lambda v: f"{v:.0%} of absences falling on Fridays/Mondays",
    "ncns_90d":              lambda v: f"{int(v)} no-call-no-show event(s) in 90 days",
    "pto_burn_rate":         lambda v: f"burning PTO at {v:.1f}× the accrual rate",
    "pto_days_30d":          lambda v: f"{int(v)} planned leave day(s) taken in the last 30 days",
    "pto_rejections_180d":   lambda v: f"{int(v)} PTO request(s) denied in the last 6 months",
    "shift_swap_requests_30d":lambda v: f"{int(v)} shift-swap requests in 30 days",
    "aht_delta_pct":         lambda v: f"AHT drifting {v:+.1f}% vs their own baseline",
    "acw_pct_delta_pp":      lambda v: f"ACW share of handle time up {v:.1f} points ('hiding in wrap')",
    "avail_time_delta_pct":  lambda v: f"available time down {abs(v):.1f}% vs baseline",
    "short_calls_rate_delta_pp": lambda v: f"short-call (<30s) rate up {v:.1f} points",
    "transfer_rate_delta_pp":lambda v: f"transfer rate up {v:.1f} points",
    "occupancy_90d":         lambda v: f"running at {v:.0f}% chronic occupancy (burnout territory)",
    "tenure_months":         lambda v: f"{int(v)} months of tenure (a known risk band)",
    "commute_minutes":       lambda v: f"a {int(v)}-minute one-way commute",
    "compa_ratio":           lambda v: f"paid at {v:.2f}× of band midpoint",
    "months_since_increment":lambda v: f"{int(v)} months since last increment",
    "ijp_rejections":        lambda v: f"{int(v)} internal job application(s) rejected",
    "tl_attrition_rate_6m":  lambda v: f"their team leader has lost {v:.0%} of the team in 6 months",
    "team_exits_60d":        lambda v: f"{int(v)} teammates exited in the last 60 days",
    "batch_attrition_rate":  lambda v: f"{v:.0%} of their training batch has already left",
    "age":                   lambda v: f"age {int(v)}",
    "dependents":            lambda v: f"{int(v)} dependents",
    "prior_jobs":            lambda v: f"{int(v)} previous jobs",
}
CAT_STORY = {"hiring_source": "hired via {}", "shift_type": "working a {} shift",
             "queue_type": "handling the {} queue", "education": "{} education"}

TIER_ACTION = {
    "Critical": ("🔴", "One-to-one retention conversation within 5 working days, led by the "
                 "supervisor with HRBP support. Evaluate schedule or queue accommodation. "
                 "Document the outcome."),
    "Elevated": ("🟠", "Structured supervisor check-in this month. Review PTO rejections, "
                 "IJP history and increment timeline. Recognise good work where merited."),
    "Watch":    ("🟡", "No direct outreach. Monitor for two consecutive months of escalation; "
                 "include in team/shift hot-spot review."),
    "Baseline": ("🟢", "Standard engagement cadence. Feeds programme-level dashboards only."),
}

def render_legend(cutoffs):
    """Clear, always-consistent explanation of the score and tiers."""
    with st.expander("📖 Legend — how to read the Attrition Score and tiers", expanded=False):
        st.markdown(f"""
**Attrition Score (0–100)** = the model's predicted probability of *voluntary exit within
the next 90 days*, ×100. A score of 32 means roughly a 32% chance of leaving in that window
(population base rate ≈ 7).

**Tiers are capacity-based, not statistical** — sized to how many retention conversations
supervisors can actually hold. Cutoffs are computed from this population:

| Tier | Score | Band | Meaning & action |
|---|---|---|---|
| 🔴 **Critical** | ≥ {cutoffs['Critical']:.0f} | Top 5% | Retention 1-on-1 within 5 working days (supervisor + HRBP) |
| 🟠 **Elevated** | ≥ {cutoffs['Elevated']:.0f} | Next 10% | Structured check-in this month; review PTO/IJP/increment history |
| 🟡 **Watch** | ≥ {cutoffs['Watch']:.0f} | Next 15% | No outreach; monitor for 2 consecutive months of escalation |
| 🟢 **Baseline** | < {cutoffs['Watch']:.0f} | Remaining 70% | Standard engagement cadence |

**Headline driver** = the single factor pushing that agent's score up the most
(from the model's own contribution values), i.e. the opening line of the story.
""")


def agent_story(row, contrib_row, top_n=6):
    """Return (drivers_df, narrative) for one agent."""
    c = contrib_row.sort_values(ascending=False)
    pos = c[c > 0.01].head(top_n)
    sentences = []
    for feat in pos.index:
        base = feat
        for cat, tmpl in CAT_STORY.items():
            if feat.startswith(cat + "_"):
                sentences.append(tmpl.format(feat.replace(cat + "_", "")))
                base = None
                break
        if base and base in STORY:
            sentences.append(STORY[base](row[base]))
    drivers = pd.DataFrame({"driver": pos.index, "impact": pos.values})
    return drivers, sentences

# ======================================================================
# UI
# ======================================================================
st.markdown('<p class="main-title">📉 Attrition Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Contact Centre Early-Warning System · WFM Simplified</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Data source")
    mode = st.radio("Mode", ["🎮 Demo (bundled data)", "🏢 Train on my company data"],
                    label_visibility="collapsed")

    up = daily_up = master_up = exit_up = None
    use_sample_raw = False
    if mode.startswith("🎮"):
        up = st.file_uploader("Upload a roster CSV to score", type="csv")
        st.caption("Upload with the `attrition_90d` column to retrain, or without it "
                   "to score your roster. No upload? Runs on 6,000 synthetic agents.")
        st.subheader("📥 Sample roster downloads")
        _base = load_data(None)
        st.download_button("Sample roster — 25 agents (with outcomes)",
                           _base.sample(25, random_state=7).to_csv(index=False).encode(),
                           "sample_roster_with_outcomes.csv", "text/csv")
        st.download_button("Blank scoring template (headers only)",
                           _base.drop(columns=["attrition_90d"]).head(0)
                                .to_csv(index=False).encode(),
                           "roster_scoring_template.csv", "text/csv")
    else:
        st.markdown("Upload **three simple exports** — the app engineers all features, "
                    "trains on your history, and scores your current employees:")
        daily_up = st.file_uploader("1️⃣ Daily activity log", type="csv",
            help="One row per agent per day: " + ", ".join(REQ_DAILY))
        master_up = st.file_uploader("2️⃣ Agent master", type="csv",
            help="One row per agent: " + ", ".join(REQ_MASTER))
        exit_up = st.file_uploader("3️⃣ Exit log", type="csv",
            help="One row per leaver: " + ", ".join(REQ_EXIT))
        use_sample_raw = st.toggle("Or try it with bundled sample company data",
                                   value=not (daily_up and master_up and exit_up))
        st.subheader("📥 File templates")
        for label, cols, fname in [
                ("Daily activity template", REQ_DAILY + ["swap_request"], "daily_activity_template.csv"),
                ("Agent master template", REQ_MASTER + ["last_increment_date", "compa_ratio"], "agent_master_template.csv"),
                ("Exit log template", REQ_EXIT, "exit_log_template.csv")]:
            st.download_button(label, (",".join(cols) + "\n").encode(), fname, "text/csv")

    st.divider()
    st.markdown("**How it works**\n\n"
                "1. Features = 30-day behaviour vs each agent's own 90-day baseline\n"
                "2. Label = voluntary exit within 90 days (leakage-safe by design)\n"
                "3. Five ML models compete; XGBoost scores everyone 0–100\n"
                "4. Every flag comes with a plain-English story")
    st.divider()
    st.markdown("Built by **Akanchha Agarwal**  \n"
                "[WFM Simplified](https://youtube.com/@wfmsimplified) · "
                "White paper in `docs/`")

# --- data flow ---
if mode.startswith("🎮"):
    base_df = load_data(None)
    up_df = load_data(up) if up is not None else None
    train_df = up_df if (up_df is not None and "attrition_90d" in up_df.columns) else base_df
    score_df = up_df if up_df is not None else base_df
    metrics, roc_data, scorer, feat_cols, test_info = train_all_models(train_df)
    scored, contribs, cutoffs = score_population(score_df, scorer, feat_cols)
    if up_df is not None and "attrition_90d" not in up_df.columns:
        st.info("Scoring your uploaded roster with the model trained on the bundled "
                "labelled dataset (no outcome column detected).", icon="📤")
else:
    @st.cache_data
    def load_raw(d, m, e):
        if d is not None and m is not None and e is not None:
            return pd.read_csv(d), pd.read_csv(m), pd.read_csv(e)
        return (pd.read_csv("data/raw/daily_activity.csv"),
                pd.read_csv("data/raw/agent_master.csv"),
                pd.read_csv("data/raw/exit_log.csv"))

    have_uploads = daily_up is not None and master_up is not None and exit_up is not None
    if not have_uploads and not use_sample_raw:
        st.info("Upload all three files in the sidebar — or flip on the sample-data "
                "toggle to see this mode in action.", icon="🏢")
        st.stop()

    daily_df, master_df, exit_df = load_raw(daily_up if have_uploads else None,
                                            master_up if have_uploads else None,
                                            exit_up if have_uploads else None)
    problems = validate(daily_df, master_df, exit_df)
    if problems:
        for p in problems:
            st.error(p, icon="🚫")
        st.stop()

    @st.cache_data
    def engineer(daily_df, master_df, exit_df):
        return build_snapshots(daily_df, master_df, exit_df)

    with st.spinner("Engineering features from your raw files…"):
        labelled, predict_df, eng_info = engineer(daily_df, master_df, exit_df)

    if len(labelled) < 200 or labelled["attrition_90d"].sum() < 20:
        st.error("Not enough labelled history to train reliably — need roughly 200+ "
                 "agent-month rows and 20+ voluntary exits. Add more months of data.",
                 icon="📉")
        st.stop()

    train_df = labelled
    metrics, roc_data, scorer, feat_cols, test_info = train_all_models(labelled, temporal=True)
    scored, contribs, cutoffs = score_population(predict_df, scorer, feat_cols)
    st.success(f"**Trained on your data:** {eng_info['labelled_rows']:,} agent-month "
               f"snapshots ({eng_info['train_span']}, "
               f"{eng_info['attrition_rate']:.1%} attrition) — now scoring your "
               f"**{eng_info['predict_rows']} current employees** as of "
               f"{eng_info['predict_date']:%b %Y}. Validation is temporal: the model "
               f"was tested on months it had never seen.", icon="🏢")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🤖 Model Lab", "🚨 Risk Register", "🔍 Agent Story"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Agents scored", f"{len(scored):,}")
    has_target = scored.attrition_90d.notna().any()
    c2.metric("90-day attrition rate",
              f"{scored.attrition_90d.mean():.1%}" if has_target else "—")
    c3.metric("🔴 Critical", int((scored.risk_tier == "Critical").sum()))
    c4.metric("🟠 Elevated", int((scored.risk_tier == "Elevated").sum()))
    c5.metric("🟡 Watch", int((scored.risk_tier == "Watch").sum()))

    render_legend(cutoffs)

    st.download_button(
        "📄 Download population risk report (PDF)",
        reports.population_report(scored, metrics, cutoffs, TIER_ACTION),
        "attrition_risk_report.pdf", "application/pdf",
        help="Branded PDF: headline numbers, legend, hot-spots, top-15 agents, "
             "model scorecard and governance note — ready to share with ops leadership.")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk-tier distribution")
        tier_counts = scored.risk_tier.value_counts().reindex(
            ["Critical", "Elevated", "Watch", "Baseline"])
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(tier_counts.index, tier_counts.values,
               color=[TIER_COLORS[t] for t in tier_counts.index])
        ax.set_ylabel("Agents"); ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
    with right:
        st.subheader("Where risk concentrates")
        dim = st.selectbox("Slice by", ["shift_type", "hiring_source",
                                        "queue_type", "education"])
        cut = scored.groupby(dim)["attrition_score"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(cut.index, cut.values, color=TEAL, edgecolor=GOLD)
        ax.set_xlabel("Mean predicted 90-day exit risk"); ax.grid(axis="x", alpha=0.25)
        st.pyplot(fig)

    st.info("**Reading this page:** tier thresholds are set by capacity, not statistics — "
            "Critical is sized to the number of retention conversations supervisors can "
            "actually hold this month (top 5%).", icon="💡")

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Five models, one honest scorecard")

    with st.expander("📖 How to read this scorecard (plain English)", expanded=False):
        n_test = test_info["n_test"]
        base = test_info["base"]
        leavers = int(round(n_test * base))
        flagged = int(n_test * 0.10)
        best_p = float(metrics["Precision@10%"].max())
        best_r = float(metrics["Recall@10%"].max())
        caught = int(round(flagged * best_p))
        random_catch = int(round(flagged * base))
        st.markdown(f"""
The model was tested on **{n_test:,} agents it had never seen**. About **{leavers} of them
({base:.0%}) really left** within 90 days. Every number below answers one question:
*how well did the model find those {leavers} people?*

**Why accuracy isn't shown:** a model that just says "nobody will leave" is
{1-base:.0%} accurate — and completely useless. So we use better questions instead:

🎯 **Precision@10%** — *"Were the conversations worth it?"*
Suppose supervisors only have time to talk to the **{flagged} riskiest agents** (top 10%).
Of those {flagged} flagged, **~{caught} really left ({best_p:.0%})**. Picking {flagged}
agents at random would have caught only ~{random_catch}. That's a
**{best_p/base:.1f}× better hit rate** — every conversation is ~{best_p/base:.0f}× more
likely to be with a genuine flight risk.

🕸️ **Recall@10%** — *"How many did we miss?"*
Those same {flagged} conversations reached **{best_r:.0%} of everyone who was going to
leave**. The rest scored below the cutoff and slipped through. Flag more people to catch
more — at the cost of more wasted conversations. It's a dial, not a grade.

📊 **ROC-AUC** — *"Does it rank people correctly?"*
Pick one random leaver and one random stayer: ROC-AUC is the chance the model scored the
leaver higher. Coin flip = 0.50, perfect = 1.00. Healthy attrition models live between
**0.75 and 0.88**. (Above 0.95 usually means the model cheated — see the note below the chart.)

📉 **PR-AUC** — same idea as ROC-AUC, but graded on a harder curve built for rare events.
Its "coin flip" baseline is the base rate ({base:.2f}), so {metrics['PR-AUC'].max():.2f}
means ~{metrics['PR-AUC'].max()/base:.0f}× better than guessing.

**Bottom line for the business:** with no model, retention conversations are guesswork.
With this one, the top-10% list is **{best_p/base:.1f}× denser in real flight risks** and
catches **{best_r:.0%} of upcoming exits** — before the resignation letter arrives.
""")

    st.dataframe(metrics.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
                                       "Precision@10%": "{:.1%}", "Recall@10%": "{:.1%}"})
                 .background_gradient(subset=["ROC-AUC"], cmap="BuGn"),
                 width='stretch', hide_index=True)

    base = scored.attrition_90d.mean()
    best_p = metrics["Precision@10%"].max()
    st.success(f"**Business translation:** flagging the top 10% catches roughly "
               f"{metrics['Recall@10%'].max():.0%} of upcoming leavers with "
               f"{best_p:.0%} precision — a **{best_p/base:.1f}× lift** over the "
               f"{base:.0%} base rate.", icon="🎯")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    palette = [TEAL, GOLD, "#7A4E9E", "#C0504D", "#2E6F40"]
    for (name, (fpr, tpr, _)), c in zip(roc_data.items(), palette):
        auc = metrics.loc[metrics.Model == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#AAB4B8", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
    st.pyplot(fig)

    st.caption(f"Validation split: {test_info['split']}. Healthy attrition models score "
               "0.75–0.88 ROC-AUC — anything above ~0.95 almost always means "
               "notice-period leakage (white paper §6.2).")

# ---------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Tiered risk register")
    render_legend(cutoffs)
    pick = st.multiselect("Tiers", ["Critical", "Elevated", "Watch", "Baseline"],
                          default=["Critical", "Elevated"])
    reg = scored[scored.risk_tier.isin(pick)].sort_values("attrition_score", ascending=False)

    view = reg[["agent_id", "attrition_score", "risk_tier", "top_driver",
                "tenure_months", "shift_type", "queue_type"]].copy()
    st.dataframe(view, width='stretch', hide_index=True,
                 column_config={"attrition_score": st.column_config.ProgressColumn(
                     "Attrition Score", min_value=0, max_value=100,
                     format="%.0f")})
    st.download_button("⬇️ Download register (CSV)",
                       view.to_csv(index=False).encode(),
                       "attrition_risk_register.csv", "text/csv")

    st.warning("**Governance reminder:** these scores exist to trigger supportive "
               "conversations and structural fixes. They must never feed appraisals, "
               "increments or termination decisions (white paper, §9).", icon="⚖️")

# ---------------------------------------------------------------- TAB 4
with tab4:
    st.subheader("The story behind the flag")
    ordered = scored.sort_values("attrition_score", ascending=False)
    sel = st.selectbox("Select an agent (sorted by risk)",
                       ordered.agent_id,
                       format_func=lambda a: f"{a}  ·  "
                       f"{ordered.loc[ordered.agent_id == a, 'risk_tier'].iloc[0]}  ·  "
                       f"score {ordered.loc[ordered.agent_id == a, 'attrition_score'].iloc[0]:.0f}")
    row = scored[scored.agent_id == sel].iloc[0]
    c_row = contribs.loc[scored[scored.agent_id == sel].index[0]]

    g, info = st.columns([1, 2])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=row.attrition_score,
            number={"suffix": " / 100", "font": {"color": TEAL_DARK, "size": 34}},
            title={"text": "Attrition Score", "font": {"size": 15}},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": TIER_COLORS[row.risk_tier]},
                   "steps": [
                       {"range": [0, cutoffs["Watch"]], "color": "#E4F0F0"},
                       {"range": [cutoffs["Watch"], cutoffs["Elevated"]], "color": "#F6EED8"},
                       {"range": [cutoffs["Elevated"], cutoffs["Critical"]], "color": "#FBE9D0"},
                       {"range": [cutoffs["Critical"], 100], "color": "#F8E3E1"}]}))
        fig.update_layout(height=260, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig, width='stretch')
        emoji, action = TIER_ACTION[row.risk_tier]
        st.markdown(f"### {emoji} {row.risk_tier}")
        st.caption("Gauge bands mirror the tier cutoffs in the legend.")

    drivers, sentences = agent_story(row, c_row)
    with info:
        st.markdown(f"**Profile:** {int(row.tenure_months)} mo tenure · "
                    f"{row.shift_type} shift · {row.queue_type} · "
                    f"hired via {row.hiring_source} · "
                    f"{int(row.commute_minutes)} min commute")
        if sentences:
            st.markdown("**What the data is saying:**")
            st.markdown("\n".join(f"- {s.capitalize()}" for s in sentences))
        else:
            st.markdown("No elevated behavioural signals — risk is at baseline.")
        st.markdown(f"**Recommended action:** {action}")
        st.download_button(
            "📄 Download this agent's story (PDF)",
            reports.agent_report(row, sentences, action, drivers),
            f"agent_story_{row.agent_id}.pdf", "application/pdf",
            help="One-pager for the supervisor: score, tier, narrative, drivers "
                 "and the recommended retention action.")

    if len(drivers):
        st.subheader("What's driving this score")
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(drivers) + 1))
        ax.barh(drivers.driver[::-1], drivers.impact[::-1],
                color=TEAL, edgecolor=GOLD)
        ax.set_xlabel("Contribution to risk (SHAP, log-odds)")
        ax.grid(axis="x", alpha=0.25)
        st.pyplot(fig)
        st.caption("Contributions computed natively by XGBoost (pred_contribs=True) — "
                   "each bar is how much that factor pushed THIS agent's score up, "
                   "relative to the average agent.")

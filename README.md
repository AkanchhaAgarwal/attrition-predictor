# 📉 Attrition Predictor — Contact Centre Early-Warning System

**Predict 90-day voluntary attrition from WFM behavioural data — and tell the story behind every flag.**

Most attrition models use HR snapshots (age, salary, tenure) and flag the wrong people too late. This app is built on a different thesis: **attrition is a change in behaviour, and behaviour lives in WFM data** — AHT drift, inflating ACW, sliding adherence, clustered Monday absences, PTO burn, shift-swap spikes.

📄 Full methodology: [`docs/whitepaper_attrition_prediction.pdf`](docs/whitepaper_attrition_prediction.pdf)

---

## ✨ What the app does

| Tab | What you get |
|---|---|
| 📊 **Overview** | Population snapshot, tier distribution, risk hot-spots, **clear score/tier legend**, and a **downloadable branded PDF risk report** |
| 🤖 **Model Lab** | 5 models compared (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) on the metrics that matter at a 7% base rate: ROC-AUC, PR-AUC, Precision/Recall @ top-10% |
| 🚨 **Risk Register** | Every agent given an **Attrition Score (0–100)** and tier (Critical / Elevated / Watch / Baseline), headline driver per agent, CSV export |
| 🔍 **Agent Story** | Pick any agent → risk gauge, plain-English narrative of *why* they're flagged (native XGBoost SHAP contributions), the recommended retention action for their tier, and a **downloadable one-pager PDF** for the supervisor |

## 🧠 The three-layer feature framework

1. **Static profile** — tenure (non-linear risk bands), commute, hiring source, compa-ratio, IJP rejections
2. **Behavioural trends** — every KPI as a *30-day vs 90-day personal baseline delta*, never a snapshot
3. **Environmental context** — team-leader churn rate, team contagion, batch survival, chronic occupancy

The bundled dataset (6,000 agents, 32 features) is synthetic but honest: a latent *disengagement* factor drives both the behavioural KPIs and the exits, reproducing the real-world correlation structure — including the multicollinearity that trips up logistic regression.

## 🚀 Run locally

```bash
git clone https://github.com/<your-username>/attrition-predictor.git
cd attrition-predictor
pip install -r requirements.txt
streamlit run app.py
```

Regenerate the dataset anytime: `python generate_dataset.py`

Have real data? Download the **sample roster** or **blank scoring template** from the sidebar, fill it, and upload — with the `attrition_90d` column to retrain, or without it to simply score your roster. Nothing leaves your machine.

## ☁️ Deploy (free, shareable link)

**Streamlit Community Cloud (recommended):**
1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → *New app*
3. Pick the repo, branch `main`, file `app.py` → **Deploy**
4. Share your `https://<app-name>.streamlit.app` link

**Railway (alternative):**
1. New Project → Deploy from GitHub repo
2. Set the start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Generate a public domain under *Settings → Networking*

## ⚖️ Governance note

Risk scores exist to trigger **supportive conversations and structural fixes**. They must never feed appraisals, increments, or termination decisions. See §9 of the white paper for the full ethics framework (protected attributes, proxy features, transparency).

## 📈 Honest benchmarks

Healthy attrition models score **0.75–0.88 ROC-AUC**. If your first model scores 0.95+, you almost certainly have notice-period leakage — features computed from the weeks after the agent already resigned (white paper §6.2).

---

Built by **Akanchha Agarwal** · [WFM Simplified](https://youtube.com/@wfmsimplified) — making workforce management concepts accessible.

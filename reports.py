"""
reports.py — branded PDF report generation for the Attrition Predictor
Two reports: (1) population risk report, (2) per-agent story one-pager.
Returns bytes, ready for st.download_button.
"""
from io import BytesIO
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)

TEAL_DARK = HexColor("#0B4F4F"); TEAL = HexColor("#0F6B6B")
GOLD = HexColor("#C9A227"); INK = HexColor("#22313A")
TEAL_LIGHT = HexColor("#E4F0F0"); GREY = HexColor("#5A6B72")
TIER_HEX = {"Critical": HexColor("#C0504D"), "Elevated": HexColor("#E8A33D"),
            "Watch": HexColor("#C9A227"), "Baseline": HexColor("#0F6B6B")}

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=14,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=TEAL_DARK,
                    spaceBefore=4, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, textColor=TEAL,
                    spaceBefore=10, spaceAfter=3)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, textColor=GREY)
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.3, leading=11, spaceAfter=0)
HEAD = ParagraphStyle("hd", parent=CELL, fontName="Helvetica-Bold", textColor=white)


def _header(story, title, subtitle):
    story.append(Paragraph(f'<font color="#C9A227"><b>WFM SIMPLIFIED</b></font>', SMALL))
    story.append(Paragraph(title, H1))
    story.append(Paragraph(subtitle, ParagraphStyle("st", parent=BODY, textColor=GREY)))
    story.append(HRFlowable(width="100%", thickness=1.2, color=GOLD))
    story.append(Spacer(1, 8))


def _table(header, rows, widths, tier_col=None):
    data = [[Paragraph(h, HEAD) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), TEAL),
             ("LINEBELOW", (0, 0), (-1, 0), 1, GOLD),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 1), (-1, -1), 0.4, HexColor("#C9D8D8")),
             ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), HexColor("#F2F8F8")))
        if tier_col is not None:
            tier = rows[i - 1][tier_col]
            if tier in TIER_HEX:
                style.append(("TEXTCOLOR", (tier_col, i), (tier_col, i), TIER_HEX[tier]))
    t.setStyle(TableStyle(style))
    return t


def population_report(scored, metrics, cutoffs, tier_actions):
    """Population-level risk report -> PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                            topMargin=14*mm, bottomMargin=14*mm,
                            title="Attrition Risk Report")
    W = A4[0] - 32*mm
    s = []
    _header(s, "Attrition Risk Report",
            f"90-day voluntary exit risk · {len(scored):,} agents scored · {date.today():%d %b %Y}")

    tiers = scored.risk_tier.value_counts()
    has_target = scored["attrition_90d"].notna().any()
    rate = f"{scored.attrition_90d.mean():.1%}" if has_target else "n/a (roster scoring)"
    s.append(Paragraph("1. Headline numbers", H2))
    s.append(_table(["Agents", "Observed 90-day attrition", "Critical", "Elevated", "Watch", "Baseline"],
                    [[f"{len(scored):,}", rate,
                      int(tiers.get("Critical", 0)), int(tiers.get("Elevated", 0)),
                      int(tiers.get("Watch", 0)), int(tiers.get("Baseline", 0))]],
                    [W*0.14, W*0.26, W*0.15, W*0.15, W*0.15, W*0.15]))

    s.append(Paragraph("2. Score legend and tier definitions", H2))
    s.append(Paragraph(
        "The Attrition Score is the model's predicted probability of voluntary exit within "
        "90 days, expressed on a 0–100 scale. Tiers are capacity-based percentile bands, "
        "so cutoffs below are computed from this population.", BODY))
    rows = [[t, f"≥ {cutoffs[t]:.0f}" if t != "Baseline" else f"< {cutoffs['Watch']:.0f}",
             band, tier_actions[t][1]]
            for t, band in [("Critical", "Top 5%"), ("Elevated", "Next 10%"),
                            ("Watch", "Next 15%"), ("Baseline", "Remaining 70%")]]
    s.append(_table(["Tier", "Score cutoff", "Population band", "Bound action"],
                    rows, [W*0.13, W*0.14, W*0.16, W*0.57], tier_col=0))

    s.append(Paragraph("3. Where risk concentrates", H2))
    hot = []
    for dim in ["shift_type", "queue_type", "hiring_source"]:
        g = scored.groupby(dim)["attrition_score"].mean().sort_values(ascending=False)
        hot.append([dim.replace("_", " ").title(), g.index[0],
                    f"{g.iloc[0]:.1f}", g.index[-1], f"{g.iloc[-1]:.1f}"])
    s.append(_table(["Dimension", "Highest-risk segment", "Avg score",
                     "Lowest-risk segment", "Avg score"],
                    hot, [W*0.20, W*0.26, W*0.14, W*0.26, W*0.14]))

    s.append(Paragraph("4. Top 15 agents by attrition score", H2))
    top = scored.sort_values("attrition_score", ascending=False).head(15)
    rows = [[r.agent_id, f"{r.attrition_score:.0f}", r.risk_tier,
             f"{int(r.tenure_months)} mo", r.shift_type, r.queue_type, r.top_driver]
            for r in top.itertuples()]
    s.append(_table(["Agent", "Score", "Tier", "Tenure", "Shift", "Queue", "Headline driver"],
                    rows, [W*0.13, W*0.08, W*0.11, W*0.09, W*0.14, W*0.19, W*0.26], tier_col=2))

    if metrics is not None:
        s.append(Paragraph("5. Model scorecard (held-out test set)", H2))
        rows = [[m.Model, f"{m._2:.3f}", f"{m._3:.3f}", f"{m._4:.1%}", f"{m._5:.1%}"]
                for m in metrics.itertuples()]
        s.append(_table(["Model", "ROC-AUC", "PR-AUC", "Precision@10%", "Recall@10%"],
                        rows, [W*0.32, W*0.17, W*0.17, W*0.17, W*0.17]))

    s.append(Spacer(1, 10))
    s.append(Paragraph(
        "<b>Governance:</b> scores exist to trigger supportive conversations and structural "
        "fixes. They must never feed appraisals, increments, promotions or termination "
        "decisions. Access restricted to supervisor, HRBP and analytics.", SMALL))
    s.append(Paragraph("Generated by the WFM Simplified Attrition Predictor · "
                       "Akanchha Agarwal", SMALL))
    doc.build(s)
    return buf.getvalue()


def agent_report(row, sentences, action, drivers):
    """Per-agent story one-pager -> PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=14*mm, bottomMargin=14*mm,
                            title=f"Agent Risk Story {row.agent_id}")
    W = A4[0] - 36*mm
    s = []
    _header(s, f"Agent Risk Story — {row.agent_id}",
            f"Confidential · supervisor & HRBP only · {date.today():%d %b %Y}")

    s.append(_table(["Attrition Score (0–100)", "Risk tier", "Tenure", "Shift", "Queue", "Hired via"],
                    [[f"{row.attrition_score:.0f}", row.risk_tier,
                      f"{int(row.tenure_months)} mo", row.shift_type,
                      row.queue_type, row.hiring_source]],
                    [W*0.20, W*0.14, W*0.12, W*0.18, W*0.20, W*0.16], tier_col=1))

    s.append(Paragraph("What the data is saying", H2))
    if sentences:
        for sent in sentences:
            s.append(Paragraph(f"• {sent[0].upper()}{sent[1:]}", BODY))
    else:
        s.append(Paragraph("No elevated behavioural signals — risk is at baseline.", BODY))

    if drivers is not None and len(drivers):
        s.append(Paragraph("Score drivers (model contribution)", H2))
        rows = [[d.driver, f"{d.impact:+.3f}"] for d in drivers.itertuples()]
        s.append(_table(["Factor", "Push on risk (log-odds)"], rows, [W*0.70, W*0.30]))

    s.append(Paragraph("Recommended action", H2))
    s.append(Paragraph(action, BODY))
    s.append(Spacer(1, 8))
    s.append(Paragraph(
        "<b>Use with care:</b> this report supports a retention conversation. It is not a "
        "performance document and must not influence appraisal or employment decisions.", SMALL))
    doc.build(s)
    return buf.getvalue()

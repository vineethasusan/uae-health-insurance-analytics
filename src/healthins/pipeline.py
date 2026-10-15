"""Run both analyses: claims FWA detection (train Jan-Aug 2025, test Sep-Dec) and 7-day bed occupancy
forecasting (train 2023-2024, test 2025). Writes outputs/report.md and outputs/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import beds, claims, fwa

OUT = Path("outputs")
AUDIT_SHARE = 0.05          # the investigations team can review 5% of claims
COST_PER_AUDIT = 150        # AED of reviewer time per claim reviewed (assumption)


def run_fwa() -> dict:
    df = claims.generate()
    split = pd.Timestamp("2025-09-01")
    base = fwa.claim_features(df)
    stats = fwa.provider_peer_stats(base[base["service_date"] < split])
    d = base.merge(stats, on="provider_id", how="left")
    train, test = d[d["service_date"] < split], d[d["service_date"] >= split].copy()
    model = fwa.fit(train)
    test["score"] = model.predict_proba(test[fwa.FEATURES])[:, 1]
    test["expected_aed"] = test["score"] * test["amount"]           # rank by money at risk, not just probability
    test["rule_hit"] = fwa.rules(test)
    k = int(len(test) * AUDIT_SHARE)
    fwa_total = test.loc[test["is_fwa"] == 1, "amount"].sum()

    def evaluate(order_by):
        top = test.sort_values(order_by, ascending=False).head(k)
        caught = top.loc[top["is_fwa"] == 1, "amount"].sum()
        return {"claims_reviewed": k, "precision": round(float(top["is_fwa"].mean()), 4),
                "fwa_claims_found": int(top["is_fwa"].sum()), "fwa_claims_total": int(test["is_fwa"].sum()),
                "aed_caught": round(float(caught)), "share_of_fwa_aed": round(float(caught / fwa_total), 4),
                "net_saving_aed": round(float(caught - k * COST_PER_AUDIT))}

    rule_hits = test[test["rule_hit"] == 1]
    rng = np.random.default_rng(0)
    test["rule_rank"] = test["rule_hit"] + rng.random(len(test)) * 0.5          # rules: hits first, in random order
    by_type = test[test["is_fwa"] == 1].assign(top=lambda x: x["score"].rank(ascending=False) <= k)
    top = test.sort_values("expected_aed", ascending=False).head(k).copy()
    top["reasons"] = top.apply(fwa.reason_codes, axis=1)
    top[["claim_id", "provider_id", "member_id", "service_date", "diagnosis", "procedures", "amount", "score", "reasons"]] \
        .head(300).to_csv(OUT / "fwa_review_queue.csv", index=False)
    caught_by_type = (test.sort_values("expected_aed", ascending=False).head(k).query("is_fwa == 1")["fraud_type"].value_counts()
                      / test.query("is_fwa == 1")["fraud_type"].value_counts()).fillna(0).round(3).to_dict()
    prov = test.groupby("provider_id").agg(score=("score", "mean"), fwa=("is_fwa", "mean")).sort_values("score", ascending=False)
    return {"test_claims": len(test), "fwa_rate": round(float(test["is_fwa"].mean()), 4), "fwa_aed_in_test": round(float(fwa_total)),
            "model": evaluate("expected_aed"), "model_by_probability": evaluate("score"), "rules": evaluate("rule_rank"),
            "rules_flag_volume": int(len(rule_hits)), "rules_flag_precision": round(float(rule_hits["is_fwa"].mean()), 4),
            "recall_by_scheme": caught_by_type,
            "top10_provider_fwa_rate": round(float(prov.head(10)["fwa"].mean()), 4),
            "example": top[["claim_id", "diagnosis", "procedures", "amount", "reasons"]].head(5).to_dict("records")}


def run_beds() -> dict:
    df = beds.simulate()
    train = df[df["date"] < "2025-01-01"]
    surv = beds.survival_curve(train)
    model = beds.admission_model(train)
    fc = beds.forecast(df, model, surv)
    t = fc[(fc["date"] >= "2025-01-01")].dropna(subset=["fc_model", "fc_persistence", "fc_4wk_avg"])
    mae = {c: round(float(np.mean(np.abs(t[c] - t["census"]))), 2) for c in ["fc_model", "fc_persistence", "fc_4wk_avg"]}
    pct = {c: round(float(np.mean(np.abs(t[c] - t["census"]) / t["census"])), 4) for c in ["fc_model", "fc_persistence", "fc_4wk_avg"]}
    hi = t["census"] / t["beds"] >= 0.92
    alert = t["fc_model"] / t["beds"] >= 0.90
    return {"test_days": int(t["date"].nunique()), "hospitals": len(beds.HOSPITALS), "mae_beds": mae, "mape": pct,
            "high_occupancy_days": int(hi.sum()),
            "alert_recall": round(float((alert & hi).sum() / max(hi.sum(), 1)), 4),
            "alert_precision": round(float((alert & hi).sum() / max(alert.sum(), 1)), 4),
            "median_los_days": int(np.argmax(surv < 0.5))}


def report(f: dict, b: dict) -> str:
    M, R = f["model"], f["rules"]
    L = ["# UAE Health Insurance & Hospital Analytics: results", "",
         f"## 1. Claims fraud, waste and abuse (test: {f['test_claims']:,} claims, Sep-Dec 2025)", "",
         f"FWA share {f['fwa_rate']:.1%} of claims, AED {f['fwa_aed_in_test']:,} at risk. Audit capacity: 5% of claims ({M['claims_reviewed']:,}).", "",
         "| Review queue ordered by | Precision | FWA claims found | AED caught | Share of FWA AED | Net saving after review cost |",
         "|---|---|---|---|---|---|",
         f"| Claim edits (rules) | {R['precision']:.1%} | {R['fwa_claims_found']} / {R['fwa_claims_total']} | {R['aed_caught']:,} | {R['share_of_fwa_aed']:.1%} | {R['net_saving_aed']:,} |",
         f"| **ML model + peer comparison, ranked by AED at risk** | **{M['precision']:.1%}** | **{M['fwa_claims_found']} / {M['fwa_claims_total']}** | **{M['aed_caught']:,}** | **{M['share_of_fwa_aed']:.1%}** | **{M['net_saving_aed']:,}** |",
         f"| ML model, ranked by probability only | {f['model_by_probability']['precision']:.1%} | {f['model_by_probability']['fwa_claims_found']} / {M['fwa_claims_total']} | {f['model_by_probability']['aed_caught']:,} | {f['model_by_probability']['share_of_fwa_aed']:.1%} | {f['model_by_probability']['net_saving_aed']:,} |", "",
         f"The rules flag {f['rules_flag_volume']:,} claims in total ({f['rules_flag_precision']:.1%} precision). The model's queue is ranked by "
         "expected AED at risk (probability x amount), so reviewers spend their time where the money is.", "",
         "Share of each scheme caught within capacity: " + ", ".join(f"{k} {v:.0%}" for k, v in f["recall_by_scheme"].items()), "",
         "Example review-queue entries:", ""]
    L += [f"- {e['claim_id']} · {e['diagnosis']} · {e['procedures']} · AED {e['amount']:,.0f} → {e['reasons']}" for e in f["example"]]
    L += ["", f"## 2. Bed occupancy forecast, 7 days ahead (2025, {b['hospitals']} hospitals)", "",
          "| Method | Mean error (beds) | Mean % error |", "|---|---|---|",
          f"| **Admissions model + length-of-stay curve** | **{b['mae_beds']['fc_model']}** | **{b['mape']['fc_model']:.1%}** |",
          f"| Same as today | {b['mae_beds']['fc_persistence']} | {b['mape']['fc_persistence']:.1%} |",
          f"| Average of the last 4 weeks | {b['mae_beds']['fc_4wk_avg']} | {b['mape']['fc_4wk_avg']:.1%} |", "",
          f"High-occupancy alert (forecast ≥ 90% when actual ≥ 92%): recall {b['alert_recall']:.0%}, precision {b['alert_precision']:.0%} "
          f"over {b['high_occupancy_days']} high-occupancy hospital-days."]
    return "\n".join(L)


def run() -> dict:
    OUT.mkdir(exist_ok=True)
    f, b = run_fwa(), run_beds()
    m = {"fwa": f, "beds": b}
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    (OUT / "report.md").write_text(report(f, b))
    return m


if __name__ == "__main__":
    m = run()
    print((OUT / "report.md").read_text())

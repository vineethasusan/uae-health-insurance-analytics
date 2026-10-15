"""Fraud, waste and abuse (FWA) detection for health insurance claims.

Three layers, as a payer's special investigations unit would run them:
  1. Claim edits (rules): coding combinations that should not happen.
  2. Behavioural features: how this claim compares with others for the same diagnosis, how often the member
     visits this provider, near-duplicate submissions, and how the provider compares with its peers.
  3. A gradient-boosting model that ranks claims for review, scored by AED recovered within audit capacity.
Every flagged claim carries reason codes an auditor can verify.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HIGH_EM = {"99215", "99285"}
SIMPLE_DX = {"J06.9", "R51", "N39.0", "I10", "M54.5"}
LABS = {"85025", "84443", "82306", "86803", "80053", "83036", "81001", "87086", "87880"}


def claim_features(df: pd.DataFrame, provider_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df.sort_values("service_date", kind="stable").copy()
    codes = d["procedures"].str.split()
    d["high_em"] = codes.map(lambda c: int(any(x in HIGH_EM for x in c)))
    d["high_em_simple_dx"] = d["high_em"] * d["diagnosis"].isin(SIMPLE_DX).astype(int)
    d["n_labs"] = codes.map(lambda c: sum(x in LABS for x in c))
    d["repeat_codes"] = codes.map(lambda c: len(c) - len(set(c)))
    d["mri_for_back_pain"] = codes.map(lambda c: int("72148" in c)) * (d["diagnosis"] == "M54.5").astype(int)
    d["delivery_split"] = codes.map(lambda c: int("59409" in c and "59425" in c))
    med = d.groupby("diagnosis")["amount"].transform("median")
    d["amount_vs_dx_median"] = d["amount"] / med
    d["out_of_emirate"] = (d["member_emirate"] != d["provider_emirate"]).astype(int)

    # member-provider visit frequency in the previous 30 days and near-duplicates within 3 days (past only)
    d["t"] = d["service_date"].astype("int64") // 86_400_000_000_000
    d["visits_30d_same_provider"] = 0
    d["near_duplicate"] = 0
    for _, g in d.groupby(["member_id", "provider_id"], sort=False):
        t = g["t"].to_numpy()
        amt = g["amount"].to_numpy()
        dx = g["diagnosis"].to_numpy()
        v = np.zeros(len(g), int)
        dup = np.zeros(len(g), int)
        for i in range(len(g)):
            prev = (t[:i] >= t[i] - 30)
            v[i] = prev.sum()
            close = (t[:i] >= t[i] - 3) & (dx[:i] == dx[i]) & (np.abs(amt[:i] - amt[i]) / amt[i] < 0.05)
            dup[i] = int(close.any())
        d.loc[g.index, "visits_30d_same_provider"] = v
        d.loc[g.index, "near_duplicate"] = dup
    d["member_claims_ytd"] = d.groupby("member_id").cumcount()

    if provider_stats is not None:
        d = d.merge(provider_stats, on="provider_id", how="left")
    return d


def provider_peer_stats(train_feats: pd.DataFrame) -> pd.DataFrame:
    """How each provider compares with peers of the same type (computed on the training window only)."""
    g = train_feats.groupby(["provider_id", "provider_type"]).agg(
        prov_high_em_rate=("high_em", "mean"), prov_amount_ratio=("amount_vs_dx_median", "mean"),
        prov_labs_per_claim=("n_labs", "mean"), prov_repeat_visits=("visits_30d_same_provider", "mean")).reset_index()
    for c in ["prov_high_em_rate", "prov_amount_ratio", "prov_labs_per_claim", "prov_repeat_visits"]:
        peer = g.groupby("provider_type")[c].transform("median")
        g[c + "_vs_peers"] = g[c] / peer.replace(0, np.nan)
    return g.drop(columns=["provider_type"])


FEATURES = ["amount", "n_procedures", "high_em", "high_em_simple_dx", "n_labs", "repeat_codes", "mri_for_back_pain",
            "delivery_split", "amount_vs_dx_median", "out_of_emirate", "visits_30d_same_provider", "near_duplicate",
            "member_claims_ytd", "age", "chronic", "prov_high_em_rate_vs_peers", "prov_amount_ratio_vs_peers",
            "prov_labs_per_claim_vs_peers", "prov_repeat_visits_vs_peers"]


def rules(d: pd.DataFrame) -> pd.Series:
    """Claim edits a payer typically runs today."""
    return ((d["high_em_simple_dx"] == 1) | (d["n_labs"] >= 5) | (d["near_duplicate"] == 1) | (d["delivery_split"] == 1)
            | (d["visits_30d_same_provider"] >= 4) | (d["repeat_codes"] >= 2)).astype(int)


def fit(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced", random_state=0)
    return m.fit(train[FEATURES], train["is_fwa"])


REASONS = [("high_em_simple_dx", 1, "High-level consultation billed for a simple diagnosis"),
           ("mri_for_back_pain", 1, "MRI billed for routine back pain"),
           ("delivery_split", 1, "Delivery package billed as separate components"),
           ("near_duplicate", 1, "Near-identical claim submitted within 3 days"),
           ("repeat_codes", 1, "Same service code billed more than once"),
           ("n_labs", 5, "Lab tests far beyond what the diagnosis needs"),
           ("visits_30d_same_provider", 4, "Unusually frequent visits to the same provider"),
           ("prov_high_em_rate_vs_peers", 1.8, "Provider bills high-level codes far more than peers"),
           ("prov_labs_per_claim_vs_peers", 1.8, "Provider orders far more tests than peers")]


def reason_codes(row) -> str:
    out = [txt for col, thr, txt in REASONS if pd.notna(row.get(col)) and row[col] >= thr]
    return "; ".join(out[:3]) or "Unusual cost and pattern for this diagnosis"

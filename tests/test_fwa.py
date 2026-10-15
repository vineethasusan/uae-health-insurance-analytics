import pandas as pd

from healthins import fwa


def test_features_use_only_the_past(claims_feats):
    d = claims_feats.sort_values("service_date", kind="stable")
    first = d.groupby(["member_id", "provider_id"]).head(1)
    assert (first["visits_30d_same_provider"] == 0).all() and (first["near_duplicate"] == 0).all()


def test_duplicates_and_upcoding_are_visible_in_features(claims_feats):
    d = claims_feats
    assert d.loc[d["fraud_type"] == "duplicate", "near_duplicate"].mean() > 0.6
    assert d.loc[d["fraud_type"] == "upcoding", "high_em"].mean() > 0.9
    assert d.loc[d["fraud_type"] == "", "high_em_simple_dx"].mean() < 0.02


def test_model_beats_rules_on_money_caught(claims_feats):
    split = pd.Timestamp("2025-09-01")
    stats = fwa.provider_peer_stats(claims_feats[claims_feats["service_date"] < split])
    d = claims_feats.merge(stats, on="provider_id", how="left")
    train, test = d[d["service_date"] < split], d[d["service_date"] >= split].copy()
    m = fwa.fit(train)
    test["ev"] = m.predict_proba(test[fwa.FEATURES])[:, 1] * test["amount"]
    k = int(len(test) * 0.05)
    model_aed = test.nlargest(k, "ev").query("is_fwa == 1")["amount"].sum()
    rules_aed = test[fwa.rules(test) == 1].head(k).query("is_fwa == 1")["amount"].sum()
    assert model_aed > rules_aed


def test_reason_codes_are_readable():
    row = {"high_em_simple_dx": 1, "n_labs": 6, "near_duplicate": 0}
    assert "simple diagnosis" in fwa.reason_codes(row) and "Lab tests" in fwa.reason_codes(row)

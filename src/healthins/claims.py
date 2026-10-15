"""Synthetic UAE health insurance claims with labelled fraud, waste and abuse (FWA).

Outpatient and inpatient claims from Dubai and Abu Dhabi providers, coded with ICD-10 diagnoses and
CPT-style procedures. Planted FWA schemes, each on a minority of providers or claims:
  * upcoding        - billing a higher-level consultation or procedure than the diagnosis supports
  * unbundling      - billing components of a package separately
  * phantom billing - services never delivered (implausible repeat visits for the same member)
  * duplicates      - the same service billed twice with small changes
  * excessive tests - lab panels far beyond what the diagnosis needs
Plus legitimate look-alikes (chronic patients with many visits, complex cases with high costs) so false positives
are realistic. All providers, members and amounts are invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIAGNOSES = {
    # ICD-10: (description, typical procedures, chronic?)
    "J06.9": ("Acute upper respiratory infection", ["99213", "87880"], False),
    "E11.9": ("Type 2 diabetes", ["99214", "83036", "80053"], True),
    "I10": ("Hypertension", ["99213", "93000"], True),
    "M54.5": ("Low back pain", ["99213", "97110", "72100"], False),
    "K21.9": ("Gastro-oesophageal reflux", ["99213", "43235"], False),
    "N39.0": ("Urinary tract infection", ["99213", "81001", "87086"], False),
    "O80": ("Normal delivery", ["59400"], False),
    "S82.8": ("Lower leg fracture", ["99284", "73590", "27786"], False),
    "J45.9": ("Asthma", ["99214", "94010"], True),
    "R51": ("Headache", ["99213"], False),
}
PROCEDURES = {
    # CPT-style: (description, price AED)
    "99213": ("Consultation, low complexity", 300), "99214": ("Consultation, moderate", 450), "99215": ("Consultation, high", 700),
    "99284": ("Emergency visit, high", 1100), "99285": ("Emergency visit, highest", 1800),
    "87880": ("Strep test", 90), "83036": ("HbA1c", 120), "80053": ("Metabolic panel", 160), "93000": ("ECG", 180),
    "97110": ("Physiotherapy session", 350), "72100": ("Spine X-ray", 280), "72148": ("MRI lumbar spine", 2600),
    "43235": ("Upper GI endoscopy", 3200), "81001": ("Urinalysis", 60), "87086": ("Urine culture", 140),
    "59400": ("Delivery package", 9500), "59409": ("Delivery only", 5200), "59425": ("Antenatal visits", 3600),
    "73590": ("Leg X-ray", 260), "27786": ("Fracture treatment", 4200), "94010": ("Spirometry", 220),
    "85025": ("Complete blood count", 70), "84443": ("Thyroid test", 150), "82306": ("Vitamin D", 210), "86803": ("Hepatitis C test", 190),
}
EXTRA_LABS = ["85025", "84443", "82306", "86803", "80053"]
EMIRATES = ["Dubai", "Abu Dhabi", "Sharjah"]


def generate(n_members: int = 12000, n_providers: int = 120, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prov = pd.DataFrame({"provider_id": [f"P{i:03d}" for i in range(n_providers)],
                         "provider_type": rng.choice(["clinic", "polyclinic", "hospital"], n_providers, p=[0.5, 0.3, 0.2]),
                         "provider_emirate": rng.choice(EMIRATES, n_providers, p=[0.55, 0.35, 0.10])})
    bad = rng.choice(n_providers, int(n_providers * 0.12), replace=False)          # 12% of providers run a scheme
    scheme = {p: rng.choice(["upcoding", "unbundling", "phantom", "excessive_tests"]) for p in bad}
    mem = pd.DataFrame({"member_id": [f"M{i:05d}" for i in range(n_members)], "age": rng.integers(1, 75, n_members),
                        "gender": rng.choice(["F", "M"], n_members), "member_emirate": rng.choice(EMIRATES, n_members, p=[0.6, 0.3, 0.1])})
    mem["chronic"] = (rng.random(n_members) < np.clip((mem["age"] - 25) / 120, 0.02, 0.35)).astype(int)

    rows = []
    dates = pd.date_range("2025-01-01", "2025-12-31")
    cid = 0
    for _, m in mem.iterrows():
        visits = rng.poisson(9 if m["chronic"] else 2.2)
        home = prov[prov["provider_emirate"] == m["member_emirate"]]
        for _ in range(visits):
            p = home.iloc[rng.integers(len(home))] if rng.random() < 0.9 else prov.iloc[rng.integers(n_providers)]
            dx_pool = [d for d, v in DIAGNOSES.items() if v[2]] if m["chronic"] and rng.random() < 0.7 else list(DIAGNOSES)
            if m["gender"] == "M" or not (18 <= m["age"] <= 45):
                dx_pool = [d for d in dx_pool if d != "O80"]
            dx = rng.choice(dx_pool)
            procs = list(DIAGNOSES[dx][1])
            fraud_type = ""
            s = scheme.get(int(p["provider_id"][1:]))
            if s and rng.random() < 0.35:
                if s == "upcoding" and procs[0] in ("99213", "99214"):
                    procs[0], fraud_type = rng.choice(["99215", "99285"]), "upcoding"
                    if dx == "M54.5":
                        procs.append("72148")
                elif s == "unbundling" and dx == "O80":
                    procs, fraud_type = ["59409", "59425", "85025", "85025"], "unbundling"
                elif s == "unbundling":
                    procs, fraud_type = procs + ["99213"] * 2, "unbundling"
                elif s == "excessive_tests":
                    procs, fraud_type = procs + list(rng.choice(EXTRA_LABS, rng.integers(3, 6), replace=False)), "excessive_tests"
                elif s == "phantom":
                    fraud_type = "phantom"
            if m["chronic"] and rng.random() < 0.15:                          # legitimate extra labs for chronic care
                procs = procs + ["85025", "80053"]
            if dx == "S82.8" and rng.random() < 0.3:                          # legitimate complex case
                procs = procs + ["99285"]
            amount = sum(PROCEDURES[c][1] for c in procs) * rng.uniform(0.9, 1.15)
            rows.append({"claim_id": f"C{cid:07d}", "member_id": m["member_id"], "provider_id": p["provider_id"],
                         "service_date": dates[rng.integers(len(dates))], "diagnosis": dx, "procedures": " ".join(procs),
                         "n_procedures": len(procs), "amount": round(amount, 2), "fraud_type": fraud_type})
            cid += 1
            if fraud_type == "phantom":                                       # extra visits that never happened, varied to look normal
                for _ in range(rng.integers(1, 4)):
                    pdx = rng.choice(["J06.9", "R51", "M54.5", "N39.0"])
                    pp = list(DIAGNOSES[pdx][1])
                    rows.append({**rows[-1], "claim_id": f"C{cid:07d}", "diagnosis": pdx, "procedures": " ".join(pp),
                                 "n_procedures": len(pp), "amount": round(sum(PROCEDURES[c][1] for c in pp) * rng.uniform(0.9, 1.15), 2),
                                 "service_date": dates[rng.integers(len(dates))]})
                    cid += 1
    df = pd.DataFrame(rows)
    # duplicates: same claim re-submitted a few days later with a slightly different amount
    dup_src = df.sample(frac=0.01, random_state=seed)
    dups = dup_src.assign(claim_id=[f"D{i:07d}" for i in range(len(dup_src))],
                          service_date=dup_src["service_date"] + pd.to_timedelta(rng.integers(0, 3, len(dup_src)), "D"),
                          amount=(dup_src["amount"] * rng.uniform(0.98, 1.02, len(dup_src))).round(2), fraud_type="duplicate")
    df = pd.concat([df, dups], ignore_index=True).sort_values("service_date").reset_index(drop=True)
    df = df.merge(mem, on="member_id").merge(prov, on="provider_id")
    df["is_fwa"] = (df["fraud_type"] != "").astype(int)
    return df

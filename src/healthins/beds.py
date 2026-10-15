"""Hospital bed occupancy forecasting, 7 days ahead.

Simulated inpatient flows for UAE hospitals with the patterns bed managers see: winter respiratory peak,
summer dip when many residents travel, fewer planned admissions during Ramadan and Eid, and the Saturday-Sunday
weekend. Occupancy 7 days out = patients already in bed who will still be there + new admissions that will
still be there. The model forecasts admissions with gradient boosting and applies the length-of-stay survival curve,
compared with the two methods most hospitals use: "same as today" and "same day last week".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HOSPITALS = {"Dubai General Hospital": 420, "Abu Dhabi Medical City": 380, "Sharjah Specialty Hospital": 220,
             "Al Ain Community Hospital": 160}
RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30")]
HORIZON = 7


def _calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    c = pd.DataFrame({"date": dates})
    doy = dates.dayofyear
    c["season"] = 1 + 0.18 * np.cos(2 * np.pi * (doy - 20) / 365.25)          # winter high, summer low
    c["summer_travel"] = ((dates.month == 7) | (dates.month == 8)).astype(int)
    c["ramadan"] = 0
    for s, e in RAMADAN:
        c.loc[(c["date"] >= s) & (c["date"] <= e), "ramadan"] = 1
    c["dow"] = dates.dayofweek
    c["weekend"] = (c["dow"] >= 5).astype(int)
    return c


def simulate(start: str = "2023-01-01", end: str = "2025-12-31", seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end)
    cal = _calendar(dates)
    out = []
    for h, beds in HOSPITALS.items():
        mean_los = 4.2
        base = beds * 0.80 / mean_los
        emergency = base * 0.55 * cal["season"].to_numpy() * np.where(cal["summer_travel"] == 1, 0.85, 1.0)
        elective = base * 0.45 * np.where(cal["weekend"] == 1, 0.35, 1.25) * np.where(cal["ramadan"] == 1, 0.7, 1.0) \
            * np.where(cal["summer_travel"] == 1, 0.8, 1.0)
        outbreak = np.zeros(len(dates))
        for _ in range(3):                                                     # a few flu / RSV surges
            s = rng.integers(0, len(dates) - 30)
            outbreak[s:s + 21] += base * 0.25 * np.sin(np.linspace(0, np.pi, 21))
        adm = rng.poisson(emergency + elective + outbreak)
        census = np.zeros(len(dates) + 60)
        for i, a in enumerate(adm):
            los = np.ceil(rng.lognormal(np.log(mean_los) - 0.3, 0.75, a)).astype(int)
            for L in los:
                census[i:i + L] += 1
        out.append(cal.assign(hospital=h, beds=beds, admissions=adm, census=census[:len(dates)].astype(int)))
    return pd.concat(out, ignore_index=True)


def survival_curve(df: pd.DataFrame, max_days: int = 60) -> np.ndarray:
    """Share of patients still in hospital k days after admission, estimated from training data.
    Approximated from census and admissions flows: S(k) such that census_t ~ sum_k adm_{t-k} S(k)."""
    from sklearn.linear_model import LinearRegression
    rows = []
    for _, g in df.groupby("hospital"):
        a = g["admissions"].to_numpy()
        c = g["census"].to_numpy()
        for t in range(max_days, len(g)):
            rows.append((a[t - max_days + 1:t + 1][::-1], c[t]))
    X = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows], float)
    s = LinearRegression(fit_intercept=False, positive=True).fit(X, y).coef_
    return np.minimum.accumulate(np.clip(s, 0, 1))


def admission_model(train: pd.DataFrame):
    X = _feat(train)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X, train["admissions"])


def _feat(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("hospital")["admissions"]
    return pd.DataFrame({"beds": d["beds"], "dow": d["dow"], "weekend": d["weekend"], "season": d["season"],
                         "summer_travel": d["summer_travel"], "ramadan": d["ramadan"], "month": d["date"].dt.month,
                         # recent level known HORIZON days earlier (captures outbreaks)
                         "adm_7d_avg_lag": g.transform(lambda s: s.shift(HORIZON).rolling(7).mean()),
                         "adm_28d_avg_lag": g.transform(lambda s: s.shift(HORIZON).rolling(28).mean())}, index=d.index)


def forecast(df: pd.DataFrame, model, surv: np.ndarray) -> pd.DataFrame:
    """For every day t, forecast census at t from information available at t - HORIZON."""
    out = []
    for h, g in df.groupby("hospital"):
        g = g.reset_index(drop=True)
        pred_adm = model.predict(_feat(g))
        a = g["admissions"].to_numpy().astype(float)
        n = len(surv)
        fc = np.full(len(g), np.nan)
        for t in range(n + HORIZON, len(g)):
            known = a[t - n + 1:t - HORIZON + 1][::-1]                         # admissions up to t-HORIZON (ages HORIZON..n-1)
            future = pred_adm[t - HORIZON + 1:t + 1][::-1]                     # forecast admissions for the last HORIZON days
            fc[t] = future @ surv[:HORIZON] + known @ surv[HORIZON:n]
        c = g["census"]
        out.append(g.assign(fc_model=fc, fc_persistence=c.shift(HORIZON),
                            fc_4wk_avg=(c.shift(7) + c.shift(14) + c.shift(21) + c.shift(28)) / 4))
    return pd.concat(out, ignore_index=True)

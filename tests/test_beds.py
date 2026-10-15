import numpy as np

from healthins import beds


def test_survival_curve_is_valid_and_forecast_beats_persistence():
    df = beds.simulate(start="2023-01-01", end="2024-12-31", seed=4)
    train = df[df["date"] < "2024-07-01"]
    s = beds.survival_curve(train)
    assert s[0] <= 1 and np.all(np.diff(s) <= 1e-12) and 2 <= np.argmax(s < 0.5) <= 8
    fc = beds.forecast(df, beds.admission_model(train), s)
    t = fc[fc["date"] >= "2024-07-01"].dropna(subset=["fc_model", "fc_persistence"])
    assert np.mean(np.abs(t["fc_model"] - t["census"])) < np.mean(np.abs(t["fc_persistence"] - t["census"]))

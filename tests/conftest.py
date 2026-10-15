import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def claims_feats():
    from healthins import claims, fwa
    df = claims.generate(n_members=3000, n_providers=60, seed=9)
    return fwa.claim_features(df)

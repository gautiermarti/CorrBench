"""Create synthetic dated returns to exercise the benchmark builder."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(20261005)
dates = pd.bdate_range("2018-01-01", periods=500)
assets = np.array([f"synthetic_{i:03d}" for i in range(100)])
returns = 0.004 * rng.normal(size=(500, 1)) + 0.01 * rng.normal(size=(500, 100))
panel = pd.DataFrame(
    {
        "date": np.repeat(dates, len(assets)),
        "asset": np.tile(assets, len(dates)),
        "return": returns.ravel(),
        "liquidity": np.tile(np.arange(1, 101), len(dates)),
    }
)
source = args.output / "returns.parquet"
panel.to_parquet(source, index=False)
config = {
    "market": "synthetic_functional_test_only",
    "source": str(source.resolve()),
    "columns": {
        "date": "date",
        "asset": "asset",
        "return": "return",
        "liquidity": "liquidity",
    },
    "sampling": {"matrix_dim": 80, "lookback": 100, "liquid_universe_size": 100},
    "regime_quantiles": [0.2, 0.8],
    "splits": {
        name: {
            "start": str(dates[start].date()),
            "end": str(dates[start + 11].date()),
            "samples_per_date": 1,
            "seed": 100 + start,
        }
        for name, start in [("train", 100), ("validation", 250), ("test", 400)]
    },
}
(args.output / "config.yaml").write_text(yaml.safe_dump(config))
print("Synthetic panel and config:", args.output)

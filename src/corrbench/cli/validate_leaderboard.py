"""Validate a curated leaderboard's evidence and protocol consistency."""

import argparse
import json
from pathlib import Path

from corrbench.leaderboard import validate_leaderboard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(validate_leaderboard(args.directory), indent=2))


if __name__ == "__main__":
    main()

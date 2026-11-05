from __future__ import annotations

import argparse
import json
from pathlib import Path

from acne_care.dataset.dataset_v3 import build_dataset_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="Resplit external data 70:15:15 while preserving baseline splits")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hash-distance", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(build_dataset_v3(args.baseline_root, args.external_root, args.output_root,
                                     seed=args.seed, max_hash_distance=args.max_hash_distance), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

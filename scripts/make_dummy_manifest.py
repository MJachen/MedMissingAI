from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a schema-only dummy manifest.")
    parser.add_argument("--output", default="data/manifest_example.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "sample_id": "case_0001",
                "label": 0,
                "split": "train",
                "t1": "data/raw/case_0001/t1.nii.gz",
                "t1ce": "data/raw/case_0001/t1ce.nii.gz",
                "t2": "",
                "flair": "data/raw/case_0001/flair.nii.gz",
            },
            {
                "sample_id": "case_0002",
                "label": 1,
                "split": "val",
                "t1": "data/raw/case_0002/t1.nii.gz",
                "t1ce": "",
                "t2": "data/raw/case_0002/t2.nii.gz",
                "flair": "data/raw/case_0002/flair.nii.gz",
            },
            {
                "sample_id": "case_0003",
                "label": 1,
                "split": "test",
                "t1": "",
                "t1ce": "data/raw/case_0003/t1ce.nii.gz",
                "t2": "data/raw/case_0003/t2.nii.gz",
                "flair": "",
            },
        ]
    )
    frame.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()


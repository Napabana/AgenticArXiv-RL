"""Build v5-named multi-turn SFT data using the P0-audited builder.

The implementation is intentionally shared with build_multiturn_sft_v4.py:
only the output dataset identity changes. This keeps the V4 source data,
task manifest, and snapshot fixed so the V5 rebuild isolates the P0 builder
changes rather than mixing in a new task set or environment snapshot.
"""

from . import build_multiturn_sft_v4 as _builder


def main() -> None:
    _builder.main(default_dataset_version=5)


if __name__ == "__main__":
    main()

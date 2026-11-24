"""Compare a single native pilot checkpoint with previously paired SFT trials."""

import argparse
from pathlib import Path

from build_native_grpo_tasks import ADAPTER_SHA, SNAPSHOT_HASH, SPLIT, load_json
from compare_native_sft_benchmark import paired
from generate_parametric_sft_data import sha256_file
from report_native_grpo import metrics, pack
from run_native_grpo import write_json


def compare(part, sft_run, sft_adapter, pilot_run, checkpoint, prior_report, *, archived_adapter_path=None):
    expected = {(tid, trial) for tid in load_json(SPLIT)["split"][part] for trial in range(3)}
    sft, before, source = paired(sft_run, part, expected, archived_adapter_path or sft_adapter)
    pilot, after, destination = paired(pilot_run, part, expected, checkpoint)
    if source != prior_report["benchmark"][part]["sft_files"] or sha256_file(
        sft_adapter / "adapter_model.safetensors"
    ) != ADAPTER_SHA:
        raise ValueError("Archived baseline trials or frozen adapter differ")
    if (sft["snapshot_sha256"] != SNAPSHOT_HASH or pilot["snapshot_sha256"] != SNAPSHOT_HASH
            or sft["requested_model"] != pilot["requested_model"]):
        raise ValueError("Base or snapshot mismatch")
    for field in ("system_prompt_sha256", "tools_sha256", "repeat", "seed", "no_thinking"):
        if sft["native_protocol"][field] != pilot["native_protocol"][field]:
            raise ValueError(f"Native protocol differs: {field}")
    old = pack(metrics(sft_run, part, before))
    new = pack(metrics(pilot_run, part, after))
    result = {"split": part, "sft": old, "pilot": new, "sft_files": source, "pilot_files": destination,
              "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
              "fail_to_pass": sorted(k for k in expected if not before[k] and after[k]),
              "pass_to_fail": sorted(k for k in expected if before[k] and not after[k])}
    if part == "dev":
        result["advance_to_heldout"] = all(new[k]["count"] >= old[k]["count"] and
                                           new[k]["total"] == old[k]["total"] for k in ("strict", "multi_step"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("dev", "iid_test", "ood_test"))
    for field in ("sft_run", "sft_adapter", "pilot_run", "checkpoint", "prior_report", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    result = compare(args.part, args.sft_run, args.sft_adapter, args.pilot_run, args.checkpoint,
                     load_json(args.prior_report))
    write_json(args.output, result)
    print(result["sft"]["strict"], result["pilot"]["strict"], result.get("advance_to_heldout"))


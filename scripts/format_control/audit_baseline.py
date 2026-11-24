"""Verify local historical assets and publish the Format-RL provenance manifest."""
import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
from transformers import AutoTokenizer
from agents.prompt_templates import NATIVE_TOOL_SYSTEM_PROMPT
from rl.format_control.runtime import ROOT, REPORT, V1, digest, read, save, verify_assets

BASELINE = "2f878f07e9f50ee611b13f8e9a65fafa37000561"


def main():
    parser = argparse.ArgumentParser()
    for name in ("base","adapter","v1-data","archives"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    verify_assets(args.base,args.adapter,V1)
    manifest_name = "native_sft_train.jsonl.manifest.json"
    manifest = read(args.v1_data/manifest_name)
    v1 = read(ROOT/"reports/native_sft_final_20261121/training.json")
    assert digest(args.v1_data/manifest_name)==v1["data_manifest_sha256"]
    assert manifest["audit_status"]=="PASS" and manifest["model_id"]=="Qwen/Qwen3.5-2B-Base"
    assert digest(args.v1_data/"native_sft_replay.manifest.json")==manifest["replay_manifest_sha256"]
    for filename, sha in manifest["output_sha256"].items():
        assert digest(args.v1_data/filename)==sha, filename
    for fn,count in [("native_sft_train.jsonl",505),("native_sft_trajectories.jsonl",188)]:
        with (args.v1_data/fn).open(encoding="utf8") as f:
            assert sum(1 for line in f if line.strip()) == count
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    assert hashlib.sha256(tok.chat_template.encode()).hexdigest()==manifest["chat_template_sha256"]
    outputs = ROOT/"reports/grpo_credit_v2"
    artifacts = read(outputs/"gpu_artifacts.json")
    pilots = {}
    for name, record in artifacts["runs"].items():
        if name not in {"E1","E2","E3","E4"}:
            continue
        p = record["pilot"]
        d = args.archives/"ablation"/name/"pilot"
        assert digest(args.archives/p["checkpoint"])==p["sha256"]
        assert digest(d/"config.json")==p["config_sha256"] and p["optimizer_steps"]==25
        cfg = read(d/"config.json")
        pilots[name] = {**{k:v for k,v in cfg.items() if k!="plan"}, "sha256":p["sha256"],
                        "seconds":p["seconds"], "peak_bytes":p["peak_allocated_bytes"]}
    assert len(pilots)==4
    evidence = {str(p.relative_to(ROOT)):digest(p) for p in [
        outputs/"ablation_report.md",outputs/"gpu_artifacts.json",outputs/"paired_case_audit.json",
        outputs/"expert_trajectory_review.md",ROOT/"reports/pending_task_v1/probe_result.md",
        ROOT/"reports/pending_task_v1/readiness.md"]}
    benchmark = {}
    for split in ("dev","iid_test","ood_test"):
        path = args.archives/"sft_benchmark"/split/"protocol_comparison.json"
        proto = read(path)
        assert proto["native_protocol"]["system_prompt_sha256"]==manifest["system_prompt_sha256"]
        assert proto["snapshot_sha256"]==manifest["snapshot_sha256"]
        benchmark[split] = dict(protocol=proto["native_protocol"], source_sha256=digest(path))
    runtime = read(args.archives/"ablation/E1/pilot/runtime.json")
    REPORT.mkdir(exist_ok=True, parents=True)
    save(REPORT/"baseline_manifest.json", dict(baseline_commit=BASELINE, original_worktree_status="clean",
         model=manifest["model_id"], base=str(args.base), base_file_hashes=(ROOT/"reports/native_sft_final_20261121/base_model.sha256").read_text().splitlines(),
         adapter=str(args.adapter), adapter_sha256=V1, adapter_config_sha256=v1["adapter_config_sha256"],
         data_root=str(args.v1_data), v1_data_sha256=manifest["output_sha256"], decisions=505, trajectories=188,
         tokenizer_sha256=manifest["tokenizer_sha256"], template_sha256=manifest["chat_template_sha256"],
         snapshot_sha256=manifest["snapshot_sha256"], tools_sha256=manifest["tools_sha256"],
         historical_native_prompt=manifest["system_prompt_sha256"], archived_protocols=benchmark,
         current_native_prompt_sha256=hashlib.sha256(NATIVE_TOOL_SYSTEM_PROMPT.encode()).hexdigest(),
         native_sft_strict={"dev":"20/24","iid":"44/57","ood":"8/12"}, E4_strict="72/93",
         ablations=pilots, gpu_runtime_historical=runtime, gpu_current=None,
         cpu_runtime=dict(python=platform.python_version(), **{k:version(k) for k in ("torch","transformers","trl","peft")}),
         evidence_sha256=evidence))


if __name__=="__main__":
    main()

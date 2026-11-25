"""Run the production benchmark with a sealed Challenge task catalog."""
import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from audit_snapshot import DATA,REPORT,write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from native_sft_v2_gpu import freeze_protocol


def main(args):
    from benchmark import run_benchmark
    manifest=json.loads((DATA/"manifest.json").read_bytes())
    assets=json.loads((REPORT/"asset_identity.json").read_bytes())
    adapter=args.adapter.resolve(strict=True)
    identity=sha256_file(adapter/"adapter_model.safetensors")
    if args.smoke_e2 and (args.policy!="E2" or args.split!="dev"):
        raise ValueError("The four-step diagnostic is restricted to E2 Dev")
    if sha256_file(args.snapshot)!=assets["snapshot_sha256"]:
        raise ValueError("snapshot mismatch")
    if args.policy=="SFT":
        if identity!=assets["adapter_sha256"]:
            raise ValueError("Unexpected baseline policy")
    else:
        training=json.loads((adapter.parent/"training.json").read_bytes())
        cfg=json.loads((adapter.parent/"config.json").read_bytes())
        planned_steps=4 if args.smoke_e2 else 25
        if cfg.get("protocol")=="single_method_DFA_transfer" and not args.smoke_e2:
            plan_file=adapter.parents[2]/"pilot_inputs.json"
            if sha256_file(plan_file)!=cfg["frozen_inputs_sha256"]:
                raise ValueError("Pilot schedule differs from its recorded input")
            planned_steps=len(json.loads(plan_file.read_bytes())["plan"])
        if identity!=training["adapter_sha256"] or cfg["experiment"]!=args.policy or cfg["steps"]!=planned_steps:
            raise ValueError("Pilot checkpoint identity mismatch")
    if args.split=="sealed_test" or args.split.startswith("production_"):
        frozen=json.loads(args.dev_freeze.read_bytes())
        if frozen["adapters"].get(args.policy)!=identity or frozen["manifest_sha256"]!=sha256_file(DATA/"manifest.json"):
            raise ValueError("Dev freeze differs from current policy or split")
        if args.split.startswith("production_") and args.policy!=frozen["winner"]:
            raise ValueError("Production regression is restricted to the Dev winner")
    if args.split.startswith("production_") and (args.output/"evaluation_identity.json").is_file():
        receipt=json.loads((args.output/"evaluation_identity.json").read_bytes())
        protocol=json.loads((args.output/"protocol_comparison.json").read_bytes())
        if (receipt["policy"]!=args.policy or receipt["split"]!=args.split or receipt["adapter_sha256"]!=identity
                or receipt["manifest_sha256"]!=sha256_file(DATA/"manifest.json")
                or receipt["details_sha256"]!=sha256_file(args.output/"summary.json")
                or receipt["trace_sha256"]!=sha256_file(args.output/"traces.jsonl")
                or protocol["snapshot_sha256"]!=assets["snapshot_sha256"]
                or any(protocol["native_protocol"][k]!=assets[k] for k in ("system_prompt_sha256","tools_sha256"))
                or protocol["native_protocol"]["adapter"]!=str(adapter)):
            raise ValueError("Existing production evaluation differs from this run")
        print(f"Reused verified {args.split} trials from {args.output}",flush=True)
        return
    args.output.mkdir(exist_ok=False,parents=True)
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",TOOL_ROUTER="policy",VLM_ADAPTER_PATH=str(adapter))
    os.environ.pop("AA_EVAL_GREEDY",None)
    for key,d in [("PDF_RAW_PATH","raw"),("PDF_TRANSLATED_PATH","translated"),("PDF_TRANSLATED_LOG_PATH","log"),
                  ("PDF_FIGURES_PATH","figures"),("PDF_CACHE_PATH","cache"),("TRANSLATE_CACHE_PATH","tcache")]:
        os.environ[key]=str(args.output.resolve()/"assets"/d)
    freeze_protocol()
    cmd=["benchmark","--backend","transformers","--model",str(args.base.resolve()),"--local-device","cuda",
         "--local-dtype","bfloat16","--agents","native_tool_calling","--offline","--snapshot",str(args.snapshot),
         "--task-set","expanded","--repeat","3","--seed","42","--no-thinking","--save-traces","--output",str(args.output)]
    if args.split.startswith("production_"):
        part=args.split.removeprefix("production_")
        with patch.object(sys,"argv",cmd+["--split",f"data/splits/v4_86.json:{part}"]):
            run_benchmark.main()
    else:
        source=DATA/(args.split+".jsonl")
        if sha256_file(source)!=manifest["inputs"][source.name]:
            raise ValueError("Challenge catalog mismatch")
        tasks=lines(source)
        with patch.object(sys,"argv",cmd),patch.object(run_benchmark,"get_expanded_tasks",return_value=tasks):
            run_benchmark.main()
    out=json.loads((args.output/"protocol_comparison.json").read_bytes())
    proto=out["native_protocol"]
    if any(proto[k]!=assets[k] for k in ("system_prompt_sha256","tools_sha256")) or proto["adapter"]!=str(adapter):
        raise ValueError("Unexpected evaluation protocol")
    metrics=out["agents"]["native_tool_calling"]
    if metrics["benchmark_errors"] or metrics["request_errors"]:
        raise ValueError("Benchmark or request errors")
    write(args.output/"evaluation_identity.json",dict(policy=args.policy,adapter_sha256=identity,smoke_e2=args.smoke_e2,
          manifest_sha256=sha256_file(DATA/"manifest.json"),split=args.split,
          details_sha256=sha256_file(args.output/"summary.json"),trace_sha256=sha256_file(args.output/"traces.jsonl")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("base","adapter","snapshot","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    parser.add_argument("--policy",choices=["SFT","E1","E2","E3","E4"],required=True)
    parser.add_argument("--split",choices=["dev","sealed_test","production_dev","production_iid_test","production_ood_test"],required=True)
    parser.add_argument("--dev-freeze",type=Path)
    parser.add_argument("--smoke-e2",action="store_true")
    args=parser.parse_args()
    if args.split.startswith("production_"):
        from filelock import FileLock
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(args.output)+".lock"):
            main(args)
    else:
        main(args)

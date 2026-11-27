"""Evaluate the unadapted Base using the registered E2-Repair inputs and production agent."""
import argparse
from collections import Counter
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from filelock import FileLock
from audit_snapshot import write
from audit_tasks import lines
from benchmark.metrics import extract_metrics
from generate_parametric_sft_data import sha256_file
from order_sensitivity import covered
from probe import frozen_inputs
from repair_4090 import fit_turn
from repair_data import RDATA
from repair_run import atomic, checked, read
from rl.anchor_state import fingerprints
from rl.model_loading import load_text_causal_lm
from rl.repair_credit import observations
from rl.task_automaton import transitions
import run_native_grpo

WEIGHT_FILE = "model.safetensors-00001-of-00001.safetensors"
WEIGHTS = {"Base":"928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d",
           "Qwen2B":"aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"}
PRODUCTION = ("dev", "iid_test", "ood_test")


def score(task, sample, native, trial):
    row = transitions(task, sample)
    if row["strict_success"] is not None:
        states = []
        with TemporaryDirectory() as tmp:
            fingerprints(native, task, sample, Path(tmp), guard_states=states)
        for decision, state in zip(row["decisions"], states, strict=True):
            decision["guard_state"] = state
        observations(task, sample, row)
    metrics = extract_metrics(task, sample, "native", trial)
    subgoals, total = covered(task, sample["history"])
    invalid = row["strict_success"] is None
    censored = row["truncated"] and row["first_error_index"] is None
    summary = dict(task=task["id"], trial=trial, invalid=invalid, censored=censored,
                   budget_ended=row["truncated"], strict=None if invalid or censored else row["strict_success"],
                   prefix=row["verified_prefix_length"], subgoals=subgoals, required=total,
                   guards=dict(Counter(d["guard"] for d in row["decisions"] if d.get("guard"))),
                   decisions=len(row["decisions"]), reference_score=metrics.ref_score,
                   termination=sample["termination_reason"])
    return row, summary


def challenge(args, tokenizer, native, receipt, weights):
    model = load_text_causal_lm(args.base, dtype=torch.bfloat16, local_files_only=True).to("cuda")
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    if getattr(model, "peft_config", None):
        raise ValueError("Expected unadapted Base")
    cfg = read(RDATA/"budget.json")
    output = args.output/args.split/args.model_kind
    seed_start = {"dev":73000, "final_holdout":85000}[args.split]
    details = []
    for i, task in enumerate(lines(RDATA/(args.split+".jsonl"))):
        for trial in range(3):
            seed = seed_start + i*3 + trial
            path = output/(task["id"]+f".{trial}.json.gz")
            if not path.exists():
                torch.manual_seed(seed)
                with torch.inference_mode():
                    batch = run_native_grpo.run_rollout(native, {**cfg,"max_turns":cfg["max_turns_by_task"][task["id"]]},
                                                        tokenizer, model, [task])
                sample, = batch["trajectory_results"]
                row, summary = score(task, sample, native, trial)
                atomic(path, dict(adapter=None, base_sha256=weights, freeze=receipt, seed=seed,
                                  sample=sample, row=row, summary=summary))
            raw = json.loads(gzip.decompress(path.read_bytes()))
            item = raw["summary"]
            if (raw["adapter"],raw["base_sha256"],raw["freeze"],raw["seed"],item["task"],item["trial"]) != (
                    None,weights,receipt,seed,task["id"],trial):
                raise ValueError(f"Trial identity mismatch at {path}")
            details.append(item)
    write(output/"summary.json",dict(adapter=None,base_sha256=weights,freeze=receipt,details=details))


def production(args, receipt, weights, assets):
    from benchmark import run_benchmark
    output = args.output/"production"/args.model_kind/args.split
    output.mkdir(parents=True,exist_ok=False)
    os.environ["TOOL_ROUTER"] = "policy"
    os.environ.pop("AA_EVAL_GREEDY",None)
    for key in ("PDF_RAW_PATH","PDF_TRANSLATED_PATH","PDF_TRANSLATED_LOG_PATH","PDF_FIGURES_PATH",
                "PDF_CACHE_PATH","TRANSLATE_CACHE_PATH"):
        os.environ[key] = str(output/"assets"/key)
    cmd = ["benchmark","--backend","transformers","--model",str(args.base),"--local-device","cuda",
           "--local-dtype","bfloat16","--agents","native_tool_calling","--offline","--snapshot",str(args.snapshot),
           "--task-set","expanded","--repeat","3","--seed","42","--no-thinking","--save-traces",
           "--output",str(output),"--split","data/splits/v4_86.json:"+args.split]
    with patch.object(sys,"argv",cmd):
        run_benchmark.main()
    report = read(output/"protocol_comparison.json")
    metrics = report["agents"]["native_tool_calling"]
    protocol = report["native_protocol"]
    if (metrics["request_errors"] or metrics["benchmark_errors"] or protocol["adapter"] is not None
            or any(protocol[k] != assets[k] for k in ("system_prompt_sha256","tools_sha256"))):
        raise ValueError("Production Base protocol mismatch")
    write(output/"identity.json",dict(adapter=None,base_sha256=weights,freeze=receipt,
                                      trace=sha256_file(output/"traces.jsonl")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite",choices=("production","challenge"))
    p.add_argument("split",choices=(*PRODUCTION,"final_holdout"))
    p.add_argument("--model-kind",choices=WEIGHTS,default="Base")
    p.add_argument("--reference-base",type=Path,help="Verified SFT/RL backbone used to audit the frozen protocol")
    for name in ("base","reference-adapter","snapshot","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    if a.split not in (PRODUCTION if a.suite=="production" else ("dev","final_holdout")):
        p.error("Invalid split for this suite")
    a.base, a.output, a.snapshot = (p.resolve() for p in (a.base,a.output,a.snapshot))
    a.output.mkdir(parents=True,exist_ok=True)
    os.environ.pop("VLM_ADAPTER_PATH",None)
    os.environ.update(HF_HUB_OFFLINE="1",STORE_BACKEND="memory",OMP_NUM_THREADS="1")
    with FileLock(str(a.output/f".{a.suite}-{a.split}.lock"),timeout=0):
        receipt = checked()
        # Verify the exact published reference assets and activate its native prompt.
        # The reference adapter is only hashed here. Neither inference path loads it.
        a.adapter = a.reference_adapter
        from argparse import Namespace
        reference = Namespace(base=a.reference_base or a.base, adapter=a.reference_adapter, snapshot=a.snapshot)
        tok, native, assets = frozen_inputs(reference)
        if a.model_kind == "Qwen2B":
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
            if hashlib.sha256(tok.chat_template.encode()).hexdigest() != assets["chat_template_sha256"]:
                raise ValueError("Official 2B chat template differs from frozen Native protocol")
            native.audit_tokenizer = tok
        weights = sha256_file(a.base/WEIGHT_FILE)
        if weights != WEIGHTS[a.model_kind]:
            raise ValueError(f"Weight mismatch for {a.model_kind}")
        write(a.output/f"model-{a.suite}-{a.split}.json", dict(model=a.model_kind,weights=weights,
              tokenizer=sha256_file(a.base/"tokenizer.json"),chat_template=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
              system_prompt=assets["system_prompt_sha256"],tools=assets["tools_sha256"],adapter=None,freeze=receipt))
        if a.suite=="production":
            production(a,receipt,weights,assets)
        else:
            with patch.object(run_native_grpo,"generate_native_turn",
                              partial(fit_turn,run_native_grpo.generate_native_turn)):
                challenge(a,tok,native,receipt,weights)


if __name__=="__main__":
    main()

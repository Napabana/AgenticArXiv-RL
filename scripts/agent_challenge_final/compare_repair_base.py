"""Verify bare-Base trials and pair both benchmarks with the archived policies."""
import argparse
import gzip
import json
from pathlib import Path

from audit_snapshot import ROOT, write
from compare_native_sft_benchmark import paired
from generate_parametric_sft_data import sha256_file
from repair_compare25 import trials
from repair_matched_results import compare, summarize
from repair_run import checked, read, evaluation_freeze

from repair_base_benchmark import WEIGHTS


def new_benchmark(root, prior, freeze, weights, kind):
    result = {}
    for split in ("dev", "final_holdout"):
        folder = root/split/kind
        base = trials(folder,None,freeze,split)
        for task, trial in base:
            raw = json.loads(gzip.decompress((folder/f"{task}.{trial}.json.gz").read_bytes()))
            if raw["base_sha256"] != WEIGHTS[kind]:
                raise ValueError("Incorrect Base weight")
        models = {p:trials(prior/split/p,h,freeze,split) for p,h in weights.items()}
        result[split] = dict(base=summarize(base),
                             models={p:summarize(rows) for p,rows in models.items()},
                             base_to_policy={p:compare(base,rows) for p,rows in models.items()})
    return result


def old_benchmark(root, prior, freeze, weights, sft_run, kind):
    old = read(ROOT/"reports/native_sft_v2/paired_evaluation.json")
    if old["v2_sha256"] != weights["SFT"]:
        raise ValueError("SFT baseline identity mismatch")
    result = {}
    for split, data in old["splits"].items():
        sft = {(r["id"],r["trial"]):r["v2"] for r in data["trials"]}
        proto, base, files = paired(root/"production"/kind,split,set(sft),None)
        base_protocol = proto["native_protocol"]
        if read(root/"production"/kind/split/"identity.json") != dict(adapter=None,base_sha256=WEIGHTS[kind],
                freeze=freeze,trace=files["trace_sha256"]):
            raise ValueError("Production Base identity mismatch")
        models = {}
        for policy in weights:
            folder = sft_run/"benchmark" if policy=="SFT" else prior/"production"/policy
            adapter = sft_run/"formal/adapter" if policy=="SFT" else prior/"train"/("checkpoint-"+policy)
            reference, rows, hashes = paired(folder,split,set(sft),adapter)
            for key in ("system_prompt_sha256","tools_sha256","repeat","seed","no_thinking","split"):
                if reference["native_protocol"][key] != base_protocol[key]:
                    raise ValueError(f"Production protocol drift in {policy}/{split}/{key}")
            if sha256_file(adapter/"adapter_model.safetensors") != weights[policy]:
                raise ValueError("Archive weight mismatch")
            if policy=="SFT" and (rows!=sft or hashes!=data["v2_files"]):
                raise ValueError("SFT archive changed")
            if policy!="SFT" and read(folder/split/"identity.json") != dict(
                    adapter=weights[policy],freeze=freeze,trace=hashes["trace_sha256"]):
                raise ValueError("RL archive changed")
            models[policy] = dict(strict=sum(rows.values()),paired=len(base),
                fail_to_pass=[k for k in sorted(base) if not base[k] and rows[k]],
                pass_to_fail=[k for k in sorted(base) if base[k] and not rows[k]])
        result[split] = dict(base=dict(strict=sum(base.values()),scorable=len(base)),models=models,files=files)
    return result


def main(root,prior,sft_run,kind):
    freeze = checked()
    weights = read(prior/"checkpoints.json")
    if weights["freeze"]!=freeze or read(prior/"evaluation_freeze.json")!=evaluation_freeze(prior,freeze,weights):
        raise ValueError("Previous evaluation freeze changed")
    result = dict(model_kind=kind,base_sha256=WEIGHTS[kind],adapter=None,freeze=freeze,reference_adapters=weights["adapters"],
                  challenge=new_benchmark(root,prior,freeze,weights["adapters"],kind),
                  production=old_benchmark(root,prior,freeze,weights["adapters"],sft_run,kind))
    write(root/"comparison.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model-kind",choices=WEIGHTS,default="Base")
    for name in ("output","prior","sft-run"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    main(a.output.resolve(),a.prior.resolve(),a.sft_run.resolve(),a.model_kind)

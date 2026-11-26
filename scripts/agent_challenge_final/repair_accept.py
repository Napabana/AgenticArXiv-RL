"""Verify CPU evidence, freeze the runnable sources and exercise the actual trainer configuration."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from audit_snapshot import ROOT, write
from audit_tasks import lines
from generate_parametric_sft_data import sha256_file
from probe import frozen_inputs
from repair_data import RDATA, RREPORT
from repair_run import audit_group, checked, read, schedule
from rl.credit_diagnostics import Diagnostics
from run_grpo_credit_v2 import execute


def require(result, why):
    if not result:
        raise ValueError(why)


def cpu_suite(path):
    root=ET.parse(path)
    suite=root.getroot().find("testsuite")
    numbers={k:int(suite.attrib[k]) for k in ("tests","errors","failures","skipped")}
    require(numbers["tests"] and numbers["errors"]==numbers["failures"]==0,str(path))
    return {**numbers,"seconds":float(suite.attrib["time"]), "junit_sha256":sha256_file(path)}


def freeze(args):
    tok,native,assets=frozen_inputs(args)
    data=read(RREPORT/"data_audit.json")
    budget=read(RDATA/"budget.json")
    credit=read(RREPORT/"credit_coverage.json")
    dependency=read(RREPORT/"dependency_audit.json")
    runs=lines(RREPORT/"historical_trajectories.jsonl")
    coverage=lines(RREPORT/"coverage_matrix.jsonl")
    require((data["unique_papers"],data["with_detail"])==(419,20),"Snapshot inventory drift")
    require(not (data["unsupported"] or data["replay_failures"] or data["holdout_collisions"]),"Corpus gate")
    require(len(coverage)==data["matrix_rows"] and all(x["recorded"] and x["executed"] for x in coverage),"Coverage gate")
    require(len(data["gold"])==44 and all(x["strict"] and x["completion_tokens"]<budget["max_completion_length"] and
        x["longest_turn"]<budget["max_new_tokens_per_turn"] and x["steps"]+1<=budget["max_turns_by_task"][x["id"]]
        for x in data["gold"]),"Gold replay or budget")
    require(len(dependency["tasks"])==44 and all(x["strict"]==1 for x in dependency["tasks"]) and
        any(x["alternate"] for x in dependency["tasks"]) and dependency["environment_control"]["group_active_tokens"]==0,"Dependency/environment gate")
    for arm, size, cuts in (("probe",768,201),("dfa-transfer",96,15),("dfa-all24",96,29)):
        history=[x for x in runs if x["source"]==arm]
        require(len(history)==size and sum(bool(x["cause"]) for x in history)==cuts,"Historical denominator")
    require(credit["probe"]["identical_task_local_groups"]>0 and credit["probe"]["nonzero_tokens"]>0,"No archived signal")
    for name in ("repeat","pending","tool_parse","reference_argument"):
        flag=credit["diagnostic_flags"]["dfa-all24/"+name]
        require(flag["new_negative"]>flag["old_negative"],"No improved localization for "+name)
    for part,n in (("train",24),("dev",8),("final_holdout",12)):
        p=RDATA/(part+".jsonl")
        require(len(lines(p))==n and sha256_file(p)==data["split_sha"][part],"Split mismatch")
    tests={k:cpu_suite(path) for k,path in (("project",args.tests),("challenge_and_guards",args.targeted))}
    require(tests["project"]["tests"]>=1000,"Partial CPU suite")
    tasks=lines(RDATA/"train.jsonl")
    plan=schedule(tasks)
    proofs={r["id"]:r["sample"] for r in map(json.loads,gzip.decompress((RREPORT/"gold.jsonl.gz").read_bytes()).splitlines())}
    diagnostics=Diagnostics()
    for task in tasks:
        group=[proofs[task["id"]]]*4
        rows,stats=audit_group(task,group,native)
        require(stats["valid"] and not stats["positive_tokens"] and not stats["negative_tokens"] and
                all(r["strict_success"]==1 for r in rows),"Gold no-contrast control")
        diagnostics.add(task,group,rows,"E2")
    diagnostics.report()
    base_hashes=dict(line.split()[::-1] for line in (ROOT/"reports/native_sft_final_20261121/base_model.sha256").read_text().splitlines())
    assets.update(base_files=base_hashes,adapter_config_sha256=sha256_file(args.adapter/"adapter_config.json"))
    write(RREPORT/"assets.json",assets)
    write(RREPORT/"cpu_tests.json",tests)
    cfg=dict(steps=72,groups_per_task=3,G=4,new_online_rollouts=288,learning_rate=5e-6,kl_beta=.04,
             dfa_lambda=.5,loss_type="dapo",reference="Native SFT v2",checkpoints=[24,48,72],seed=42,plan=plan)
    write(RDATA/"train_config.json",cfg)
    dry=SimpleNamespace(smoke=False,pilot_steps=72,experiment="E2",dry_run=True)
    execute(dry,challenge=(dict(tasks=tasks,**budget),tok,native,assets,plan,{"checkpoint_interval":24,"loss_type":"dapo"}))
    sources=set()
    for folder in ("AgenticArxiv/rl","AgenticArxiv/agents","AgenticArxiv/benchmark","AgenticArxiv/tools","AgenticArxiv/utils","scripts/agent_challenge_final"):
        sources.update((ROOT/folder).rglob("*.py"))
    for name in ("run_grpo_credit_v2","run_native_grpo","audit_grpo_credit_v2","native_sft_v2_gpu","compare_native_sft_benchmark"):
        sources.add(ROOT/"scripts"/(name+".py"))
    sources.add(ROOT/"scripts/agent_challenge_final/repair_pipeline.sh")
    sources.add(ROOT/"AgenticArxiv/tests/test_e2_repair.py")
    files=[*(RDATA/n for n in ("train.jsonl","dev.jsonl","final_holdout.jsonl","budget.json","train_config.json")),
           *(RREPORT/n for n in ("data_audit.json","coverage_matrix.jsonl","gold.jsonl.gz","dependency_audit.json","credit_coverage.json",
                                 "historical_trajectories.jsonl","credit_positions.jsonl","truncation.json","assets.json","cpu_tests.json"))]
    write(RDATA/"freeze.json",dict(cpu_pass=True,files={p.relative_to(ROOT).as_posix():sha256_file(p) for p in files},
        source_lf_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in sorted(sources)}))
    print("CPU_PASS",checked(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("base","adapter","snapshot","tests","targeted"):
        p.add_argument("--"+key,type=Path,required=True)
    freeze(p.parse_args())

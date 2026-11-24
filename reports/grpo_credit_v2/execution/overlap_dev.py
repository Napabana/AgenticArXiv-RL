"""Overlap one existing Dev benchmark with the next pilot on the same GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import time
from concurrent.futures import ThreadPoolExecutor

repo = Path.cwd()
sys.path.insert(0, str(repo/"scripts"))
from grpo_credit_v2_ablation import load_json, train

base, snapshot, dest, ckpt = map(Path, sys.argv[1:5])
run = dest.parents[1]
mode = dest.parent.name
root = run.parent
benchmark = ["bash", "scripts/run_qwen35_native_benchmark.serial.sh", *sys.argv[1:]]
next_mode = {"E1":"E2","E2":"E3","E3":"E4"}.get(mode)
if not (next_mode and sys.argv[5:] == ["dev"] and (root / ("overlap-" + mode)).is_file()):
    sys.exit(subprocess.call(benchmark))

def digest(f):
    with f.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

assert ckpt == run/mode/"pilot/checkpoint-25"
assert digest(ckpt/"adapter_model.safetensors") == load_json(ckpt.parent/"training.json")["adapter_sha256"]
assert (run/next_mode/"smoke/training.json").is_file()
cfg = SimpleNamespace(out=run, base=base, adapter=Path(os.environ["SFT_ADAPTER"]), snapshot=snapshot, dry_run=False)
timeline = root / ("overlap-" + mode + "-events.jsonl")
def log(event, **kw):
    with timeline.open("a",encoding="utf8") as stream:
        stream.write(json.dumps(dict(event=event, epoch=time.time(), **kw))+"\n")

def launch_training():
    log("training_started",mode=next_mode)
    try:
        train(cfg, next_mode, False)
    finally:
        log("training_process_finished",mode=next_mode)

def oom(path):
    if not path.exists():
        return False
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-16000))
        s=stream.read().lower()
    return b"cuda out of memory" in s or b"outofmemoryerror" in s

if dest.exists(): raise RuntimeError(f"Unexpected existing Dev output {dest}")
with ThreadPoolExecutor(max_workers=1) as pool:
    training=pool.submit(launch_training)
    log("dev_started", mode=mode)
    with (root/("overlap-"+mode+"-benchmark.log")).open("a") as stream:
        status=subprocess.call(benchmark, stdout=stream, stderr=subprocess.STDOUT)
    log("dev_finished",status=status)
    try:
        training.result()
    except subprocess.CalledProcessError:
        if not oom(run/next_mode/"pilot.log"): raise
        log("training_oom_serial_fallback")
        (root/"overlap-disabled").touch()
        # The outer controller resumes the pilot after returning from this benchmark.
if status:
    if not oom(root/("overlap-"+mode+"-benchmark.log")): sys.exit(status)
    from grpo_credit_v2_ablation import preserve_attempt
    if dest.exists(): preserve_attempt(dest)
    log("dev_oom_serial_retry")
    (root/"overlap-disabled").touch()
    subprocess.run(benchmark, check=True)
log("pair_finished")

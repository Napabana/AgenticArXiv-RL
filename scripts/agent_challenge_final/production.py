"""Pair the Dev winner with the archived SFT v2 (83/93) production trials."""
import argparse
import json
from pathlib import Path
from audit_snapshot import ROOT,write
from compare_native_sft_benchmark import paired
from generate_parametric_sft_data import sha256_file


def run(baseline,after,adapter):
    v2=json.loads((ROOT/"reports/native_sft_v2/paired_evaluation.json").read_bytes())
    result={}
    for part,archive in v2["splits"].items():
        keys={(r["id"],r["trial"]) for r in archive["trials"]}
        source=json.loads((baseline/part/"protocol_comparison.json").read_bytes())
        original=Path(source["native_protocol"]["adapter"])
        if sha256_file(original/"adapter_model.safetensors")!=v2["v2_sha256"]:
            raise ValueError("Production SFT v2 reference mismatch")
        old,a,evidence=paired(baseline,part,keys,original)
        new,b,files=paired(after,part,keys,adapter)
        if evidence!=archive["v2_files"] or old["snapshot_sha256"]!=new["snapshot_sha256"]:
            raise ValueError("Archived benchmark traces or snapshot changed")
        for k in ("system_prompt_sha256","tools_sha256","repeat","seed","no_thinking","split"):
            if old["native_protocol"][k]!=new["native_protocol"][k]:
                raise ValueError(f"Protocol changed: {k}")
        result[part]=dict(sft=sum(a.values()),winner=sum(b.values()),n=len(keys),
                          trials=[dict(id=t,trial=i,sft=a[t,i],winner=b[t,i]) for t,i in sorted(keys)],
                          repairs=[k for k in sorted(keys) if not a[k] and b[k]],regressions=[k for k in sorted(keys) if a[k] and not b[k]],
                          sft_files=evidence,winner_files=files)
    if sum(s["sft"] for s in result.values())!=83 or sum(s["n"] for s in result.values())!=93:
        raise ValueError("Archived 83/93 baseline not reproduced")
    write(after.parent/"production_paired.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("baseline","after","adapter"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    run(a.baseline,a.after,a.adapter)

"""Compare fixed real rollout inputs on the same SFT weights. No policy update or task scoring."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
import torch.nn.functional as F
from audit_tasks import lines
from probe import frozen_inputs
from repair_data import RDATA
from repair_run import checked, read
from run_native_grpo import load_policy


def measure(function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    result=function()
    torch.cuda.synchronize()
    return result,dict(seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated(),
                       peak_reserved=torch.cuda.max_memory_reserved())


def sample_args(a,tok,native,group_index,row_index):
    group=read(a.old/"train/groups"/f"group_{group_index:03}.json")
    sample=group["trajectories"][row_index]
    t=next(x for x in lines(RDATA/"train.jsonl") if x["id"]==group["task_id"])
    messages=sample["native_messages"]
    end=next(i for i,m in enumerate(messages) if m["role"]=="assistant")
    with TemporaryDirectory() as tmp:
        ep=native.episode(t,Path(tmp))
        ep.prompt=messages[:end]
        prompt=native.prompt_ids(tok,ep)
    spans=sample["turn_token_spans"]
    if len(spans)<2:raise ValueError("Missing real second action")
    start,stop=spans[1]
    ids=prompt+sample["completion_ids"][:start]
    target=sample["completion_ids"][start:min(stop,start+16)]
    if len(target)!=16:raise ValueError("Insufficient recorded assistant tokens")
    sha=hashlib.sha256(json.dumps([ids,target],separators=(",",":")).encode()).hexdigest()
    return dict(tokens=len(ids),target_tokens=len(target),ids_sha256=sha,task=t["id"],
                group_index=group_index,row_index=row_index),torch.tensor([ids],device="cuda"),torch.tensor([target],device="cuda")


def one(model,ids,target):
    model.eval()
    with torch.no_grad():
        pre,t1=measure(lambda:model(input_ids=ids,use_cache=True,logits_to_keep=1))
        last_logits=pre.logits.float()
        first_score=F.cross_entropy(last_logits[:,-1],target[:,0]).item()
        cache=pre.past_key_values
        del pre,last_logits
        def decode():
            nonlocal cache
            for i in range(target.size(1)):
                out=model(input_ids=target[:,i:i+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=out.past_key_values
        _,t2=measure(decode)
        del cache
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(512)
    def backward():
        logits=model(input_ids=torch.cat([ids,target],dim=1),use_cache=False,logits_to_keep=target.size(1)+1).logits[:,:-1]
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target.reshape(-1))
        loss.backward()
        return loss.detach().item()
    loss,t3=measure(backward)
    model.zero_grad(set_to_none=True)
    return dict(prefill=t1,decode16=t2,forward_backward=t3,first_target_nll=first_score,teacher_forced_loss=loss)


def main(a):
    tok,native,assets=frozen_inputs(a)
    model=load_policy(a,trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    result=dict(freeze=checked(),adapter_sha256=assets["adapter_sha256"],gpu=torch.cuda.get_device_name(),backend=a.backend,cases={})
    for label,group,index in (("short",7,0),("long",1,2)):
        meta,ids,target=sample_args(a,tok,native,group,index)
        rows=[]
        try:
            warmup=one(model,ids,target)
            for _ in range(2):rows.append(one(model,ids,target))
            result["cases"][label]=dict(**meta,warmup=warmup,runs=rows)
        except torch.OutOfMemoryError as exc:
            result["cases"][label]=dict(**meta,error=str(exc),runs=rows)
        gc.collect()
        torch.cuda.empty_cache()
        a.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("base","adapter","snapshot","old","output"):p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--backend",choices=["reference","fast"],required=True)
    main(p.parse_args())

"""Exercise the four Qwen3.5 fast kernels on CUDA and compare with the installed torch references."""
import argparse
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen


def dispatch(fn):
    while hasattr(fn,"__wrapped__"):
        closure=inspect.getclosurevars(fn).nonlocals
        if "is_new_implementation" in closure:
            impl=closure["implementation"]
            if not closure["is_new_implementation"]:
                raise RuntimeError(f"{fn.__name__} selected the fallback")
            return impl.__module__+"."+impl.__name__
        fn=fn.__wrapped__
    raise RuntimeError("Missing Qwen kernel dispatch wrapper")


def relative(actual,expected):
    a,b=actual.float(),expected.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite output")
    return ((a-b).norm()/b.norm().clamp_min(1e-8)).item()


def check():
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.05)
    funcs=[qwen.causal_conv1d_fn,qwen.causal_conv1d_update,qwen.torch_chunk_gated_delta_rule,qwen.torch_recurrent_gated_delta_rule]
    using={fn.__name__:dispatch(fn) for fn in funcs}
    rnd=lambda *shape:torch.randn(*shape,device="cuda",dtype=torch.bfloat16)
    x,w=rnd(1,64,128),rnd(64,4)
    def fwdbwd(fn, inputs, get):
        values=[s.clone().requires_grad_() for s in inputs]
        out=get(fn,values)
        grads=torch.autograd.grad(out.float().square().mean(),values)
        return [out.detach(),*[g.detach() for g in grads]]
    err={}
    for func,inputs,call in [
        (funcs[0],[x,w],lambda f,v:f(*v,activation="silu")),
        (funcs[2],[rnd(1,128,2,128) for _ in range(3)]+[-torch.rand(1,128,2,device="cuda"),torch.rand(1,128,2,device="cuda",dtype=torch.bfloat16)],
         lambda f,v:f(*v[:3],g=v[3],beta=v[4],use_qk_l2norm_in_kernel=True)[0])
    ]:
        got=fwdbwd(func,inputs,call)
        ref=fwdbwd(inspect.unwrap(func),inputs,call)
        err[func.__name__]=[relative(a,b) for a,b in zip(got,ref,strict=True)]
    state=rnd(1,64,4)
    one=rnd(1,64,1)
    ref_state,fast_state=state.clone(),state.clone()
    ref=inspect.unwrap(funcs[1])(one,ref_state,w,activation="silu")
    got=funcs[1](one,fast_state,w,activation="silu")
    err[funcs[1].__name__]=[relative(got,ref),relative(fast_state,ref_state)]
    q,k,v=[rnd(1,1,2,128) for _ in range(3)]
    kw=dict(g=-torch.rand(1,1,2,device="cuda"),beta=torch.rand(1,1,2,device="cuda",dtype=torch.bfloat16),
            initial_state=torch.zeros(1,2,128,128,device="cuda"),output_final_state=True,use_qk_l2norm_in_kernel=True)
    got=funcs[3](q,k,v,**kw)
    ref=inspect.unwrap(funcs[3])(q,k,v,**kw)
    err[funcs[3].__name__]=[relative(a,b) for a,b in zip(got,ref,strict=True)]
    if max(max(values) for values in err.values())>.05:
        raise ValueError(f"CUDA/reference relative error exceeds 5%: {err}")
    return dict(implementations=using,relative_l2=err,device=torch.cuda.get_device_name(),
                versions={k:version(k) for k in ("torch","triton","transformers","trl","causal-conv1d","fla-core","flash-linear-attention","einops")})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    a=p.parse_args()
    data=check()
    a.output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data),flush=True)

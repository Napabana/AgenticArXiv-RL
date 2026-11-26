import pytest
import torch

from repair_4090 import fit_turn


def test_oom_bisection_preserves_rows_and_discards_failed_rng_draws(monkeypatch):
    gpu=torch.Generator()
    monkeypatch.setattr(torch.cuda,"get_rng_state",gpu.get_state)
    monkeypatch.setattr(torch.cuda,"set_rng_state",gpu.set_state)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)

    def generate(model,tokenizer,contexts,budget,temperature):
        assert (model,tokenizer,budget,temperature)==("model","tok",1536,1.0)
        draws=(torch.rand(1).item(),torch.rand(1,generator=gpu).item())
        if len(contexts)>1:
            raise torch.OutOfMemoryError()
        return [(contexts[0],draws)]

    contexts=[[1],[2,3],[4],[5,6]]
    torch.manual_seed(42)
    gpu.manual_seed(1234)
    actual=fit_turn(generate,"model","tok",contexts,1536,1.0)
    torch.manual_seed(42)
    gpu.manual_seed(1234)
    assert actual==[generate("model","tok",[row],1536,1.0)[0] for row in contexts]


def test_irreducible_oom_propagates():
    def generate(*args,**kwargs):
        raise torch.OutOfMemoryError("single trajectory")
    with pytest.raises(torch.OutOfMemoryError,match="single trajectory"):
        fit_turn(generate,None,None,[[1]],1536,1.0)

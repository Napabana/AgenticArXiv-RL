"""Format contract, partitioning, admission and the actual pinned TRL training entry."""
import json
import sys
from types import SimpleNamespace

import pytest
from rl.format_control.task_generator import dataset, prompt
from rl.format_control.validator import parse, score, expected
from rl.format_control.runtime import ROOT, DATA, V1, digest, read, sample

ROW = {"record":{"a":True,"b":None,"c":"南\\|=","d":"","e":-4}, "order":["c","b","d","a","e"]}
GOLD = "@ROW|c=南\\\\\\|\\=|b=NA|d=|a=YES|e=-4\n@END"


@pytest.mark.parametrize("wrong", [
    GOLD.replace("a=YES","a=true"), GOLD.replace("b=NA","b=null"),
    GOLD.replace("|d=",""), GOLD.replace("|e=-4","|b=-4"),
    GOLD.replace("|d=|a=YES","|a=YES|d="), GOLD.replace("e=-4","e=-04"),
    GOLD.replace("\\=","="), GOLD.replace("\\=","\\x"), GOLD.replace("\\|","|"),
    GOLD.replace("南",""), GOLD.replace("@ROW","ROW"), GOLD[:-1], GOLD+"\n", "note\n"+GOLD,
    "~~~\n"+GOLD+"\n~~~", GOLD.replace("|d=","|d= "), GOLD.replace("e=-4","e=-4|"),
])
def test_strict(wrong):
    assert expected(ROW)==GOLD and score(GOLD,ROW)["strict"]==1
    assert score(wrong,ROW)==score(wrong,ROW) and score(wrong,ROW)["strict"]==0


def test_inputs_and_eval_pairs():
    from rl.format_control.evaluate import paired, SEEDS
    records = set()
    for split in ("train","dev","test"):
        rows = dataset(split)
        assert rows == dataset(split)
        for row in rows:
            key=json.dumps(row["record"],sort_keys=True)
            assert key not in records and prompt(row)
            records.add(key)
            gold=expected(row)
            assert score(gold,row)["strict"] and parse(gold)
    base=[dict(id="test-a",split="test",seed=s,strict=0) for s in SEEDS]
    _, summary=paired(base,[{**r,"strict":1} for r in base])
    assert summary["task_bootstrap_95"]==[1,1]
    with pytest.raises(ValueError):
        paired(base,base[::-1])


def test_format_trainer_cpu(tmp_path, monkeypatch):
    """Scripted draws isolate reward-to-gradient plumbing; this is not model-effectiveness evidence."""
    import torch
    import trl
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from rl.format_control import train_format_grpo as entry
    from rl.format_control.runtime import save, train_prompt_sha
    from format_control.run_probe import summarize

    rows=dataset("train")
    groups=[]
    for task in rows:
        for j in range(8):
            groups.append(dict(id=task["id"],group=j,std=.5,samples=[{"strict":x} for x in [0,0,1,1]]))
    gate=summarize(groups,rows)
    assert len(gate["eligible"])==24 and gate["informative_fraction"]==1
    assert not summarize([{**g,"std":0,"samples":[{"strict":0}]*4} for g in groups],rows)["passed"]
    raw=tmp_path/"probe_groups.jsonl"
    raw.write_text("".join(json.dumps(g)+"\n" for g in groups))
    save(tmp_path/"probe.json",dict(**gate,adapter_sha256=V1,train_sha256=digest(DATA/"train.jsonl"),
                                    train_prompt_sha256=train_prompt_sha(), raw_sha256=digest(raw)))
    vocab={t:i for i,t in enumerate(["<pad>","<|im_end|>",*sorted(pre_tokenizers.ByteLevel.alphabet())])}
    backend=Tokenizer(models.BPE(vocab=vocab,merges=[]))
    backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder=decoders.ByteLevel()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token="<pad>",eos_token="<|im_end|>")
    tok.chat_template="{{ messages[0].content }}"
    model=Qwen3_5ForCausalLM(Qwen3_5TextConfig(vocab_size=len(tok),num_hidden_layers=1,hidden_size=24,
          intermediate_size=48,num_attention_heads=2,num_key_value_heads=1,head_dim=12,layer_types=["full_attention"],
          eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
          rope_parameters={"rope_theta":10000.0,"partial_rotary_factor":.5}))
    policy=get_peft_model(model,LoraConfig(r=2,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    monkeypatch.setattr(entry,"load",lambda *a,**kw:(policy,tok))
    # Use the real sampler once, including EOS trimming and repeated seeded draws.
    import rl.format_control.runtime as rt
    with monkeypatch.context() as patch:
        patch.setattr(rt,"TOKENS",3)
        policy.eval()
        assert sample(policy,tok,rows[0],7,4)==sample(policy,tok,rows[0],7,4)
    original=trl.GRPOConfig
    def cpu_config(**kw):
        kw.update(use_cpu=True,bf16=False,optim="adamw_torch",gradient_checkpointing=False,disable_tqdm=True)
        return original(**kw)
    monkeypatch.setattr(trl,"GRPOConfig",cpu_config)
    by_prompt={prompt(r):r for r in rows}
    def samples(self, prompt_ids, *args, **kwargs):
        ids=[]
        for i, seq in enumerate(prompt_ids):
            task=by_prompt[tok.decode(seq)]
            ids.append(tok.encode(expected(task) if i%2 else "invalid")+[tok.eos_token_id])
        return ids,None,None
    monkeypatch.setattr(trl.GRPOTrainer,"_generate_single_turn",samples)
    monkeypatch.setattr(sys,"argv",["train","--base","base","--adapter","v1","--probe",str(tmp_path/"probe.json"),
                                    "--output",str(tmp_path/"run")])
    entry.main()
    trained=read(tmp_path/"run/training_manifest.json")
    assert trained["effective_samples"]==100 and trained["steps"]==25 and trained["clipped"]==0
    assert trained["reference"]["reference_frozen"] and trained["adapter_sha256"]!=V1

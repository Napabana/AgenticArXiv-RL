import json
from types import SimpleNamespace

import pytest
from trl import GRPOConfig
from rl.credit_v2_trainer import CreditV2Trainer
from repair_loss25 import MatchedTrainer, SaveStep25


def test_loss_changes_without_changing_grouping_or_learning_rate_horizon(tmp_path, monkeypatch):
    cfg=GRPOConfig(output_dir=str(tmp_path),use_cpu=True,bf16=False,per_device_train_batch_size=1,
                   gradient_accumulation_steps=4,num_generations=4,max_steps=25,learning_rate=5e-6,beta=.04,loss_type="dapo")
    def init(self,*,args):
        self.loss_type=args.loss_type
        assert args.generation_batch_size==cfg.generation_batch_size==4
        assert args.steps_per_generation==cfg.steps_per_generation==4
    monkeypatch.setattr(CreditV2Trainer,"__init__",init)
    monkeypatch.setattr(CreditV2Trainer,"create_scheduler",lambda self,horizon,optimizer:horizon)
    trainer=MatchedTrainer(args=cfg)
    assert trainer.loss_type=="grpo" and cfg.loss_type=="dapo"
    assert trainer.create_scheduler(25)==72
    assert json.loads((tmp_path/"effective_optimizer.json").read_bytes())["loss_type"]=="grpo"
    with pytest.raises(ValueError):
        trainer.create_scheduler(24)


def test_extra_snapshot_runs_only_at_25(tmp_path):
    class Model:
        def save_pretrained(self,path):
            (path/"adapter_model.safetensors").write_bytes(b"test fixture")
    args=SimpleNamespace(output_dir=str(tmp_path))
    callback=SaveStep25()
    for step in (24,25,26):
        callback.on_step_end(args,SimpleNamespace(global_step=step,is_world_process_zero=True),None,model=Model())
        assert (tmp_path/"paired-step-25/identity.json").exists()==(step>=25)

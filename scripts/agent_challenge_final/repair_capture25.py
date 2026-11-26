"""Add a read-only adapter snapshot at step 25 of the frozen 72-step DAPO training."""
from pathlib import Path
import runpy
from unittest.mock import patch

from repair_loss25 import SaveStep25
from rl import credit_v2_trainer


class CaptureTrainer(credit_v2_trainer.CreditV2Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.loss_type!="dapo" or self.args.max_steps!=72:
            raise ValueError("Extra step-25 snapshot is only for the frozen 72-step DAPO arm")
        self.add_callback(SaveStep25())


if __name__=="__main__":
    with patch.object(credit_v2_trainer,"CreditV2Trainer",CaptureTrainer):
        runpy.run_path(str(Path(__file__).with_name("repair_4090.py")),run_name="__main__")

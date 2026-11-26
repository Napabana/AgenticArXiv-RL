from types import SimpleNamespace

from transformers import TrainerControl, TrainerState

from repair_matched_checkpoints import CheckpointSchedule, POLICIES, STEPS


def test_checkpoint_schedule_after_default_flow():
    callback = CheckpointSchedule()
    saved = []
    for step in range(1, 73):
        # Exercise both overriding an earlier save and requesting a new one.
        control = TrainerControl(should_save=step % 24 == 0, should_training_stop=step == 72)
        state = TrainerState(global_step=step)
        assert callback.on_step_end(SimpleNamespace(), state, control) is control
        assert control.should_training_stop == (step == 72)
        if control.should_save:
            saved.append(step)
    assert saved == list(STEPS) == [25, 48, 72]
    assert POLICIES == ("SFT", "GRPO25", "25", "48", "72")

"""Run frozen E2 on a 4090; bisect only a generation batch that exhausts CUDA memory."""
import gc
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def fit_turn(generate, model, tokenizer, contexts, budget, temperature):
    # A failed batched prefill must not consume random draws for its retries.
    if len(contexts) < 2:
        return generate(model, tokenizer, contexts, budget, temperature=temperature)
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    try:
        return generate(model, tokenizer, contexts, budget, temperature=temperature)
    except torch.OutOfMemoryError:
        pass
    # Release the unwound exception frames before asking the allocator to free memory.
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    middle = len(contexts) // 2
    print(json.dumps(dict(event="prefill_microbatch_retry", widths=[len(r) for r in contexts],
                          budget=budget, split=[middle,len(contexts)-middle])), file=sys.stderr, flush=True)
    return [*fit_turn(generate, model, tokenizer, contexts[:middle], budget, temperature),
            *fit_turn(generate, model, tokenizer, contexts[middle:], budget, temperature)]


if __name__ == "__main__":
    from functools import partial
    import run_native_grpo
    run_native_grpo.generate_native_turn = partial(fit_turn, run_native_grpo.generate_native_turn)
    runpy.run_path(str(Path(__file__).with_name("repair_run.py")), run_name="__main__")

import argparse
from pathlib import Path

from build_dataset import build
from rl.format_control.runtime import TOKENS, rendered, split_rows
from rl.format_control.validator import expected

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    build(check=True)
    if args.base:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
        lengths = []
        for split in ("train", "dev", "test"):
            for row in split_rows(split):
                text = expected(row)
                ids = tokenizer.encode(text, add_special_tokens=False)
                assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text
                assert "<tool_call>" not in rendered(tokenizer, row)
                lengths.append(len(ids) + 1)
        assert max(lengths) < TOKENS
        print(f"QWEN_TEMPLATE_PASS {len(lengths)} targets; max gold tokens with EOS={max(lengths)}/{TOKENS}")

"""Verify the inputs used to produce the 83/93 Native SFT v2 baseline without loading model weights."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from audit_snapshot import DATA, REPORT, ROOT, write
from native_sft_v2_gpu import freeze_protocol
from generate_parametric_sft_data import sha256_file


def text_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def verify(base, adapter, snapshot):
    from transformers import AutoTokenizer
    from agents import prompt_templates
    from agents.native_tool_agent import NativeToolCallingAgent
    from rl.format_control.runtime import verify_assets
    from tools.bootstrap import require_all_tools
    from rl.native_grpo_rollout import NativeRollout
    from trl import GRPOConfig
    from importlib.metadata import version

    train = ROOT / "data/sft/native_sft_v2/native_sft_v2_train.jsonl"
    manifest_file = train.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_file.read_bytes())
    trained = json.loads((ROOT / "reports/native_sft_v2/training_manifest.json").read_bytes())
    verify_assets(base, adapter, trained["adapter_sha256"])
    require_all_tools("Challenge asset audit")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    default_prompt = text_hash(prompt_templates.NATIVE_TOOL_SYSTEM_PROMPT)
    freeze_protocol()
    with TemporaryDirectory() as tmp:
        ep = NativeRollout(snapshot).episode({"id":"audit","task":"协议核验"},Path(tmp))
        messages = ep.prompt
        schema = [t["function"] for t in ep.tools]
    checked = dict(snapshot_sha256=sha256_file(snapshot), tokenizer_sha256=sha256_file(base/"tokenizer.json"),
                   chat_template_sha256=text_hash(tokenizer.chat_template), system_prompt_sha256=text_hash(messages[0]["content"]),
                   tools_sha256=text_hash(json.dumps(schema,ensure_ascii=False,sort_keys=True)))
    if len(schema)!=10 or any(val != manifest[key] for key,val in checked.items()):
        raise ValueError("Effective Native protocol differs from SFT v2")
    if (sha256_file(train)!=trained["data_sha256"] or sha256_file(manifest_file)!=trained["data_manifest_sha256"]
            or sha256_file(adapter/"adapter_config.json")!=trained["adapter_config_sha256"]):
        raise ValueError("SFT v2 data or Adapter config mismatch")
    loss=GRPOConfig.__dataclass_fields__["loss_type"].default
    if version("trl")!="1.17.1" or loss!="dapo":
        raise ValueError("TRL version or default loss differs from earlier Credit v2")
    return dict(**checked, adapter_sha256=trained["adapter_sha256"],data_sha256=trained["data_sha256"],
                data_manifest_sha256=trained["data_manifest_sha256"], default_module_prompt_sha256=default_prompt,
                effective_prompt_source="native_sft_v2_gpu.freeze_protocol",
                visible_setup_function_sha256=text_hash(inspect.getsource(prompt_templates.build_visible_setup_context)),
                native_setup_function_sha256=text_hash(inspect.getsource(NativeToolCallingAgent.setup_context)),
                tool_names=[t["name"] for t in schema], trl=version("trl"), historical_default_loss=loss)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base","adapter","snapshot"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    out=verify(a.base,a.adapter,a.snapshot)
    write(REPORT/"asset_identity.json",out)
    print(json.dumps(out,ensure_ascii=False))

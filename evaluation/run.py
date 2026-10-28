import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from euroeval import Benchmarker
from euroeval.enums import GenerativeType
from huggingface_hub import snapshot_download

EVALUATION_DIR = Path(__file__).parent
TRAINING_DIR = EVALUATION_DIR.parent
CUDA_BIN = next(
    path.parent
    for path in Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu*/bin/nvcc")
)
os.environ["PATH"] = f"{CUDA_BIN}:{os.environ['PATH']}"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_model(checkpoint: str, checkpoint_config: dict) -> str:
    model = checkpoint_config["model"]
    if subfolder := checkpoint_config.get("subfolder"):
        snapshot = snapshot_download(
            repo_id=model,
            local_dir=EVALUATION_DIR / "cache" / "models" / checkpoint,
            allow_patterns=f"{subfolder}/*",
        )
        return str(Path(snapshot) / subfolder)
    local_path = TRAINING_DIR / model
    if local_path.exists():
        model_path = local_path.resolve()
    elif template_path := checkpoint_config.get("chat_template_path"):
        model_path = Path(
            snapshot_download(
                repo_id=model,
                local_dir=EVALUATION_DIR / "cache" / "models" / checkpoint,
            )
        )
        tokenizer_config_path = model_path / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_config["chat_template"] = (TRAINING_DIR / template_path).read_text()
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    else:
        return model
    return str(model_path)


def evaluate(checkpoint: str, config: dict) -> None:
    model = resolve_model(checkpoint, config["checkpoints"][checkpoint])
    output_dir = EVALUATION_DIR / config["results_dir"] / "raw" / checkpoint

    common = dict(
        dataset=config["datasets"],
        language=config["language"],
        num_iterations=config["num_iterations"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        attention_backend=config["attention_backend"],
        cache_dir=str(EVALUATION_DIR / "cache"),
        save_results=True,
        raise_errors=True,
    )
    for use_bpc, model_type in (
        (False, GenerativeType.INSTRUCTION_TUNED),
        (True, GenerativeType.BASE),
    ):
        mode_dir = output_dir / ("bpc" if use_bpc else "task")
        mode_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(mode_dir)
        for test_split in (False, True):
            Benchmarker(**common).benchmark(
                model=model,
                evaluate_test_split=test_split,
                few_shot=True,
                generative_type=model_type,
                use_bits_per_character=use_bpc,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config", type=Path, default=EVALUATION_DIR / "config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint not in config["checkpoints"]:
        parser.error(f"unknown checkpoint: {args.checkpoint}")
    evaluate(args.checkpoint, config)

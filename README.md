# Dyna-instruct experiments

Full-parameter instruction tuning of Gemma 3 1B on Faroese and Icelandic
DynaWord-derived instruction data and FLAN datasets. Experiments use both
`google/gemma-3-1b-it` and `google/gemma-3-1b-pt`.
`train.py` wires the YAML config to TRL, which handles dataset preparation, training,
evaluation, checkpoints, and W&B logging.

## Experiments

| Experiment | Language | Starting model | Data | Report |
| --- | --- | --- | --- | --- |
| [1](configs/experiment1.yaml) | Faroese | Gemma 3 1B IT | Instruct + selected FLAN sources | [Results](experiment/experiment.md) |
| [2](configs/experiment2.yaml) | Icelandic | Gemma 3 1B IT | Instruct + selected FLAN sources | [Results](experiment/experiment2.md) |
| [3](configs/experiment3.yaml) | Icelandic | Gemma 3 1B IT | Instruct + full FLAN | [Results](experiment/experiment3.md) |
| [4](configs/experiment4.yaml) | Faroese | Gemma 3 1B IT | Instruct + full FLAN | [Results](experiment/experiment4.md) |
| [5](configs/experiment5.yaml) | Faroese | Gemma 3 1B PT | Same data as experiment 1 | [Results](experiment/experiment5.md) |
| [6](configs/experiment6.yaml) | Icelandic | Gemma 3 1B PT | Same data as experiment 2 | [Results](experiment/experiment6.md) |

## Setup

Use Python 3.12 or newer and a CUDA GPU environment. The experiment launchers
assume two GPUs; the experiment reports describe the original hardware and settings.

Accept the [Gemma license](https://huggingface.co/google/gemma-3-1b-it), then run:

```bash
git clone https://github.com/V4ldeSalnikov/Dyna-instruct-experiments.git
cd Dyna-instruct-experiments
uv sync
uv run hf auth login
uv run wandb login
```

Authenticate locally with these commands. Keep API keys and credentials out of
source files and YAML configs; `.gitignore` excludes common secret and cache files.
The numbered experiment configs enable Hugging Face uploads, so review their
`hub_model_id`, `push_to_hub`, and W&B project settings before running them.

## Train

```bash
uv run python train.py --config configs/sft.yaml
```

Train on both user and assistant tokens instead of masking the user instruction:

```bash
uv run python train.py --config configs/sft.yaml --assistant_only_loss false
```

Resume from a checkpoint:

```bash
uv run python train.py --config configs/sft.yaml \
  --resume_from_checkpoint outputs/gemma-3-1b-it-sft/checkpoint-STEP
```

The config creates a 95/5 train/evaluation split and trains only on assistant tokens.
W&B records loss, evaluation loss, entropy, token accuracy, learning rate, gradient norm,
epoch, and processed tokens. Checkpoints remain under `outputs/`.

For an individual experiment, select its config, for example:

```bash
uv run accelerate launch --num_processes 2 train.py --config configs/experiment1.yaml
```

## Evaluation and saved results

See [evaluation/README.md](evaluation/README.md) for EuroEval commands. Aggregate
CSV metrics and plots are included under `evaluation/results/`, alongside the
experiment reports. Raw evaluation records, logs, model weights, virtual environments,
and caches are excluded from this repository. Run evaluation to regenerate raw records
before rerunning the summarizer.

Some evaluation configs refer to locally generated checkpoints under `outputs/`;
train those experiments first or update the checkpoint locations. The
`run_full_experiments.sh` launcher also expects the baseline raw evaluation records
from experiments 1 and 2. `run_experiments_5_6.sh` runs the pretrained-model experiments
and their evaluations.

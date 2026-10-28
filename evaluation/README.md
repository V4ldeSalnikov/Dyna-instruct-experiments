# Faroese EuroEval

`run.py` evaluates one checkpoint on the validation and full test splits. It runs the
normal task metric in instruction-tuned mode and diagnostic BPC in base-decoder mode.
The latter is required because EuroEval does not expose BPC for instruction-tuned mode.

The seven datasets in `config.yaml` support both modes. Chat-only datasets are excluded
because EuroEval cannot compute BPC for them. Test-split task scores use the same split,
few-shot setting, ten iterations, prompts, and metrics as the official leaderboard.

```bash
uv run python evaluation/run.py checkpoint-82
uv run python evaluation/summarize.py

uv run python evaluation/run.py checkpoint-285 --config evaluation/experiment2.yaml
uv run python evaluation/summarize.py --config evaluation/experiment2.yaml
```

The runner writes raw JSONL. The summarizer writes long and wide CSV tables plus
progress plots with EuroEval uncertainty bars to `evaluation/results/`.

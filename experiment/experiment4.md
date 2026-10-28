# Experiment 4: Full Faroese FLAN SFT

## Objective

Repeat the full-FLAN experiment in Faroese after Experiment 3 and its
evaluation complete.

## Data

| Dataset | Configuration | Raw rows |
| --- | --- | ---: |
| `V4ldeLund/Faroese-Instruct` | default | 8,609 |
| `Frejams/faroese-flan` | default (all 13 component sources) | 1,000,543 |
| **Total** | | **1,009,152** |

No FLAN source is excluded. The dataset's `default` configuration is its full
union, so the component configurations are not loaded separately. Examples
whose assistant response is entirely beyond the 2,048-token window are still
dropped by the standard SFT preprocessing.

## Training

- Full-parameter SFT on two B200 GPUs
- Same Gemma 3 1B model, global batch size 64, BF16, assistant-only loss, and
  optimizer settings as the earlier experiments
- One epoch and 15,724 optimizer steps after preprocessing, with 5% linear
  warmup followed by cosine decay
- Checkpoints at steps 5,000, 10,000, and 15,000 plus the final model
- No validation split; evaluation uses EuroEval
- W&B run: `gemma-3-1b-it-faroese-full-experiment4`
- Hugging Face repository: `V4ldeLund/gemma-3-1b-it-faroese-full-experiment4`

## Result

Completed on 2026-08-30.

- W&B run: `90q0fwaz`
- Hugging Face revision: `e98eeda53e05a7b8a8427a962a07a83bd69291ed`
- Training time: 9,097 seconds
- Throughput: 110.6 samples/second and 1.728 optimizer steps/second
- Tokens processed: 65.06 million
- Training loss: 0.701
- Assistant-token accuracy: 85.98%

All 15,724 steps and the three planned checkpoints completed successfully.

## EuroEval evaluation

The base model, steps 5,000, 10,000, and 15,000, and the final model were scored
on the Faroese validation and full test splits. Every dataset received both its
normal task metric and diagnostic BPC.

### Test results

| Dataset | Task metric | Base | Final | Best | BPC base | BPC final | Lowest BPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FoSent | MCC | 19.60 | 21.59 | 22.42 (step 15,000) | 0.3968 | 0.3476 | 0.3275 (step 5,000) |
| ScaLA-fo | MCC | 1.39 | -0.30 | 1.39 (base) | 0.9186 | 0.6855 | 0.6690 (step 10,000) |
| FoNE | micro F1, no MISC | 37.50 | 37.01 | 37.50 (base) | 1.1884 | 0.5281 | 0.5278 (step 15,000) |
| FoQA | F1 | 22.55 | 33.80 | 33.80 (final) | 1.2107 | 0.7378 | 0.7375 (step 15,000) |
| WikiANN-fo | micro F1, no MISC | 34.56 | 36.01 | 36.02 (step 15,000) | 1.4447 | 0.6413 | 0.6389 (step 10,000) |
| MultiWikiQA-fo | F1 | 21.91 | 26.88 | 29.03 (step 5,000) | 1.3049 | 0.9947 | 0.9550 (step 5,000) |
| GerLangMod-fo | micro F1 | 10.37 | 7.71 | 10.37 (base) | 3.3240 | 2.3757 | 2.3743 (step 15,000) |

BPC improves over the base model on all seven datasets. Task scores improve
substantially on FoQA (+11.25), MultiWikiQA (+4.97), FoSent (+2.00), and
WikiANN (+1.45). FoNE is approximately flat, while ScaLA and GerLangMod regress.

Compared with the curated Experiment 1 final model, full-FLAN is better on
FoQA, MultiWikiQA, and WikiANN, but worse on FoSent, FoNE, and GerLangMod. The
full mixture therefore adds useful QA coverage but does not replace the curated
mixture as a general Faroese instruction-tuning set.

Step 15,000 is the strongest single full-mixture checkpoint overall, although
MultiWikiQA peaks much earlier at step 5,000. Structured-output behavior is
mixed: final FoNE failures fall from 48 for the base model to 9, while
GerLangMod failures rise from 189 to 422. Full CSV tables and plots are in
`evaluation/results/experiment4/`.

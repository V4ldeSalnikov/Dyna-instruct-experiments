# Experiment 1: Faroese SFT

## Objective

Test full-parameter Faroese instruction tuning of `google/gemma-3-1b-it` on
two B200 GPUs before introducing Icelandic or the large lexical FLAN tasks.

## Data

| Dataset | Configuration | Rows |
| --- | --- | ---: |
| `V4ldeLund/Faroese-Instruct` | default | 8,609 |
| `Frejams/faroese-flan` | `lum` | 1,265 |
| `Frejams/faroese-flan` | `logting_spurningar` | 213 |
| `Frejams/faroese-flan` | `fpsc` | 5,163 |
| `Frejams/faroese-flan` | `fo_wikipedia` | 355 |
| `Frejams/faroese-flan` | `kunngerdaportalur` | 2,404 |
| **Total** | | **18,009** |

Dataset revisions observed when the experiment was prepared:

- `V4ldeLund/Faroese-Instruct`: `c1f5cea1df16535c5df452c6a8d85a8be1640e61`
- `Frejams/faroese-flan`: `c120626bf4bc41ef8120c5b71a4c4772683c198a`

The FLAN configurations were selected for composed or institutional Faroese.
The first experiment excludes the high-volume pronunciation, word-class,
morphology, translation, and diacritic-restoration tasks so that short answers
do not dominate the run. Faroese Wikipedia overlaps with one DynaWord source,
but is retained because FLAN supplies a real article-to-lead summarization pair.

## Training

- Full-parameter SFT on two B200 GPUs with Accelerate DDP
- One epoch, 248 optimizer steps at global batch size 64 after preprocessing
- Per-device batch size 32, no gradient accumulation
- BF16 parameters, TF32 matrix operations, and gradient checkpointing
- Maximum sequence length 2,048; no packing
- Loss on assistant tokens only, using a Gemma 3 template with explicit assistant boundaries
- AdamW with learning rate `2e-5`, cosine decay, 12 warmup steps (5%), and weight decay `0.01`
- No validation split; EuroEval will be run after training
- W&B run: `gemma-3-1b-it-faroese-experiment1`
- Hugging Face repository: `V4ldeLund/gemma-3-1b-it-faroese-experiment1`

Model-only checkpoints are saved at steps 82, 164, and 246, followed by the
final model at step 248. They are suitable for evaluation but do not contain
optimizer state for resuming training.

## Result

Completed on 2026-08-29.

- W&B run: `mqgrbpd3`
- Hugging Face revision: `d3519e917187c8e6e33328d29b6fc81bd79741bf`
- Training time: 412 seconds
- Throughput: 38.47 samples/second and 0.602 optimizer steps/second
- Tokens processed: 8,269,596
- Training loss: 2.879
- Assistant-token accuracy: 51.17%

The initial launch without gradient checkpointing ran out of GPU memory before
the first optimizer step. Enabling it reduced memory use to about 26 GB per GPU
while preserving the global batch size of 64. The successful run then completed
all 248 steps, and the three planned checkpoints plus the final model were
verified on the Hugging Face Hub.

## EuroEval evaluation

Evaluated with EuroEval 17.7.0 on the Faroese validation and full test splits.
All runs used few-shot prompting and 10 iterations. The evaluated model states
were the base model (step 0), checkpoints 82, 164, and 246, and the final model
(step 248).

Seven datasets support both normal task scoring and BPC: FoSent, ScaLA-fo,
FoNE, FoQA, WikiANN-fo, MultiWikiQA-fo, and GerLangMod-fo. Normal metrics use
Gemma as an instruction-tuned decoder. BPC is a diagnostic base-decoder view of
the same weights because EuroEval does not support BPC in instruction-tuned
mode. It is therefore useful for comparing these checkpoints, but it is not an
official leaderboard score.

### Test results

| Dataset | Task metric | Base | Final | Best | BPC base | BPC final | Lowest BPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FoSent | MCC | 19.60 | 29.42 | 30.10 (step 164) | 0.3968 | 0.3054 | 0.3048 (step 164) |
| ScaLA-fo | MCC | 1.39 | -0.39 | 1.39 (base) | 0.9186 | 1.0551 | 0.9186 (base) |
| FoNE | micro F1, no MISC | 37.50 | 41.41 | 41.58 (step 164) | 1.1884 | 0.5725 | 0.5650 (step 82) |
| FoQA | F1 | 22.55 | 28.14 | 28.34 (step 164) | 1.2107 | 0.8250 | 0.8250 (final) |
| WikiANN-fo | micro F1, no MISC | 34.56 | 34.76 | 34.76 (final) | 1.4447 | 0.6882 | 0.6840 (step 82) |
| MultiWikiQA-fo | F1 | 21.91 | 24.26 | 24.37 (step 164) | 1.3049 | 0.9761 | 0.9745 (step 164) |
| GerLangMod-fo | micro F1 | 10.37 | 10.13 | 10.37 (base) | 3.3240 | 2.2222 | 2.2222 (final) |

Instruction tuning improves the final test score over the base model on FoSent
(+9.82 points), FoQA (+5.59), FoNE (+3.90), MultiWikiQA (+2.35), and WikiANN
(+0.20). It slightly reduces GerLangMod (-0.24) and clearly hurts ScaLA
(-1.78). Test BPC improves on six of seven datasets; ScaLA is again the sole
regression.

Most BPC improvement is already present at step 82. Task performance generally
peaks at step 164: it is best on FoSent, FoNE, FoQA, and MultiWikiQA, while later
training adds no consistent gain. The final checkpoint only establishes the
best WikiANN task score and the lowest FoQA and GerLangMod BPC values, all by
small margins.

Validation and test agree on the large FoNE, FoQA, and MultiWikiQA gains and on
the ScaLA regression. They disagree for FoSent and WikiANN: FoSent validation
falls by 2.03 points while test rises by 9.82, and WikiANN validation falls by
1.84 while test is nearly flat. The plotted uncertainty intervals overlap for
many nearby checkpoints, so step-164 versus later differences should not be
treated as statistically decisive.

EuroEval also reports invalid generated outputs on the structured tasks. For
the final checkpoint, the aggregate full-test counts are 44 for FoNE, 16 for
WikiANN, and 200 for GerLangMod; the other four task datasets and every BPC run
have zero. These counts must be considered alongside the task scores because
the structured evaluations also depend on format following. The complete
counts are kept in the CSV tables.

For the next experiment, step 164 is the most sensible tuned checkpoint to
carry forward, with ScaLA, structured-output validity, and the split
disagreements treated as explicit regression checks.

Full tables, uncertainty values, raw EuroEval records, and plots are in
`evaluation/results/`.

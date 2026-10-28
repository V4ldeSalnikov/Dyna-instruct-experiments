# Experiment 2: Icelandic SFT

## Objective

Repeat the first experiment in Icelandic: full-parameter instruction tuning of
`google/gemma-3-1b-it` on two B200 GPUs, without mixing in Faroese.

## Data

| Dataset | Configuration | Raw rows | Usable at 2,048 tokens |
| --- | --- | ---: | ---: |
| `V4ldeLund/Icelandic-Instruct` | default | 8,108 | 8,108 |
| `Frejams/Icelandic-Flan` | `igc_news` | 27,711 | 27,314 |
| `Frejams/Icelandic-Flan` | `ejournals_is` | 695 | 467 |
| `Frejams/Icelandic-Flan` | `eso` | 4,181 | 4,181 |
| `Frejams/Icelandic-Flan` | `umbodsmadur` | 3,914 | 2,165 |
| `Frejams/Icelandic-Flan` | `hagstofa` | 6,174 | 6,065 |
| `Frejams/Icelandic-Flan` | `stjornartidindi` | 6,789 | 6,779 |
| **Total** | | **57,572** | **55,079** |

Dataset revisions observed when the experiment was prepared:

- `V4ldeLund/Icelandic-Instruct`: `500a3a0d0c27bcce81a437df74892ae368f9930c`
- `Frejams/Icelandic-Flan`: `157af28dcb761d914687275176535a069886470f`

The FLAN subset contains news summarization, academic abstracts, popular-science
translation, administrative reasoning, official statistics, and legislation.
Lexical, morphology, hyphenation, diacritic-restoration, and transcript-repair
sources are excluded so short or copying-based tasks do not dominate training.
The 61 `sedlabanki` rows are also excluded because every assistant response falls
beyond the 2,048-token boundary and would be dropped during preprocessing.

## Training

- Same model and optimization settings as Experiment 1
- Full-parameter SFT on two B200 GPUs with Accelerate DDP
- One epoch and an expected 861 optimizer steps at global batch size 64
- Per-device batch size 32, no gradient accumulation
- BF16 parameters, TF32 matrix operations, and gradient checkpointing
- Maximum sequence length 2,048; no packing
- Loss on assistant tokens only
- AdamW with learning rate `2e-5`, cosine decay, 43 warmup steps, and weight decay `0.01`
- No validation split; EuroEval will be run after training
- Planned checkpoints at steps 285, 570, and 855
- W&B run: `gemma-3-1b-it-icelandic-experiment2`
- Hugging Face repository: `V4ldeLund/gemma-3-1b-it-icelandic-experiment2`

## Result

Completed on 2026-08-29.

- W&B run: `9tm8dv6g`
- Hugging Face revision: `0fa045d4c5434b67abbef0739a5ab154b1a06e08`
- Training time: 1,431 seconds
- Throughput: 38.49 samples/second and 0.602 optimizer steps/second
- Tokens processed: 39,930,000
- Training loss: 2.120
- Assistant-token accuracy: 59.52%

All 861 steps completed, and checkpoints 285, 570, and 855 plus the final
model were verified on the Hugging Face Hub.

## EuroEval evaluation

Evaluated with EuroEval 17.7.0 on the eight official Icelandic datasets
that provide both validation and test splits: Hotter and Colder Sentiment,
ScaLA-is, MIM-GOLD-NER, NQiI, RRN, Icelandic Knowledge, Winogrande-is, and
ZebraPuzzlesEasy-is. Each checkpoint received its normal task metric in
instruction-tuned mode and diagnostic BPC in base-decoder mode.

### Test results

| Dataset | Task metric | Base | Final | Best | BPC base | BPC final | Lowest BPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hotter and Colder Sentiment | MCC | 7.90 | 3.19 | 7.90 (base) | 0.3756 | 0.3235 | 0.3122 (step 285) |
| ScaLA-is | MCC | -0.16 | -0.13 | -0.13 (step 285) | 1.3639 | 1.0871 | 1.0272 (step 285) |
| MIM-GOLD-NER | micro F1, no MISC | 14.54 | 20.83 | 21.79 (step 285) | 0.8384 | 0.4506 | 0.4081 (step 285) |
| NQiI | F1 | 29.93 | 29.50 | 29.93 (base) | 0.7674 | 0.6103 | 0.6099 (step 855) |
| RRN | chrF-3++ | 29.04 | 21.84 | 29.04 (base) | 1.8590 | 1.0688 | 1.0688 (final) |
| Icelandic Knowledge | MCC | 1.58 | 0.25 | 1.85 (step 285) | 3.4274 | 2.1069 | 2.1069 (final) |
| Winogrande-is | MCC | -0.60 | 1.44 | 1.74 (step 570) | 2.2124 | 1.1882 | 1.1882 (final) |
| ZebraPuzzlesEasy-is | cell accuracy | 35.29 | 27.67 | 35.29 (base) | 3.6141 | 2.3465 | 2.3465 (step 855/final) |

BPC improves over the base model on every test dataset. The largest absolute
reductions are on Icelandic Knowledge, ZebraPuzzlesEasy, Winogrande, and RRN.
Most of this gain is already present at step 285; later checkpoints mainly make
small additional BPC improvements on knowledge, summarization, and reasoning.

Normal task performance is mixed. MIM-GOLD-NER improves by 6.29 points and
Winogrande MCC by 2.04 from base to final. NQiI and ScaLA are approximately
flat, while RRN, ZebraPuzzlesEasy, sentiment, and Icelandic Knowledge regress.
Validation results show the same broad pattern. Step 285 is the strongest tuned
checkpoint overall: it has the best NER and Icelandic Knowledge task scores and
the best BPC on sentiment, NER, and ScaLA.

One of the ten full-test sentiment iterations produced 1,081 invalid outputs
for every tuned checkpoint, while the other nine iterations and every
validation iteration had none. The sentiment test result should therefore be
treated cautiously and the exact prompt/output format inspected before using it
as a model-selection signal. NER formatting improves over the base model: test
failures fall from 39 to 12 at step 285 and 22 for the final model.

Complete CSV tables and plots are in `evaluation/results/experiment2/`; raw
EuroEval records remain on the evaluation node.

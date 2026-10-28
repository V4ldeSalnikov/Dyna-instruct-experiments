# Experiment 5: Faroese SFT from the pretrained checkpoint

## Objective

Repeat Experiment 1 from `google/gemma-3-1b-pt` instead of the instruction-tuned
checkpoint. This isolates the effect of the starting checkpoint.

## Setup

- Same 18,009 Faroese rows and dataset configurations as Experiment 1
- Same seeds, one epoch, optimizer, schedule, global batch size 64, and 2,048-token limit
- Same Gemma 3 chat template and assistant-only loss
- Planned checkpoints: steps 82, 164, and 246; expected final step: 248
- W&B run: `gemma-3-1b-pt-faroese-experiment5`
- Hugging Face repository: `V4ldeLund/gemma-3-1b-pt-faroese-experiment5`

## Result

Completed on 2026-08-30.

- W&B run: `nfn18wd5`
- Hugging Face revision: `2bfae254d14d549cf08d5ee65c800a7f9efb6fed`
- Training time: 411 seconds
- Throughput: 38.56 samples/second and 0.603 optimizer steps/second
- Tokens processed: 8,269,596
- Training loss: 2.448
- Assistant-token accuracy: 55.79%

All 248 steps completed. Checkpoints 82, 164, and 246 and the final model were
uploaded to the Hub.

## EuroEval evaluation

The base model, three checkpoints, and final model were evaluated on the same
seven Faroese datasets, validation and test splits, normal task metrics, and
BPC as Experiment 1. All 140 expected dataset-mode-split results completed.

### Test results

| Dataset | Task metric | PT base | PT final | Best PT | BPC base | BPC final | Lowest BPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FoSent | MCC | 5.89 | 10.74 | 10.74 (final) | 0.2700 | 0.2605 | 0.2601 (step 246) |
| ScaLA-fo | MCC | -0.06 | 0.00 | 0.00 (all tuned) | 0.6709 | 0.8653 | 0.6709 (base) |
| FoNE | micro F1, no MISC | 25.94 | 15.35 | 25.94 (base) | 0.5561 | 0.6356 | 0.5561 (base) |
| FoQA | F1 | 8.68 | 16.14 | 20.61 (step 82) | 0.6516 | 0.6071 | 0.6049 (step 164) |
| WikiANN-fo | micro F1, no MISC | 32.69 | 26.50 | 32.69 (base) | 0.6266 | 0.6975 | 0.6266 (base) |
| MultiWikiQA-fo | F1 | 7.77 | 17.90 | 19.69 (step 82) | 0.7095 | 0.6777 | 0.6760 (step 164) |
| GerLangMod-fo | micro F1 | 6.77 | 8.25 | 8.25 (final) | 1.9220 | 1.9180 | 1.9180 (final) |

Starting from PT and applying this small SFT improves the final task score on
FoSent, FoQA, MultiWikiQA, and GerLangMod. It harms both NER datasets, while
ScaLA collapses to a constant prediction and receives MCC 0. Step 82 is the
best tuned checkpoint for both QA tasks; later training partly erases those
gains.

The final PT-start model is substantially weaker than Experiment 1's IT-start
final model on every normal task except the already-broken ScaLA result. For
example, final FoNE is 15.35 versus 41.41, FoQA 16.14 versus 28.14, and FoSent
10.74 versus 29.42. One epoch over 18,009 examples is therefore not enough to
reconstruct Gemma's general instruction-following behavior from the pretrained
checkpoint.

BPC and task quality should not be conflated. The PT model has lower BPC than
the IT model on several datasets even when its task scores are far worse. PT
fine-tuning improves BPC on four datasets but regresses FoNE, WikiANN, and
ScaLA, matching the task regressions on those three datasets.

The final structured-task invalid-output counts are 65 for FoNE, 57 for
WikiANN, and 4 for GerLangMod. Complete tables, raw records, and plots are in
`evaluation/results/experiment5/`.

# Experiment 6: Icelandic SFT from the pretrained checkpoint

## Objective

Repeat Experiment 2 from `google/gemma-3-1b-pt` instead of the instruction-tuned
checkpoint. This isolates the effect of the starting checkpoint.

## Setup

- Same 57,572 raw Icelandic rows and dataset configurations as Experiment 2
- Same seeds, one epoch, optimizer, schedule, global batch size 64, and 2,048-token limit
- Same Gemma 3 chat template and assistant-only loss
- Planned checkpoints: steps 285, 570, and 855; expected final step: 861
- W&B run: `gemma-3-1b-pt-icelandic-experiment6`
- Hugging Face repository: `V4ldeLund/gemma-3-1b-pt-icelandic-experiment6`

## Result

Completed on 2026-08-30.

- W&B run: `rlxaj61z`
- Hugging Face revision: `36a9c518eed55e36db6ae5eb7e8046105edee011`
- Training time: 1,429 seconds
- Throughput: 38.53 samples/second and 0.602 optimizer steps/second
- Tokens processed: 39,930,000
- Training loss: 1.811
- Assistant-token accuracy: 63.19%

All 861 steps completed. Checkpoints 285, 570, and 855 and the final model
were verified on the Hub.

## EuroEval evaluation

Completed on 2026-08-31. The base model, three checkpoints, and final model
were evaluated on the same eight Icelandic datasets, validation and test
splits, normal task metrics, and BPC as Experiment 2. All 160 expected
dataset-mode-split results completed without a runtime error.

### Test results

| Dataset | Task metric | PT base | PT final | Best PT | BPC base | BPC final | Lowest BPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hotter and Colder Sentiment | MCC | 6.47 | 8.87 | 8.98 (step 855) | 0.2999 | 0.2893 | 0.2893 (step 855) |
| ScaLA-is | MCC | -0.21 | 0.83 | 1.05 (step 285) | 0.6672 | 0.7043 | 0.6672 (base) |
| MIM-GOLD-NER | micro F1, no MISC | 13.21 | 5.69 | 13.21 (base) | 0.3587 | 0.4007 | 0.3587 (base) |
| NQiI | F1 | 13.78 | 28.56 | 28.56 (final) | 0.4695 | 0.4953 | 0.4695 (base) |
| RRN | chrF-3++ | 15.32 | 22.37 | 22.43 (step 855) | 1.0514 | 0.9010 | 0.9010 (final) |
| Icelandic Knowledge | MCC | 0.22 | 0.93 | 1.34 (step 855) | 1.9419 | 1.8292 | 1.8290 (step 855) |
| Winogrande-is | MCC | 0.34 | 1.06 | 1.51 (step 855) | 0.9869 | 0.9565 | 0.9561 (step 855) |
| ZebraPuzzlesEasy-is | cell accuracy | 24.68 | 28.30 | 28.30 (final) | 2.6211 | 2.3339 | 2.3339 (step 855/final) |

The final model improves over the PT base on seven of eight normal task
metrics. The largest gains are NQiI (+14.78), RRN (+7.05), and Zebra (+3.61).
MIM-GOLD-NER is the clear failure: it falls from 13.21 to 5.69. BPC improves
on sentiment, RRN, Icelandic Knowledge, Winogrande, and Zebra, but regresses
on ScaLA, MIM-GOLD-NER, and NQiI. As in Experiment 5, BPC and instruction-task
quality do not move together consistently.

Step 855 is the strongest practical PT checkpoint. It has the best task score
on sentiment, RRN, Icelandic Knowledge, and Winogrande and the best or tied
BPC on five datasets. The final checkpoint is best on NQiI and Zebra, but most
differences from step 855 are small.

Compared with Experiment 2's IT-start final model, the PT-start final model is
better on sentiment, RRN, Icelandic Knowledge, ScaLA, and Zebra, but worse on
MIM-GOLD-NER, NQiI, and Winogrande. The NER gap is large: 5.69 for PT-start
versus 20.83 for IT-start. PT-start has lower final BPC on all eight datasets,
which reinforces that lower answer likelihood loss is not sufficient evidence
of better instruction following.

Structured-output validity is a serious caveat. The PT final checkpoint has
180 failed MIM-GOLD-NER test instances and 31 validation failures. It also has
624 failed ScaLA test instances, so the small positive ScaLA MCC should not be
used as a model-selection signal. The other final task evaluations have no
failed instances.

Complete tables, raw records, and plots are in
`evaluation/results/experiment6/`.

# Experiment 3: Full Icelandic FLAN SFT

## Objective

Measure what changes when Experiment 2 is expanded to every Icelandic-FLAN
source while keeping the model and optimization setup fixed.

## Data

| Dataset | Configuration | Raw rows |
| --- | --- | ---: |
| `V4ldeLund/Icelandic-Instruct` | default | 8,108 |
| `Frejams/Icelandic-Flan` | default (all 16 component sources) | 602,057 |
| **Total** | | **610,165** |

No FLAN source is excluded. The dataset's `default` configuration is its full
union, so the component configurations are not loaded separately. Examples
whose assistant response is entirely beyond the 2,048-token window are still
dropped by the standard SFT preprocessing.

## Training

- Full-parameter SFT on two B200 GPUs
- Same Gemma 3 1B model, global batch size 64, BF16, assistant-only loss, and
  optimizer settings as Experiment 2
- One epoch and 9,494 optimizer steps after preprocessing, with 5% linear
  warmup followed by cosine decay
- Checkpoints at steps 3,000, 6,000, and 9,000 plus the final model
- No validation split; evaluation uses EuroEval
- W&B run: `gemma-3-1b-it-icelandic-full-experiment3`
- Hugging Face repository: `V4ldeLund/gemma-3-1b-it-icelandic-full-experiment3`

## Result

Completed on 2026-08-30.

- W&B run: `8l8aty1a`
- Hugging Face revision: `6fe7d6dba9883bd2af1488f55f187e4b6584267b`
- Training time: 8,667 seconds
- Throughput: 70.11 samples/second and 1.095 optimizer steps/second
- Tokens processed: 71.92 million
- Training loss: 1.416
- Assistant-token accuracy: 73.38%

All 9,494 steps and the three planned checkpoints completed successfully.

## EuroEval evaluation

The base model, steps 3,000, 6,000, and 9,000, and the final model were scored
on the Icelandic validation and full test splits. Every dataset received both
its normal task metric and diagnostic BPC.

### Test results

| Dataset | Task metric | Base | Final | Best | BPC base | BPC final | Lowest BPC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hotter and Colder Sentiment | MCC | 7.90 | -1.64 | 7.90 (base) | 0.3756 | 0.2160 | 0.2160 (final) |
| ScaLA-is | MCC | -0.16 | 0.00 | 0.56 (step 3,000) | 1.3639 | 0.6668 | 0.6661 (step 9,000) |
| MIM-GOLD-NER | micro F1, no MISC | 14.54 | 23.67 | 24.65 (step 6,000) | 0.8384 | 0.4651 | 0.4427 (step 3,000) |
| NQiI | F1 | 29.93 | 25.47 | 29.93 (base) | 0.7674 | 0.6862 | 0.6768 (step 3,000) |
| RRN | chrF-3++ | 29.04 | 21.69 | 29.04 (base) | 1.8590 | 0.9936 | 0.9936 (final) |
| Icelandic Knowledge | MCC | 1.58 | -0.25 | 1.58 (base) | 3.4274 | 1.9922 | 1.9922 (final) |
| Winogrande-is | MCC | -0.60 | 0.06 | 1.24 (step 6,000) | 2.2124 | 1.0587 | 1.0574 (step 9,000) |
| ZebraPuzzlesEasy-is | cell accuracy | 35.29 | 23.56 | 35.29 (base) | 3.6141 | 1.9129 | 1.9076 (step 6,000) |

BPC improves over the base model on all eight datasets. Compared with the
curated Experiment 2 final model, full-FLAN also improves BPC on six datasets,
but is worse on MIM-GOLD-NER and NQiI.

Normal task performance is much less favorable. NER improves by 9.13 points
over base, and ScaLA and Winogrande show small gains at intermediate
checkpoints. Sentiment, NQiI, RRN, Icelandic Knowledge, and Zebra reasoning all
regress. Compared with Experiment 2, the full mixture only clearly improves
NER. This is consistent with the high-volume lexical and copying tasks
improving Icelandic token prediction without providing equally useful
instruction-following practice.

Step 6,000 is the strongest full-mixture checkpoint for downstream tasks. NER
format failures nevertheless increase from 39 for the base model to 51 at step
6,000 and 74 for the final model. Full CSV tables and plots are in
`evaluation/results/experiment3/`.

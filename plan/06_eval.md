# Evaluation

## Primary metrics

- Balanced accuracy (near-50/50 real/fake by construction, but report balanced for robustness).
- AUROC on the single-logit output.
- Per-`k` (temporal stride) accuracy — does the detector rely on motion cues, or is it purely spatial?
- Confusion analysis: worst 100 `real→predicted_fake` and 100 `fake→predicted_real`, dumped with `clip_id` + source caption.

## Held-out generalization tests

Run these **after** the first training run lands. They define the narrowness of the detector.

1. **Unseen captions, same real dataset**: fresh 2k clips from OpenVid not used in train/val. Expected: matches val accuracy.
2. **Out-of-distribution real video**: 500 clips from Kinetics-700. Expected: accuracy holds for the "real" class.
3. **Different generator**: 500 clips from LTX-Video, generated from the same captions as test set 1. Expected: accuracy drops hard. This is the interesting number.

## Script

`src/eval.py`

Args:
- `--checkpoint runs/X/best.pt`
- `--manifest path/to/eval_manifest.parquet`
- `--out-csv path/to/predictions.csv`
- `--device`

Output:
- Prints metrics table.
- Writes per-clip predictions CSV: `clip_id, label, logit, prob, correct`.

## Optional viewer

Gradio or plain HTML page: misclassified clips side-by-side with captions and predictions. Build only if the confusion analysis turns up something interesting.

## Verification of the full pipeline

- **Smoke training** on MPS (50 steps) — loss decreases.
- **Param split sanity** at training startup — count of Muon params vs. AdamW params both non-zero; sum equals total param count.
- **One-clip overfit** — train on a single batch of 8 triplets for 200 steps; loss should drop near zero. If not, the model or tubelet embedding is wrong.
- **Preprocess determinism** — reprocessing the same MP4 twice with the same seed produces byte-identical shards.
- **End-to-end dry run** — 1k real + 1k fake clips through the full pipeline on MPS, reach >80% val accuracy within 10 minutes. If not, stop and debug before spending GPU money.

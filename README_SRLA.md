# Hard-SRLA (frozen experiment implementation)

SRLA is a training-only target assignment strategy for YOLOv5 detection. It adds no inference module.
The final implementation is the Hard-SRLA code used for the three-seed experiments, tag
`srla-hard-3seed-final` (`a2f51f4b`). This branch only adds evaluation tools, focused regression tests,
documentation, and corrects a stale configuration comment. It does not retune the analyzer or loss.

## Training

Run from the repository root in the original experiment environment:

```bash
python train.py --weights yolov5s.pt --data /path/to/visdrone_aic/dataset.yaml --hyp data/hyps/hyp.srla.yaml --imgsz 640 --batch-size 16 --epochs 100 --workers 4 --device 0 --seed 0 --patience 0 --project runs/srla --name hard_seed0
```

For the paired baseline, use `data/hyps/hyp.scratch-low.yaml` with otherwise identical settings.
Use seeds 0, 1 and 2 in separate output directories. The experiment subset contains 1499 training and
300 validation images (20328 validation GT), not the complete official VisDrone benchmark.
Exact original options, split/label hashes and initial-weight hash are preserved in the experiment archive.
These commands do not download or recreate the archived split.

## Algorithm and scope

- Analyze the final augmented images under `no_grad`, using luminance and native-pixel context crops.
- Context expansion 1.5; object and up to four unmasked background strips; descriptors resized to 64.
- Mean-centered Hann window, log FFT magnitude, four radial bands, Jensen-Shannon distance.
- Simulate strides using each native crop's size, area downsampling and bilinear reconstruction.
  Padding buckets batch work without quantizing the original crop or sample sizes.
- Retention is clipped degraded/base spectral contrast. Weak base contrast (<1e-4), unsupported
  small crops and missing context use neutral retention (all ones).
- Keep original anchor-compatible candidates only where **R > 0.5**. If no compatible layer survives,
  restore the best-retention originally compatible layer. Originally unmatched GT remains unmatched.
- Box/class/objectness loss definitions and IoU objectness targets are unchanged.
  Validation and inference do not call the analyzer. There is no Soft or random gating in this branch.
- Epoch logs count GT-anchor candidates **before** neighbor expansion, not unique positive cells.

Batching is numerically close, not bitwise equivalent to the slow reference: archived augmented data
showed one threshold crossing among 5184 decisions (0.499952 to 0.500470). Do not claim exact equivalence.
Original reference, profiling and roundoff diagnostics remain in the experiment archive, outside formal code.

## Final results

Three seeds, mean +/- sample SD (ddof=1), best CSV fitness epoch:

| Metric | Baseline | Hard-SRLA | Paired improvement |
|---|---:|---:|---:|
| mAP50 (%) | 20.930 +/- 0.203 | 22.390 +/- 0.259 | +1.460 +/- 0.056 pp |
| mAP50:95 (%) | 10.742 +/- 0.101 | 11.462 +/- 0.150 | +0.720 +/- 0.104 pp |
| Last mAP50:95 (%) | 10.659 +/- 0.070 | 11.360 +/- 0.208 | +0.701 +/- 0.143 pp |

Seed0 Baseline / Matched Random / Hard / Soft mAP50:95: 10.859 / 11.036 / 11.591 / 10.390%.
The random control matches candidate sparsity, not target-size or expanded-cell counts.
The completed three-seed Control reaches 11.034 +/- 0.040% mAP50:95. Hard exceeds it in every seed
(+0.555 / +0.425 / +0.303 pp), with paired mean +0.428 +/- 0.126 pp; Control exceeds Baseline by
+0.292 +/- 0.108 pp. This supports spectral-selection value beyond random pruning, without a
statistical-significance claim. Both new seeds passed all 100 epoch candidate-pairing checks and
the fixed per-level removal-rate tolerance.
The tested Soft rule failed; this is not a claim about every possible soft rule.

## Evaluation

`tools/benchmark_srla_inference.py` compares two checkpoints on the same 32 validation images:
FP32, batch=1, 640, fused forward, 50 warmups and 200 timed forwards in each of five alternating rounds.
CUDA events are synchronized. It reports parameters, THOP MACs x2 GFLOPs, checkpoint bytes,
latency/FPS, PyTorch allocated/reserved memory and incremental forward memory.
Timing excludes disk loading, preprocessing, transfers and NMS; FPS is forward-only throughput.
Device-wide memory and end-to-end application FPS are different measurements.

```bash
python tools/benchmark_srla_inference.py --baseline /path/to/baseline/best.pt --hard /path/to/hard/best.pt --images /path/to/val.txt --out runs/inference_new
python tools/analyze_srla_final.py --experiment-root /path/to/AIC_Program --inference runs/inference_new/results.json --out runs/final_report_new
YOLO_AUTOINSTALL=false YOLOv5_AUTOINSTALL=false python -m pytest tests/test_srla.py -q
```

The report builder expects the preserved experiment directories `YOLOv5-srla-fast`,
`YOLOv5-srla-soft`, and `YOLOv5-matched-control` under the experiment root. It verifies 100-row runs,
uses fixed CSV fitness selection (earliest tie), computes paired statistics, and generates tables and
exactly three figure groups. It refuses to overwrite an output directory. CSV rounding may obscure
the precise checkpoint-selected epoch in close ties.

Final local artifacts: `runs/srla-final/report/README.md`; raw inference: `runs/srla-final/inference_threads1/`.
The initial uncontrolled-thread timing run is retained under `runs/srla-final/inference/`.
Latency varies on the laptop; no claim of exact identical speed or acceleration is made.
Training overhead must be reported alongside accuracy, using recorded end-to-end training wall times.

## Archives

- Hard: `YOLOv5-srla-fast/runs/srla-fast/freeze/srla-hard-3seed-final`.
- Matched Control: `YOLOv5-matched-control/runs/matched-gating/freeze/matched-control-seed0-final`.
  Commit b1f037e0, seed0, 100 epochs, best/last weights, logs, configuration, removal counts,
  environment and source archive; SHA256SUMS verified and files made read-only.
- Soft and size evaluation: `YOLOv5-srla-soft/runs/srla-soft/`.
- Final tables, plots, timing and source snapshot have their own SHA256 manifest.

Only Hard implementation, necessary training/loss integration, configuration, tests and evaluation tools
belong to the formal code diff. Experimental variants and raw debugging evidence remain archived.

## Three-seed Control charts

Pass --control-summary /path/to/matched-control-3seed-extension/seeds012/summary.json to tools/analyze_srla_final.py. This adds control_three_seeds and control_paired_gains in PNG/PDF, with paired sample SD and individual seed points. The earlier seed0 figures remain explicitly labeled as historical context.

## Final freeze v2

The authoritative consolidated snapshot is
runs/srla-final/freeze/srla-final-v2 in the YOLOv5-srla-final-v2 worktree.
It includes all six Baseline/Hard runs, all three Matched Control runs, the single Soft run,
assignment evidence, three-seed figures, inference measurements, environments, exact source
archives and SHA256 manifests. The older v1 and experiment snapshots remain unchanged.
See its README.md for the final result index. PR publication is a separate next step.

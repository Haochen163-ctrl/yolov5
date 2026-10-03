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

### Matched-Gating Control: three paired seeds

| Method | mAP50 (%) | mAP50:95 (%) | Last mAP50:95 (%) |
|---|---:|---:|---:|
| Baseline | 20.930 +/- 0.203 | 10.742 +/- 0.101 | 10.659 +/- 0.070 |
| Matched Random | 21.722 +/- 0.138 | 11.034 +/- 0.040 | 10.954 +/- 0.073 |
| Hard-SRLA | 22.390 +/- 0.259 | 11.462 +/- 0.150 | 11.360 +/- 0.208 |

**Matched-Baseline: +0.292 +/- 0.108 pp. Hard-Matched: +0.428 +/- 0.126 pp.**
Hard exceeds Matched in every seed: +0.555 / +0.425 / +0.303 pp.
The fixed random control matches candidate sparsity before neighbor expansion, with removal targets
P3/P4/P5 = 9.08/37.18/59.28%. It does not match target-size or expanded-cell counts.
All three seeds passed 100 epoch candidate-pairing checks and the fixed removal-rate tolerance.
These results support selection-specific value relative to this random control, without establishing
statistical significance or separating causal contributions from size-related effects.

The single pre-fixed Soft rule reached 10.390% at seed0, versus Hard's 11.591%, and was stopped.
It was not repeated at seeds 1/2; this is not a claim about every possible soft rule.

Training wall time: Baseline 40.838 +/- 0.399 min versus Hard 78.765 +/- 2.329 min (1.929x).
Both have 7,046,599 parameters and 15.831 GFLOPs. Forward-only mean latency was 13.957 versus
14.557 ms, with substantial round variation; median latency was 13.560 versus 13.497 ms.
The architecture is unchanged; the measurements do not establish exact equal speed or acceleration.

## Evaluation

`tools/benchmark_srla_inference.py` compares two checkpoints on the same 32 validation images:
FP32, batch=1, 640, fused forward, 50 warmups and 200 timed forwards in each of five alternating rounds.
CUDA events are synchronized. It reports parameters, THOP MACs x2 GFLOPs, checkpoint bytes,
latency/FPS, PyTorch allocated/reserved memory and incremental forward memory.
Timing excludes disk loading, preprocessing, transfers and NMS; FPS is forward-only throughput.
Device-wide memory and end-to-end application FPS are different measurements.

```bash
python tools/benchmark_srla_inference.py --baseline /path/to/baseline/best.pt --hard /path/to/hard/best.pt --images /path/to/val.txt --out runs/inference_new
python tools/analyze_srla_final.py --experiment-root /path/to/AIC_Program --inference runs/inference_new/results.json --control-summary /path/to/matched-control-3seed-extension/seeds012/summary.json --out runs/final_report_new
YOLO_AUTOINSTALL=false YOLOv5_AUTOINSTALL=false python -m pytest tests/test_srla.py -q
```

The report builder expects the preserved experiment directories `YOLOv5-srla-fast`,
`YOLOv5-srla-soft`, and `YOLOv5-matched-control` under the experiment root, plus the frozen
three-seed Control summary and its adjacent seed1/2 CSVs. It verifies all nine 100-row Baseline/Hard/Control CSVs against the summary,
uses fixed CSV fitness selection (earliest tie), computes paired statistics, and generates tables and
exactly three figure groups. It refuses to overwrite an output directory. CSV rounding may obscure
the precise checkpoint-selected epoch in close ties.

Final report: `runs/srla-final/freeze/srla-final-v3/report/README.md` in the report-v3 worktree.
Raw inference is preserved in the v2 data snapshot under `final_v1/inference_threads1/`.
The initial uncontrolled-thread timing run is retained under `runs/srla-final/inference/`.
Latency varies on the laptop; no claim of exact identical speed or acceleration is made.
Training overhead must be reported alongside accuracy, using recorded end-to-end training wall times.

## Archives

- Hard: `YOLOv5-srla-fast/runs/srla-fast/freeze/srla-hard-3seed-final`.
- Matched Control: `YOLOv5-matched-control/runs/matched-gating/freeze/matched-control-seed0-final`.
  Commit b1f037e0, seed0, 100 epochs, best/last weights, logs, configuration, removal counts,
  environment and source archive; SHA256SUMS verified and files made read-only.
  Seeds 1/2 and the combined three-seed summary are in
  `YOLOv5-matched-control-seeds12/runs/matched-gating/freeze/matched-control-3seed-extension`.
- Soft and size evaluation: `YOLOv5-srla-soft/runs/srla-soft/`.
- Final tables, plots, timing and source snapshot have their own SHA256 manifest.

Only Hard implementation, necessary training/loss integration, configuration, tests and evaluation tools
belong to the formal code diff. Experimental variants and raw debugging evidence remain archived.

## Final freeze v3

The report snapshot `srla-final-v3` in the `YOLOv5-srla-report-v3` worktree supersedes earlier
report text. It contains regenerated tables/figures, the final literature check, validation evidence,
source archive, PR description and SHA256 manifest. Its manifest references the unchanged
`srla-final-v2` data snapshot, which retains all six Baseline/Hard runs, three Matched runs and one
Soft run, including weights, environments and assignment evidence. Historical snapshots remain
unchanged; weights and complete experimental bundles are not part of this PR.

## Relationship to prior work (checked 2026-10-04)

The proposed combination is **instance-level image-domain spectral retention under level-specific
sampling simulations, used to gate cross-level anchor-compatible positives**. This is training-only,
with no inference module. Sampling is a proxy for spatial resolution loss, not a simulation of the
learned backbone's actual transfer function or a Shannon-information measurement.

[FSAF (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Feature_Selective_Anchor-Free_Module_for_Single-Shot_Object_Detection_CVPR_2019_paper.html)
already assigns instances across pyramid levels.
[SET (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.pdf)
already provides training-time spectral enhancement without extra inference burden.
[PLUSNet (2025 preprint)](https://arxiv.org/html/2504.20602v1) combines frequency processing with
geometric label assignment; [CDCEDet (2026)](https://jeit.ac.cn/en/article/doi/10.11999/JEIT260317)
also combines frequency enhancement and spatial label assignment.
[Spectra-Net (2026)](https://www.mdpi.com/1999-4893/19/8/628) addresses spectral preservation during
backbone downsampling; [DyFrDet (2026 preprint)](https://arxiv.org/html/2608.02495v1) combines
frequency-aware features and probabilistic label disambiguation.

Within the sources inspected, no direct match to the specific retention-to-assignment combination
was identified. This bounded search does not establish a universal first. Do not claim novelty for
frequency processing, cross-level assignment, or training-only operation individually.
The detailed source matrix, access limitations and recommended Chinese wording are archived in
`srla-final-v3/literature_check_2026-10-04.md`.

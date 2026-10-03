"""Compare two detection checkpoints on identical preloaded images (forward only)."""

import argparse
import copy
import gc
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.experimental import attempt_load
from utils.augmentations import letterbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Validation image list")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = args.images.read_text().splitlines()[:32]
    images = []
    for path in paths:
        im = cv2.imread(path)
        if im is None:
            raise ValueError(path)
        im = letterbox(im, (640, 640), auto=False)[0]
        images.append(torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255)
    inputs = torch.cat(images).cuda()
    result = {
        "protocol": "FP32, batch=1, 640x640, fused eval forward only; no loading/H2D/preprocess/NMS; CUDA events synchronized",
        "cpu_threads": torch.get_num_threads(),
        "warmup_per_round": args.warmup,
        "iterations_per_round": args.iterations,
        "rounds": args.rounds,
        "order": "alternating baseline-hard / hard-baseline",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        "input_tensor_sha256": hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
        "models": {},
    }
    checkpoints = {"baseline": args.baseline, "hard": args.hard}
    for name, path in checkpoints.items():
        raw = torch.load(path, map_location="cpu", weights_only=False)
        original = raw.get("ema") or raw["model"]
        params = sum(p.numel() for p in original.parameters())
        model = attempt_load(path, device=torch.device("cuda"), fuse=True).float().eval()
        probe = copy.deepcopy(model)
        with torch.inference_mode():
            macs, _ = profile(probe, inputs=(inputs[:1],), verbose=False)
        result["models"][name] = {
            "checkpoint": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameters_unfused": params,
            "parameters_fused": sum(p.numel() for p in model.parameters()),
            "gflops": macs * 2 / 1e9,
            "flops_convention": "THOP MACs x 2, actual 1x3x640x640 fused forward",
            "checkpoint_bytes": path.stat().st_size,
            "rounds": [],
        }
        del probe, model, raw, original
        gc.collect()
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for r in range(args.rounds):
            for name in ["baseline", "hard"] if r % 2 == 0 else ["hard", "baseline"]:
                model = attempt_load(checkpoints[name], device=torch.device("cuda"), fuse=True).float().eval()
                for i in range(args.warmup):
                    model(inputs[i % len(inputs) : i % len(inputs) + 1])
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
                events = [
                    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    for _ in range(args.iterations)
                ]
                for i, (start, end) in enumerate(events):
                    start.record()
                    output = model(inputs[i % len(inputs) : i % len(inputs) + 1])
                    end.record()
                    del output
                torch.cuda.synchronize()
                ms = [a.elapsed_time(b) for a, b in events]
                row = {
                    "round": r,
                    "latency_ms": ms,
                    "mean_ms": float(np.mean(ms)),
                    "fps": 1000 / float(np.mean(ms)),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "resident_mib": resident / 2**20,
                    "forward_increment_mib": (torch.cuda.max_memory_allocated() - resident) / 2**20,
                }
                result["models"][name]["rounds"].append(row)
                print(name, r, row["mean_ms"], flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    for model in result["models"].values():
        values = np.concatenate([r["latency_ms"] for r in model["rounds"]])
        means = [r["mean_ms"] for r in model["rounds"]]
        model["summary"] = {
            "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "round_mean_std_ms": float(np.std(means, ddof=1)),
            "fps": 1000 / float(values.mean()),
            "peak_allocated_mib": max(r["peak_allocated_mib"] for r in model["rounds"]),
            "peak_reserved_mib": max(r["peak_reserved_mib"] for r in model["rounds"]),
            "forward_increment_mib": max(r["forward_increment_mib"] for r in model["rounds"]),
        }
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    (args.out / "environment.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    print(json.dumps({k: v["summary"] for k, v in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()

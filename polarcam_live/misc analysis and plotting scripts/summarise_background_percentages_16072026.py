from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BACKGROUND_DIR = ROOT / "datasets" / "background characterisation" / "16072026 backgrounds"
OUT_DIR = BACKGROUND_DIR / "percentage distributions"
ROD_INTENSITY_CSV = ROOT / "summarise background statistics" / "rod_angle_window_intensity_values.csv"
ANGLE_WINDOW_RAW = 14
MAX_FRAMES_PER_STACK = 20
MAX_SAMPLES_PER_FILE = 350_000


def box_mean_14x14(image: np.ndarray) -> np.ndarray:
    return cv2.blur(
        np.asarray(image, dtype=np.float32),
        (ANGLE_WINDOW_RAW, ANGLE_WINDOW_RAW),
        borderType=cv2.BORDER_REFLECT,
    )


def sample_values(values: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(values).ravel()
    if flat.size <= max_count:
        return flat.astype(np.float32, copy=True)
    idx = rng.choice(flat.size, size=max_count, replace=False)
    return flat[idx].astype(np.float32, copy=False)


def describe(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(v))
    std = float(np.std(v))
    if std > 0:
        centered = v - mean
        skew = float(np.mean(centered**3) / std**3)
        excess_kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        skew = 0.0
        excess_kurtosis = 0.0
    return {
        "mean": mean,
        "std": std,
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "zero_fraction": float(np.mean(v <= 0)),
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
    }


def load_mean_rod_intensities() -> dict[str, float]:
    rows = list(csv.DictReader(ROD_INTENSITY_CSV.open("r", encoding="utf-8", newline="")))
    out: dict[str, float] = {}
    for size in ["40x65nm", "25x65nm"]:
        vals = np.asarray(
            [float(row["angle_window_intensity_mean"]) for row in rows if row["rod_size"] == size],
            dtype=np.float64,
        )
        if vals.size == 0:
            raise ValueError(f"No angle-window intensity values found for {size}")
        out[size] = float(np.mean(vals))
    return out


def selected_indices(n_frames: int) -> np.ndarray:
    n = min(MAX_FRAMES_PER_STACK, n_frames)
    return np.unique(np.linspace(0, n_frames - 1, n, dtype=int))


def analyse_stack(path: Path, rng: np.random.Generator) -> dict:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"{path} is not a frame stack: shape={arr.shape}")
    idx = selected_indices(int(arr.shape[0]))
    per_frame = max(1, MAX_SAMPLES_PER_FILE // len(idx))

    pixel_samples = []
    window_samples = []
    for i in idx:
        frame = np.asarray(arr[i], dtype=np.float32)
        pixel_samples.append(sample_values(frame, per_frame, rng))
        window_samples.append(sample_values(box_mean_14x14(frame), per_frame, rng))

    pixels = np.concatenate(pixel_samples)
    windows = np.concatenate(window_samples)

    avg_path = path.with_name(path.stem + " average frame.npy")
    if avg_path.exists():
        avg = np.asarray(np.load(avg_path, mmap_mode="r"), dtype=np.float32)
    else:
        acc = np.zeros(arr.shape[1:], dtype=np.float64)
        for i in range(int(arr.shape[0])):
            acc += np.asarray(arr[i], dtype=np.float64)
        avg = (acc / int(arr.shape[0])).astype(np.float32)

    avg_pixels = sample_values(avg, MAX_SAMPLES_PER_FILE, rng)
    avg_windows = sample_values(box_mean_14x14(avg), MAX_SAMPLES_PER_FILE, rng)
    classification = "post-subtraction/background-subtracted" if describe(pixels)["zero_fraction"] > 0.5 else "as-acquired/no subtraction"

    return {
        "name": path.name,
        "shape": tuple(int(x) for x in arr.shape),
        "dtype": str(arr.dtype),
        "classification_inferred": classification,
        "pixels": pixels,
        "windows": windows,
        "avg_pixels": avg_pixels,
        "avg_windows": avg_windows,
        "pixel_stats": describe(pixels),
        "window_stats": describe(windows),
        "avg_pixel_stats": describe(avg_pixels),
        "avg_window_stats": describe(avg_windows),
    }


def write_summary_csv(results: list[dict], rod_means: dict[str, float]) -> None:
    fields = [
        "recording",
        "classification_inferred",
        "shape",
        "dtype",
        "pixel_mean",
        "pixel_std",
        "pixel_p50",
        "pixel_p99",
        "pixel_zero_fraction",
        "window14_mean",
        "window14_std",
        "window14_p50",
        "window14_p99",
        "avg_window14_mean",
        "avg_window14_std",
        "mean_percent_40x65nm",
        "std_percent_40x65nm",
        "mean_percent_25x65nm",
        "std_percent_25x65nm",
    ]
    with (OUT_DIR / "background_percentage_summary_by_recording.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "recording": r["name"],
                "classification_inferred": r["classification_inferred"],
                "shape": r["shape"],
                "dtype": r["dtype"],
                "pixel_mean": r["pixel_stats"]["mean"],
                "pixel_std": r["pixel_stats"]["std"],
                "pixel_p50": r["pixel_stats"]["p50"],
                "pixel_p99": r["pixel_stats"]["p99"],
                "pixel_zero_fraction": r["pixel_stats"]["zero_fraction"],
                "window14_mean": r["window_stats"]["mean"],
                "window14_std": r["window_stats"]["std"],
                "window14_p50": r["window_stats"]["p50"],
                "window14_p99": r["window_stats"]["p99"],
                "avg_window14_mean": r["avg_window_stats"]["mean"],
                "avg_window14_std": r["avg_window_stats"]["std"],
                "mean_percent_40x65nm": 100.0 * r["window_stats"]["mean"] / rod_means["40x65nm"],
                "std_percent_40x65nm": 100.0 * r["window_stats"]["std"] / rod_means["40x65nm"],
                "mean_percent_25x65nm": 100.0 * r["window_stats"]["mean"] / rod_means["25x65nm"],
                "std_percent_25x65nm": 100.0 * r["window_stats"]["std"] / rod_means["25x65nm"],
            }
            writer.writerow(row)


def write_percentage_samples(results: list[dict], rod_means: dict[str, float]) -> None:
    for r in results:
        safe = r["name"].replace(".npy", "")
        data = {
            "window14_value": r["windows"],
            "percent_of_40x65nm_mean_signal": 100.0 * r["windows"] / rod_means["40x65nm"],
            "percent_of_25x65nm_mean_signal": 100.0 * r["windows"] / rod_means["25x65nm"],
        }
        n = min(len(data["window14_value"]), MAX_SAMPLES_PER_FILE)
        with (OUT_DIR / f"{safe}_window14_percentage_samples.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(data.keys()))
            for i in range(n):
                writer.writerow([data[k][i] for k in data])


def plot_distributions(results: list[dict], rod_means: dict[str, float]) -> None:
    plt.figure(figsize=(10, 6))
    for r in results:
        vals = r["windows"]
        upper = max(np.percentile(vals, 99.5), 1.0)
        bins = np.linspace(0, upper, 120)
        plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, label=r["name"].replace("frame_stack_", "").replace(".npy", ""))
    plt.title("14x14 background window distributions")
    plt.xlabel("14x14 window mean")
    plt.ylabel("Probability density")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "window14_background_distributions_each_recording.png", dpi=240)
    plt.close()

    for size, mean_signal in rod_means.items():
        plt.figure(figsize=(10, 6))
        all_upper = max(np.percentile(100.0 * r["windows"] / mean_signal, 99.5) for r in results)
        bins = np.linspace(0, max(all_upper, 1.0), 140)
        for r in results:
            vals = 100.0 * r["windows"] / mean_signal
            label = r["name"].replace("frame_stack_", "").replace(".npy", "")
            plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, label=label)
        plt.title(f"14x14 background as percentage of mean {size} rod signal")
        plt.xlabel("Background / mean rod signal (%)")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"percentage_distribution_each_recording_{size}.png", dpi=240)
        plt.close()

    names = [r["name"].replace("frame_stack_", "").replace(".npy", "") for r in results]
    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, (size, mean_signal) in zip([-width / 2, width / 2], rod_means.items()):
        means = [100.0 * r["window_stats"]["mean"] / mean_signal for r in results]
        stds = [100.0 * r["window_stats"]["std"] / mean_signal for r in results]
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=size)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Background / mean rod signal (%)")
    ax.set_title("Mean ± std of 14x14 background percentage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "percentage_mean_std_by_recording.png", dpi=240)
    plt.close(fig)


def write_readme(results: list[dict], rod_means: dict[str, float]) -> None:
    lines = [
        "16072026 background percentage distributions",
        "",
        f"Input folder: {BACKGROUND_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "Method:",
        "- Analysed the four full frame_stack_*.npy recordings, excluding the saved average-frame npy files as separate recordings.",
        f"- Used a {ANGLE_WINDOW_RAW}x{ANGLE_WINDOW_RAW} raw-pixel box mean to match the angle-calculation window.",
        "- Converted each 14x14 window value into percentage of rod signal using the corrected mean rod intensity measured in the same 14x14 angle window.",
        f"- Mean 40x65nm rod signal: {rod_means['40x65nm']:.6g}.",
        f"- Mean 25x65nm rod signal: {rod_means['25x65nm']:.6g}.",
        "- Inferred background-subtracted recordings from their high zero fraction; filenames are preserved so this can be checked against acquisition notes.",
        "",
        "Summary:",
    ]
    for r in results:
        lines.append(
            f"- {r['name']} ({r['classification_inferred']}): "
            f"14x14 mean {r['window_stats']['mean']:.4g}, std {r['window_stats']['std']:.4g}; "
            f"40x65nm {100.0 * r['window_stats']['mean'] / rod_means['40x65nm']:.3g} +/- {100.0 * r['window_stats']['std'] / rod_means['40x65nm']:.3g}%; "
            f"25x65nm {100.0 * r['window_stats']['mean'] / rod_means['25x65nm']:.3g} +/- {100.0 * r['window_stats']['std'] / rod_means['25x65nm']:.3g}%."
        )
    lines.extend(
        [
            "",
            "Distribution type:",
            "- As-acquired background windows are right-skewed rather than clean Gaussian because the full-frame background has spatial structure and clipping.",
            "- Background-subtracted windows are zero-inflated/right-skewed because subtraction is clipped at zero.",
        ]
    )
    (OUT_DIR / "README_16072026_background_percentages.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(16072026)
    rod_means = load_mean_rod_intensities()
    paths = sorted(p for p in BACKGROUND_DIR.glob("frame_stack_*.npy") if "average frame" not in p.name)
    results = []
    for path in paths:
        print(f"Analysing {path.name}")
        results.append(analyse_stack(path, rng))
    write_summary_csv(results, rod_means)
    write_percentage_samples(results, rod_means)
    plot_distributions(results, rod_means)
    write_readme(results, rod_means)
    with (OUT_DIR / "analysis_parameters.json").open("w") as f:
        json.dump(
            {
                "background_dir": str(BACKGROUND_DIR),
                "angle_window_raw": ANGLE_WINDOW_RAW,
                "rod_mean_intensities_14x14_angle_window": rod_means,
                "max_frames_per_stack": MAX_FRAMES_PER_STACK,
                "max_samples_per_file": MAX_SAMPLES_PER_FILE,
            },
            f,
            indent=2,
        )
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

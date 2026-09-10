from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
BACKGROUND_DIR = ROOT / "datasets" / "background characterisation" / "16072026 backgrounds"
PCT_DIR = BACKGROUND_DIR / "percentage distributions"
OUT_DIR = PCT_DIR / "excluding first and min subtraction test"
ROD_INTENSITY_CSV = ROOT / "summarise background statistics" / "rod_angle_window_intensity_values.csv"
ANGLE_WINDOW_RAW = 14
MAX_FRAMES = 20
MAX_SAMPLES = 350_000
RAW_TEST_RECORDING = BACKGROUND_DIR / "frame_stack_20260716-140000.npy"
UNRECOVERABLE_TARGET = BACKGROUND_DIR / "frame_stack_20260716-140047.npy"


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
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "zero_fraction": float(np.mean(v <= 0)),
    }


def rod_means() -> dict[str, float]:
    rows = list(csv.DictReader(ROD_INTENSITY_CSV.open("r", encoding="utf-8", newline="")))
    out = {}
    for size in ["40x65nm", "25x65nm"]:
        vals = np.asarray(
            [float(row["angle_window_intensity_mean"]) for row in rows if row["rod_size"] == size],
            dtype=np.float64,
        )
        out[size] = float(np.mean(vals))
    return out


def plot_existing_excluding_first(rod_mean: dict[str, float]) -> None:
    summary_rows = list(csv.DictReader((PCT_DIR / "background_percentage_summary_by_recording.csv").open()))
    keep = summary_rows[1:]
    for size in ["40x65nm", "25x65nm"]:
        plt.figure(figsize=(9, 5.5))
        all_vals = []
        vals_by_name = []
        for row in keep:
            sample_csv = PCT_DIR / row["recording"].replace(".npy", "_window14_percentage_samples.csv")
            data = np.genfromtxt(sample_csv, delimiter=",", names=True)
            vals = data[f"percent_of_{size}_mean_signal"]
            vals_by_name.append((row["recording"], vals))
            all_vals.append(vals)
        upper = max(float(np.percentile(v, 99.5)) for v in all_vals)
        bins = np.linspace(0.0, max(upper, 1.0), 140)
        for name, vals in vals_by_name:
            label = name.replace("frame_stack_", "").replace(".npy", "")
            plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.7, label=label)
        plt.title(f"Background percentage distributions excluding first recording ({size})")
        plt.xlabel("Background / mean rod signal (%)")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"percentage_distribution_excluding_first_{size}.png", dpi=240)
        plt.close()


def selected_indices(n_frames: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_frames - 1, min(MAX_FRAMES, n_frames), dtype=int))


def compute_profiles(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames = int(stack.shape[0])
    acc = np.zeros(stack.shape[1:], dtype=np.float64)
    min_profile = None
    for i in range(n_frames):
        frame = np.asarray(stack[i], dtype=np.float32)
        acc += frame
        if min_profile is None:
            min_profile = frame.copy()
        else:
            np.minimum(min_profile, frame, out=min_profile)
    mean_profile = (acc / n_frames).astype(np.float32)
    assert min_profile is not None
    return mean_profile, min_profile


def min_subtraction_test(rng: np.random.Generator) -> dict[str, np.ndarray | dict[str, float]]:
    stack = np.load(RAW_TEST_RECORDING, mmap_mode="r")
    mean_profile, min_profile = compute_profiles(stack)
    idx = selected_indices(int(stack.shape[0]))
    per_frame = max(1, MAX_SAMPLES // len(idx))
    original_windows = []
    mean_sub_windows = []
    min_sub_windows = []
    for i in idx:
        frame = np.asarray(stack[i], dtype=np.float32)
        original_windows.append(sample_values(box_mean_14x14(frame), per_frame, rng))
        mean_resid = np.clip(frame - mean_profile, 0.0, None)
        min_resid = frame - min_profile
        min_resid = np.clip(min_resid, 0.0, None)
        mean_sub_windows.append(sample_values(box_mean_14x14(mean_resid), per_frame, rng))
        min_sub_windows.append(sample_values(box_mean_14x14(min_resid), per_frame, rng))
    out = {
        "original": np.concatenate(original_windows),
        "mean_profile_subtracted_clipped": np.concatenate(mean_sub_windows),
        "minimum_profile_subtracted": np.concatenate(min_sub_windows),
    }
    out["stats"] = {key: describe(vals) for key, vals in out.items()}
    return out


def write_min_test_outputs(test: dict[str, np.ndarray | dict[str, float]], rod_mean: dict[str, float]) -> None:
    stats = test["stats"]
    with (OUT_DIR / "mean_vs_min_profile_subtraction_140000_summary.csv").open("w", newline="") as f:
        fields = [
            "method",
            "window14_mean",
            "window14_std",
            "window14_p50",
            "window14_p99",
            "zero_fraction",
            "mean_percent_40x65nm",
            "std_percent_40x65nm",
            "mean_percent_25x65nm",
            "std_percent_25x65nm",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, d in stats.items():
            writer.writerow(
                {
                    "method": method,
                    "window14_mean": d["mean"],
                    "window14_std": d["std"],
                    "window14_p50": d["p50"],
                    "window14_p99": d["p99"],
                    "zero_fraction": d["zero_fraction"],
                    "mean_percent_40x65nm": 100.0 * d["mean"] / rod_mean["40x65nm"],
                    "std_percent_40x65nm": 100.0 * d["std"] / rod_mean["40x65nm"],
                    "mean_percent_25x65nm": 100.0 * d["mean"] / rod_mean["25x65nm"],
                    "std_percent_25x65nm": 100.0 * d["std"] / rod_mean["25x65nm"],
                }
            )

    for size, denom in rod_mean.items():
        plt.figure(figsize=(9, 5.5))
        values = [
            100.0 * test["original"] / denom,
            100.0 * test["mean_profile_subtracted_clipped"] / denom,
            100.0 * test["minimum_profile_subtracted"] / denom,
        ]
        labels = ["original raw", "subtract temporal mean, clip >=0", "subtract temporal minimum"]
        upper = max(float(np.percentile(v, 99.5)) for v in values)
        bins = np.linspace(0.0, max(upper, 1.0), 160)
        for vals, label in zip(values, labels):
            plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.7, label=label)
        plt.title(f"Mean-profile vs minimum-profile subtraction on raw 140000 ({size})")
        plt.xlabel("Background / mean rod signal (%)")
        plt.ylabel("Probability density")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"mean_vs_min_profile_subtraction_140000_{size}.png", dpi=240)
        plt.close()


def write_readme(test: dict[str, np.ndarray | dict[str, float]], rod_mean: dict[str, float]) -> None:
    stats = test["stats"]
    lines = [
        "Exclude-first and minimum-profile subtraction test",
        "",
        f"Input folder: {BACKGROUND_DIR}",
        f"Output folder: {OUT_DIR}",
        "",
        "What was changed:",
        "- The first recording, frame_stack_20260716-135510.npy, is excluded from the new distribution plots.",
        "- The lowest-mean existing distribution is frame_stack_20260716-140047.npy, but it is already background-subtracted and clipped. The original negative residuals are not recoverable, so the true minimum-profile alternative cannot be reconstructed from that file.",
        f"- To test the idea fairly, the raw non-subtracted recording {RAW_TEST_RECORDING.name} was processed three ways: original, subtract temporal mean profile with clipping at zero, subtract temporal minimum profile.",
        "",
        f"Mean rod signals used: 40x65nm={rod_mean['40x65nm']:.6g}, 25x65nm={rod_mean['25x65nm']:.6g}.",
        "",
        "140000 raw-stack test:",
    ]
    for method, d in stats.items():
        lines.append(
            f"- {method}: window mean {d['mean']:.4g}, std {d['std']:.4g}; "
            f"40x65nm {100.0 * d['mean'] / rod_mean['40x65nm']:.3g} +/- {100.0 * d['std'] / rod_mean['40x65nm']:.3g}%; "
            f"25x65nm {100.0 * d['mean'] / rod_mean['25x65nm']:.3g} +/- {100.0 * d['std'] / rod_mean['25x65nm']:.3g}%."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Subtracting the temporal minimum profile removes less background than subtracting the temporal mean profile, because the minimum is below the mean at most pixels.",
            "- It guarantees non-negative residuals without needing clipping, but it leaves a positive offset/noise floor rather than centering residuals around zero.",
        ]
    )
    (OUT_DIR / "README_excluding_first_and_min_subtraction.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(160720261)
    means = rod_means()
    plot_existing_excluding_first(means)
    test = min_subtraction_test(rng)
    write_min_test_outputs(test, means)
    write_readme(test, means)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

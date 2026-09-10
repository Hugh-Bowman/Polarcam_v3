from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


SOURCE_DIR = Path(r"E:\40nm apd precision")
SOURCE_ANALYSIS_DIR = SOURCE_DIR / "analysis"
TARGET_DIR = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm APD stuck rod data")
RAW_TDMS_DIR = TARGET_DIR / "raw tdms"
ANALYSIS_DIR = TARGET_DIR / "analysis"


def reset_dir(path: Path) -> None:
    target = path.resolve()
    root = TARGET_DIR.resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Refusing to clear path outside target root: {target}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        source = Path(row["file"])
        local_tdms = RAW_TDMS_DIR / source.name
        local_index = RAW_TDMS_DIR / f"{source.name}_index"
        out.append(
            {
                "rod_file": source.name,
                "source_tdms_path": str(source),
                "local_tdms_path": str(local_tdms),
                "local_tdms_index_path": str(local_index) if local_index.exists() else "",
                "theta_deg_current_curve": row["theta_deg"],
                "r_mean": row["r_mean"],
                "r_shift_to_rmax": row["r_shift_to_rmax"],
                "r_p16": row["r_p16"],
                "r_p84": row["r_p84"],
                "r_p16_raw": row["r_p16_raw"],
                "r_p84_raw": row["r_p84_raw"],
                "sigma_theta_deg_bandwidth_1Hz_to_nyquist": row["sigma_theta_deg"],
                "sigma_phi_deg_bandwidth_1Hz_to_nyquist": row["sigma_phi_deg"],
                "sigma_theta_deg_unfiltered_percentile": row["sigma_theta_deg_unfiltered_percentile"],
                "sigma_phi_deg_unfiltered_circular": row["sigma_phi_deg_unfiltered_circular"],
                "phi_mean_deg": row["phi_mean_deg"],
                "x_mean": row["x_mean"],
                "y_mean": row["y_mean"],
                "n_samples": row["n_samples"],
                "sample_rate_hz": row["sample_rate_hz"],
                "duration_s": row["duration_s"],
                "mean_I0": row["mean_I0"],
                "mean_I45": row["mean_I45"],
                "mean_I90": row["mean_I90"],
                "mean_I135": row["mean_I135"],
            }
        )
    return out


def main() -> None:
    if not SOURCE_ANALYSIS_DIR.exists():
        raise SystemExit(f"Missing APD analysis folder: {SOURCE_ANALYSIS_DIR}")
    reset_dir(TARGET_DIR)
    RAW_TDMS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    tdms_files = sorted(SOURCE_DIR.glob("*.tdms"), key=lambda p: p.name.lower())
    copied_tdms = []
    for tdms in tdms_files:
        copy_file(tdms, RAW_TDMS_DIR / tdms.name)
        copied_tdms.append(str(RAW_TDMS_DIR / tdms.name))
        index = tdms.with_name(f"{tdms.name}_index")
        if index.exists():
            copy_file(index, RAW_TDMS_DIR / index.name)

    for item in SOURCE_ANALYSIS_DIR.iterdir():
        if item.is_file():
            copy_file(item, ANALYSIS_DIR / item.name)

    analysis_rows = read_csv(SOURCE_ANALYSIS_DIR / "apd_precision_points.csv")
    plot_rows = make_plot_rows(analysis_rows)
    write_csv(TARGET_DIR / "apd_plot_points.csv", plot_rows)

    summary = {
        "source_dir": str(SOURCE_DIR),
        "source_analysis_dir": str(SOURCE_ANALYSIS_DIR),
        "target_dir": str(TARGET_DIR),
        "raw_tdms_folder": str(RAW_TDMS_DIR),
        "analysis_folder": str(ANALYSIS_DIR),
        "n_tdms_copied": len(copied_tdms),
        "n_analysed_points": len(plot_rows),
        "plot_points_csv": str(TARGET_DIR / "apd_plot_points.csv"),
        "theta_r_curve_used_for_current_theta_columns": {
            "label": "Fourkas finite-NA water/buffer",
            "J1": 0.65235,
            "J2": 0.03744,
            "J3": 0.10765,
            "r_max": 0.8914452224590091,
        },
        "channel_scalings": {
            "ai0_90": 1.0000,
            "ai1_45": 2.8859,
            "ai2_135": 2.3267,
            "ai3_0": 1.0732,
        },
        "note": "apd_plot_points.csv includes r_mean, shifted and raw r percentiles, and current theta/sigma columns so a new theta(r) curve can be applied later.",
    }
    (TARGET_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (TARGET_DIR / "README.txt").write_text(
        "\n".join(
            [
                "40 nm APD stuck rod data package",
                "",
                "raw tdms/ contains the TDMS files and TDMS index files copied from E:\\40nm apd precision.",
                "analysis/ contains the current APD plots and the original apd_precision_points.csv analysis table.",
                "apd_plot_points.csv is the simplified plot-ready table.",
                "To remap theta with a new theta(r) curve, use r_mean for point location and r_p16/r_p84 for sigma_theta.",
                "For clipped high-r rows, r_shift_to_rmax records the shift applied before calculating r_p16/r_p84.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Output: {TARGET_DIR}")
    print(f"Copied TDMS files: {len(copied_tdms)}")
    print(f"Analysed points: {len(plot_rows)}")


if __name__ == "__main__":
    main()

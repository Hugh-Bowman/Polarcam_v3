from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from nptdms import TdmsFile


NPY_CSV = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm stuck rod data\all_plot_points.csv")
APD_CSV = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm APD stuck rod data\apd_plot_points.csv")
APD_RAW = Path(r"E:\40nm apd precision")
OUT = Path(r"C:\Polarcam Software\Polarcam_v3\polarcam_live\datasets\40nm APD stuck rod data\r_value_comparison.json")

SCALES = {"ai0": 1.0, "ai1": 2.8859, "ai2": 2.3267, "ai3": 1.0732}
CHANNELS = {"I90": "ai0", "I45": "ai1", "I135": "ai2", "I0": "ai3"}


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def short_channel_name(name: str) -> str:
    return name.rsplit("/", 1)[-1].lower()


def read_tdms_r(path: Path, scaled: bool) -> float:
    tdms = TdmsFile.read(path)
    group = tdms.groups()[0]
    raw = {short_channel_name(ch.name): np.asarray(ch[:], dtype=np.float64) for ch in group.channels()}
    vals = {}
    for logical, ai in CHANNELS.items():
        vals[logical] = raw[ai] * (SCALES[ai] if scaled else 1.0)
    eps = 1e-15
    x = (vals["I0"] - vals["I90"]) / np.maximum(vals["I0"] + vals["I90"], eps)
    y = (vals["I45"] - vals["I135"]) / np.maximum(vals["I45"] + vals["I135"], eps)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.hypot(x[ok], y[ok])))


def main() -> None:
    npy_by_path: dict[str, dict] = {}
    with NPY_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            source = row["source_path"]
            if source not in npy_by_path:
                npy_by_path[source] = {
                    "recording": Path(source).name,
                    "source_path": source,
                    "r_mean": float(row["r_mean"]),
                    "exposures": set(),
                    "kinds": set(),
                }
            npy_by_path[source]["exposures"].add(row["exposure_label"])
            npy_by_path[source]["kinds"].add(row["plot_kind"])
    npy_rows = sorted(npy_by_path.values(), key=lambda row: row["r_mean"])

    tdms_post = {}
    with APD_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tdms_post[row["rod_file"]] = float(row["r_mean"])

    tdms_rows = []
    for name in sorted(tdms_post, key=natural_key):
        tdms_rows.append(
            {
                "file": name,
                "r_mean_pre_scaling": read_tdms_r(APD_RAW / name, scaled=False),
                "r_mean_post_scaling": tdms_post[name],
            }
        )

    summary = {
        "npy_count_unique_recordings": len(npy_rows),
        "npy_over_0p94_count": sum(row["r_mean"] > 0.94 for row in npy_rows),
        "npy_over_0p94_percent": 100.0 * sum(row["r_mean"] > 0.94 for row in npy_rows) / len(npy_rows),
        "tdms_count": len(tdms_rows),
        "tdms_pre_scaling_over_0p94_count": sum(row["r_mean_pre_scaling"] > 0.94 for row in tdms_rows),
        "tdms_pre_scaling_over_0p94_percent": 100.0
        * sum(row["r_mean_pre_scaling"] > 0.94 for row in tdms_rows)
        / len(tdms_rows),
        "tdms_post_scaling_over_0p94_count": sum(row["r_mean_post_scaling"] > 0.94 for row in tdms_rows),
        "tdms_post_scaling_over_0p94_percent": 100.0
        * sum(row["r_mean_post_scaling"] > 0.94 for row in tdms_rows)
        / len(tdms_rows),
    }

    serialisable_npy = []
    for row in npy_rows:
        serialisable_npy.append(
            {
                "recording": row["recording"],
                "source_path": row["source_path"],
                "r_mean": row["r_mean"],
                "exposures": sorted(row["exposures"]),
                "kinds": sorted(row["kinds"]),
            }
        )

    OUT.write_text(
        json.dumps({"summary": summary, "npy": serialisable_npy, "tdms": tdms_rows}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Output: {OUT}")
    print("NPY_R_VALUES")
    print(", ".join(f"{row['r_mean']:.6f}" for row in npy_rows))
    print("TDMS_R_VALUES file pre_scaling post_scaling")
    for row in tdms_rows:
        print(f"{row['file']}: {row['r_mean_pre_scaling']:.6f}, {row['r_mean_post_scaling']:.6f}")


if __name__ == "__main__":
    main()

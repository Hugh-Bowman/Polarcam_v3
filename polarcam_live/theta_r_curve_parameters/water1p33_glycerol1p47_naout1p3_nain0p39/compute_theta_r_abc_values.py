from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ABC_CSV = OUT_DIR / "theta_r_abc_values.csv"
CURVE_CSV = OUT_DIR / "theta_r_curve_values.csv"

NA_IN = 0.39
NA_OUT = 1.3
R_SAMPLES = 1001

MEDIA = [
    {"model": "water", "n_medium": 1.33},
    {"model": "glycerol", "n_medium": 1.47},
]


def q_from_na(na: float, n_medium: float, label: str) -> float:
    ratio = na / n_medium
    if ratio < 0.0:
        raise ValueError(f"{label} / n must be non-negative.")
    if ratio > 1.0:
        raise ValueError(f"{label} / n = {ratio:.6f} exceeds 1, so q would be non-real.")
    return math.sqrt(1.0 - ratio * ratio)


def compute_coefficients(na_in: float, na_out: float, n_medium: float) -> dict[str, float]:
    q_in = q_from_na(na_in, n_medium, "NA_in")
    q_out = q_from_na(na_out, n_medium, "NA_out")

    j1 = 0.25 * (
        (q_in**3 + q_in**2 + 3.0 * q_in)
        - (q_out**3 + q_out**2 + 3.0 * q_out)
    )
    j2 = (1.0 / 12.0) * (((1.0 - q_out) ** 3) - ((1.0 - q_in) ** 3))
    j3 = (q_in - (q_in**3) / 3.0) - (q_out - (q_out**3) / 3.0)

    a = 2.0 * j3
    b = j1 - j2
    c = j1 + j2 - 2.0 * j3
    r_max = b / (a + c)

    alpha_in_deg = math.degrees(math.asin(na_in / n_medium)) if na_in > 0 else 0.0
    alpha_out_deg = math.degrees(math.asin(na_out / n_medium))

    return {
        "alpha_in_deg": alpha_in_deg,
        "alpha_out_deg": alpha_out_deg,
        "q_in": q_in,
        "q_out": q_out,
        "J1": j1,
        "J2": j2,
        "J3": j3,
        "A": a,
        "B": b,
        "C": c,
        "r_max": r_max,
    }


def theta_from_r(r: np.ndarray, a: float, b: float, c: float, r_max: float) -> np.ndarray:
    rr = np.asarray(r, dtype=np.float64)
    theta = np.full(rr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    denom = b - c * rr[valid]
    val = (a * rr[valid]) / np.maximum(1e-15, denom)
    val = np.clip(val, 0.0, 1.0)
    theta[valid] = np.arcsin(np.sqrt(val))
    theta[np.isfinite(rr) & np.isclose(rr, r_max)] = np.pi / 2.0
    return theta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    abc_rows: list[dict[str, float | str]] = []
    curve_rows: list[list[float | str]] = []

    for medium in MEDIA:
        model = str(medium["model"])
        n_medium = float(medium["n_medium"])
        coeffs = compute_coefficients(NA_IN, NA_OUT, n_medium)
        abc_rows.append(
            {
                "model": model,
                "n_medium": n_medium,
                "na_in": NA_IN,
                "na_out": NA_OUT,
                **coeffs,
            }
        )

        r_grid = np.linspace(0.0, coeffs["r_max"], R_SAMPLES, dtype=np.float64)
        theta_rad = theta_from_r(r_grid, coeffs["A"], coeffs["B"], coeffs["C"], coeffs["r_max"])
        theta_deg = np.degrees(theta_rad)
        for r_value, th_rad, th_deg in zip(r_grid, theta_rad, theta_deg):
            curve_rows.append(
                [
                    model,
                    n_medium,
                    float(r_value),
                    float(th_rad),
                    float(th_deg),
                    coeffs["A"],
                    coeffs["B"],
                    coeffs["C"],
                    coeffs["r_max"],
                ]
            )

    with ABC_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "n_medium",
                "na_in",
                "na_out",
                "alpha_in_deg",
                "alpha_out_deg",
                "q_in",
                "q_out",
                "J1",
                "J2",
                "J3",
                "A",
                "B",
                "C",
                "r_max",
            ],
        )
        writer.writeheader()
        writer.writerows(abc_rows)

    with CURVE_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "n_medium", "r", "theta_rad", "theta_deg", "A", "B", "C", "r_max"])
        writer.writerows(curve_rows)

    print(f"ABC CSV: {ABC_CSV}")
    print(f"Curve CSV: {CURVE_CSV}")
    for row in abc_rows:
        print(
            f"{row['model']}: "
            f"J1={float(row['J1']):.9f} "
            f"J2={float(row['J2']):.9f} "
            f"J3={float(row['J3']):.9f} "
            f"A={float(row['A']):.9f} "
            f"B={float(row['B']):.9f} "
            f"C={float(row['C']):.9f} "
            f"r_max={float(row['r_max']):.9f}"
        )


if __name__ == "__main__":
    main()

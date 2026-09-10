from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ABC_CSV = OUT_DIR / "theta_r_abc_values.csv"
CURVE_CSV = OUT_DIR / "theta_r_curve_values.csv"

NA_OUT = 1.3
NA_IN = 0.39
R_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float64)

MEDIA = [
    {"label": "water", "n_medium": 1.33},
    {"label": "glycerol", "n_medium": 1.48},
]


def clip_unit_interval(x: float, label: str) -> float:
    if x < 0.0:
        raise ValueError(f"{label} must be non-negative; got {x}.")
    if x > 1.0:
        return 1.0
    return x


def angular_coefficients(na_out: float, na_in: float, n_medium: float) -> dict[str, float]:
    sin_theta_out = clip_unit_interval(na_out / n_medium, "NA_out / n_medium")
    sin_theta_in = clip_unit_interval(na_in / n_medium, "NA_in / n_medium")
    if sin_theta_in > sin_theta_out + 1e-12:
        raise ValueError("Inner NA exceeds outer NA in the chosen medium.")

    theta_out = math.asin(sin_theta_out)
    theta_in = math.asin(sin_theta_in)
    q_out = math.cos(theta_out)
    q_in = math.cos(theta_in)

    def F1(u: float) -> float:
        return (3.0 / 4.0) * u + (1.0 / 6.0) * u**3 + (3.0 / 20.0) * u**5

    def F2(u: float) -> float:
        return (1.0 / 4.0) * u - (1.0 / 6.0) * u**3 + (1.0 / 20.0) * u**5

    def F3(u: float) -> float:
        return (1.0 / 3.0) * u**3 - (1.0 / 5.0) * u**5

    J1 = F1(q_in) - F1(q_out)
    J2 = F2(q_in) - F2(q_out)
    J3 = F3(q_in) - F3(q_out)

    # Asin-form coefficients:
    # theta(r) = asin(sqrt((A*r) / (B - C*r)))
    A = 2.0 * J3
    B = J1 - J2
    C = J1 + J2 - 2.0 * J3
    r_max = B / (A + C)

    return {
        "theta_in_deg": math.degrees(theta_in),
        "theta_out_deg": math.degrees(theta_out),
        "q_in": q_in,
        "q_out": q_out,
        "J1": J1,
        "J2": J2,
        "J3": J3,
        "A": A,
        "B": B,
        "C": C,
        "r_max": r_max,
    }


def theta_from_r(r: np.ndarray, A: float, B: float, C: float, r_max: float) -> np.ndarray:
    rr = np.asarray(r, dtype=np.float64)
    theta = np.full(rr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= 0.0) & (rr < r_max)
    denom = B - (C * rr[valid])
    val = (A * rr[valid]) / np.maximum(1e-15, denom)
    val = np.clip(val, 0.0, 1.0)
    theta[valid] = np.arcsin(np.sqrt(val))
    theta[np.isfinite(rr) & np.isclose(rr, r_max)] = np.pi / 2.0
    return theta


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    abc_rows: list[dict[str, float | str]] = []
    curve_rows: list[list[float | str]] = []

    for medium in MEDIA:
        label = str(medium["label"])
        n_medium = float(medium["n_medium"])
        coeffs = angular_coefficients(NA_OUT, NA_IN, n_medium)
        abc_rows.append(
            {
                "model": label,
                "n_medium": n_medium,
                "na_out": NA_OUT,
                "na_in": NA_IN,
                "theta_in_deg": coeffs["theta_in_deg"],
                "theta_out_deg": coeffs["theta_out_deg"],
                "q_in": coeffs["q_in"],
                "q_out": coeffs["q_out"],
                "J1": coeffs["J1"],
                "J2": coeffs["J2"],
                "J3": coeffs["J3"],
                "A": coeffs["A"],
                "B": coeffs["B"],
                "C": coeffs["C"],
                "r_max": coeffs["r_max"],
            }
        )

        theta_rad = theta_from_r(R_GRID, coeffs["A"], coeffs["B"], coeffs["C"], coeffs["r_max"])
        theta_deg = np.degrees(theta_rad)
        for r_value, th_rad, th_deg in zip(R_GRID, theta_rad, theta_deg):
            curve_rows.append(
                [
                    label,
                    n_medium,
                    float(r_value),
                    float(th_rad) if np.isfinite(th_rad) else "",
                    float(th_deg) if np.isfinite(th_deg) else "",
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
                "na_out",
                "na_in",
                "theta_in_deg",
                "theta_out_deg",
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
    write_outputs()

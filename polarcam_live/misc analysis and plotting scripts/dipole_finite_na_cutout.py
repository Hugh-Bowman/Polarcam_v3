#!/usr/bin/env python3
"""
Dipole emission model with finite NA and a circular central BFP cutout.

Implements the model:

    theta_max = asin(NA_out / n_medium)
    sin(theta_in) = rho * sin(theta_max)

or, equivalently, if the central hole is specified as a physical NA,

    sin(theta_in) = NA_hole / n_medium

The angular integrals are

    J1 = ∫[theta_in, theta_max] (2 cos^2(theta) + 3/4 sin^4(theta)) sin(theta) dtheta
    J2 = ∫[theta_in, theta_max] (1/4 sin^4(theta)) sin(theta) dtheta
    J3 = ∫[theta_in, theta_max] (sin^2(theta) cos^2(theta)) sin(theta) dtheta

with closed-form expressions using u = cos(theta).

The anisotropy coordinates are

    X = (J1 - J2)(dx^2 - dy^2) / D
    Y = 2(J1 - J2)dxdy / D

where

    D = (J1 + J2)(dx^2 + dy^2) + 2 J3 dz^2.

Therefore

    phi_d = 0.5 * atan2(Y, X)

is unaffected by the cutout, while

    theta_d = atan(sqrt(2 J3 r / ((J1 - J2) - r (J1 + J2))))

with r = sqrt(X^2 + Y^2), must be corrected for the finite central cutout.

The square root follows directly from solving Eq. (43), because r is proportional
to sin^2(theta_d), not sin(theta_d). If a PDF/OCR version appears to omit the
sqrt, that expression will not invert the forward model correctly.

Examples
--------
Default plot for water and 50% glycerol, including a BFP hole rho=0.39:

    python dipole_finite_na_cutout.py

Use physical hole NA values instead of fractional BFP radii:

    python dipole_finite_na_cutout.py --hole-mode na --holes 0 0.25 0.39

Use only water and save outputs to a custom directory:

    python dipole_finite_na_cutout.py --media water:1.333 --outdir results_water

Notes
-----
- If NA_out / n_medium > 1, the script clips sin(theta_max) to 1 and warns.
  This means collection is treated as reaching theta_max = 90 degrees in that
  medium. For water with NA_out = 1.4, NA_out / n_water > 1, so clipping occurs.
- The refractive indices are configurable. Defaults are approximate:
      water:        n = 1.333
      50% glycerol: n = 1.398
  Exact glycerol-water refractive index depends on concentration definition,
  wavelength, and temperature.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Medium:
    """Optical medium with a display name and refractive index."""
    name: str
    n: float


@dataclass(frozen=True)
class AngularIntegrals:
    """Angular limits and J coefficients for one NA / medium / hole configuration."""
    medium: Medium
    na_out: float
    hole_value: float
    hole_mode: str
    theta_in: float
    theta_max: float
    rho_effective: float
    na_hole_effective: float
    J1: float
    J2: float
    J3: float

    @property
    def A(self) -> float:
        """J1 - J2."""
        return self.J1 - self.J2

    @property
    def B(self) -> float:
        """J1 + J2."""
        return self.J1 + self.J2

    @property
    def r_max(self) -> float:
        """Maximum anisotropy radius, reached for theta_d = pi/2."""
        return self.A / self.B


def _clip_sin_value(x: float, label: str) -> float:
    """Clip a sine argument into [0, 1] with a warning if needed."""
    if x < 0:
        raise ValueError(f"{label} must be non-negative; got {x}.")
    if x > 1:
        warnings.warn(
            f"{label} = {x:.6g} is greater than 1. "
            "Clipping to 1, so the corresponding angle is 90 degrees.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0
    return x


def angular_integrals(
    na_out: float = 1.4,
    n_medium: float = 1.333,
    hole: float = 0.39,
    hole_mode: str = "rho",
    medium_name: str = "medium",
) -> AngularIntegrals:
    """
    Compute J1, J2, J3 for a finite-NA dipole emission model.

    Parameters
    ----------
    na_out:
        Outer numerical aperture.
    n_medium:
        Refractive index of the sample medium.
    hole:
        Central hole size. Interpretation depends on `hole_mode`.
    hole_mode:
        "rho" means hole is a fractional BFP radius, so
            sin(theta_in) = rho sin(theta_max).
        "na" means hole is a physical central-hole NA, so
            sin(theta_in) = NA_hole / n_medium.
    medium_name:
        Display name used in output tables and plot labels.

    Returns
    -------
    AngularIntegrals
        Dataclass containing angular limits and J coefficients.
    """
    if na_out <= 0:
        raise ValueError("na_out must be positive.")
    if n_medium <= 0:
        raise ValueError("n_medium must be positive.")
    if hole < 0:
        raise ValueError("hole must be non-negative.")

    sin_theta_max = _clip_sin_value(na_out / n_medium, "NA_out / n_medium")
    theta_max = math.asin(sin_theta_max)

    if hole_mode == "rho":
        if hole > 1:
            raise ValueError("For hole_mode='rho', hole must be in [0, 1].")
        sin_theta_in = hole * sin_theta_max
        rho_effective = hole
        na_hole_effective = hole * na_out
    elif hole_mode == "na":
        sin_theta_in = _clip_sin_value(hole / n_medium, "NA_hole / n_medium")
        if sin_theta_in > sin_theta_max + 1e-12:
            raise ValueError(
                f"Central hole NA {hole:g} exceeds the effective outer collection NA "
                f"{na_out:g} in medium n={n_medium:g}."
            )
        rho_effective = sin_theta_in / sin_theta_max if sin_theta_max > 0 else 0.0
        na_hole_effective = hole
    else:
        raise ValueError("hole_mode must be either 'rho' or 'na'.")

    theta_in = math.asin(sin_theta_in)

    # u = cos(theta). Since theta_in <= theta_max, u_in >= u_max.
    u_in = math.cos(theta_in)
    u_max = math.cos(theta_max)

    def F1(u: float) -> float:
        return (3.0 / 4.0) * u + (1.0 / 6.0) * u**3 + (3.0 / 20.0) * u**5

    def F2(u: float) -> float:
        return (1.0 / 4.0) * u - (1.0 / 6.0) * u**3 + (1.0 / 20.0) * u**5

    def F3(u: float) -> float:
        return (1.0 / 3.0) * u**3 - (1.0 / 5.0) * u**5

    # Integral theta_in -> theta_max = F(u_in) - F(u_max).
    J1 = F1(u_in) - F1(u_max)
    J2 = F2(u_in) - F2(u_max)
    J3 = F3(u_in) - F3(u_max)

    return AngularIntegrals(
        medium=Medium(medium_name, n_medium),
        na_out=na_out,
        hole_value=hole,
        hole_mode=hole_mode,
        theta_in=theta_in,
        theta_max=theta_max,
        rho_effective=rho_effective,
        na_hole_effective=na_hole_effective,
        J1=J1,
        J2=J2,
        J3=J3,
    )


def dipole_vector(theta_d: np.ndarray | float, phi_d: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert rod polar/azimuthal angles to dipole direction components.

    theta_d is measured from the optical axis z.
    phi_d is the azimuth in the x-y image plane.
    """
    theta_d = np.asarray(theta_d)
    phi_d = np.asarray(phi_d)
    dx = np.sin(theta_d) * np.cos(phi_d)
    dy = np.sin(theta_d) * np.sin(phi_d)
    dz = np.cos(theta_d)
    return dx, dy, dz


def intensities_from_orientation(
    theta_d: np.ndarray | float,
    phi_d: np.ndarray | float,
    coeffs: AngularIntegrals,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute I0, I90, I45, I135 for a rod orientation.
    """
    dx, dy, dz = dipole_vector(theta_d, phi_d)

    J1, J2, J3 = coeffs.J1, coeffs.J2, coeffs.J3

    I0 = math.pi * (dx**2 * J1 + dy**2 * J2 + dz**2 * J3)
    I90 = math.pi * (dy**2 * J1 + dx**2 * J2 + dz**2 * J3)

    I45 = math.pi * (
        0.5 * (J1 + J2) * (dx**2 + dy**2)
        + (J1 - J2) * dx * dy
        + J3 * dz**2
    )
    I135 = math.pi * (
        0.5 * (J1 + J2) * (dx**2 + dy**2)
        - (J1 - J2) * dx * dy
        + J3 * dz**2
    )

    return I0, I90, I45, I135


def anisotropy_from_intensities(
    I0: np.ndarray | float,
    I90: np.ndarray | float,
    I45: np.ndarray | float,
    I135: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute anisotropy coordinates X, Y and radius r from four polarization channels.
    """
    I0 = np.asarray(I0)
    I90 = np.asarray(I90)
    I45 = np.asarray(I45)
    I135 = np.asarray(I135)

    X = (I0 - I90) / (I0 + I90)
    Y = (I45 - I135) / (I45 + I135)
    r = np.sqrt(X**2 + Y**2)
    return X, Y, r


def anisotropy_from_orientation(
    theta_d: np.ndarray | float,
    phi_d: np.ndarray | float,
    coeffs: AngularIntegrals,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute X, Y and r directly from rod orientation and angular coefficients.
    """
    dx, dy, dz = dipole_vector(theta_d, phi_d)

    denominator = coeffs.B * (dx**2 + dy**2) + 2.0 * coeffs.J3 * dz**2
    X = coeffs.A * (dx**2 - dy**2) / denominator
    Y = 2.0 * coeffs.A * dx * dy / denominator
    r = np.sqrt(X**2 + Y**2)
    return X, Y, r


def phi_from_xy(X: np.ndarray | float, Y: np.ndarray | float) -> np.ndarray:
    """
    Reconstruct image-plane azimuth phi_d from anisotropy X, Y.

    The result is modulo pi, as expected for a rod/dipole axis.
    """
    return 0.5 * np.arctan2(Y, X)


def theta_from_r(
    r: np.ndarray | float,
    coeffs: AngularIntegrals,
    invalid: str = "nan",
) -> np.ndarray:
    """
    Reconstruct polar angle theta_d from anisotropy radius r.

    Formula:
        theta_d = atan(sqrt(2 J3 r / ((J1 - J2) - r (J1 + J2))))

    This is obtained by writing t = tan^2(theta_d), giving

        t = 2 J3 r / ((J1 - J2) - r (J1 + J2)).

    Parameters
    ----------
    r:
        Anisotropy radius.
    coeffs:
        Angular coefficients.
    invalid:
        "nan" returns NaN for r outside [0, r_max].
        "clip" clips r into [0, r_max].
        "raise" raises a ValueError.
    """
    r = np.asarray(r, dtype=float)
    r_max = coeffs.r_max
    bad = (r < -1e-12) | (r > r_max + 1e-12)

    if np.any(bad):
        message = f"Some r values are outside the valid interval [0, r_max={r_max:.6g}]."
        if invalid == "raise":
            raise ValueError(message)
        if invalid == "nan":
            r_work = r.copy()
            r_work[bad] = np.nan
        elif invalid == "clip":
            r_work = np.clip(r, 0.0, r_max)
        else:
            raise ValueError("invalid must be 'nan', 'clip', or 'raise'.")
    else:
        r_work = r

    numerator = 2.0 * coeffs.J3 * r_work
    denominator = coeffs.A - r_work * coeffs.B

    tan2_theta = numerator / denominator
    # Numerical roundoff can make tan2 very slightly negative at r=0.
    tan2_theta = np.where(tan2_theta < 0.0, np.nan, tan2_theta)
    return np.arctan(np.sqrt(tan2_theta))


def theta_curve(coeffs: AngularIntegrals, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate r and theta_d arrays over the valid range for one configuration.
    """
    r = np.linspace(0.0, coeffs.r_max * (1.0 - 1e-9), points)
    theta = theta_from_r(r, coeffs)
    return r, theta


def parse_media(media_args: Iterable[str]) -> list[Medium]:
    """
    Parse strings of the form 'name:n' into Medium objects.
    """
    media = []
    for item in media_args:
        if ":" not in item:
            raise ValueError(
                f"Could not parse medium '{item}'. Use the form name:n, e.g. water:1.333."
            )
        name, n_text = item.split(":", 1)
        name = name.strip()
        n = float(n_text)
        if not name:
            raise ValueError(f"Medium name cannot be empty in '{item}'.")
        if n <= 0:
            raise ValueError(f"Refractive index must be positive in '{item}'.")
        media.append(Medium(name=name, n=n))
    return media


def label_for_coeffs(coeffs: AngularIntegrals) -> str:
    """
    Human-readable label for plots.
    """
    if coeffs.hole_mode == "rho":
        hole_label = rf"$\rho={coeffs.rho_effective:.3g}$"
    else:
        hole_label = rf"hole NA={coeffs.na_hole_effective:.3g}"

    return (
        f"{coeffs.medium.name}, n={coeffs.medium.n:g}, "
        f"{hole_label}, rmax={coeffs.r_max:.3f}"
    )


def write_coefficients_csv(coeffs_list: list[AngularIntegrals], path: Path) -> None:
    """
    Write a CSV table of angular coefficients and angular limits.
    """
    fields = [
        "medium",
        "n_medium",
        "na_out",
        "hole_mode",
        "input_hole_value",
        "rho_effective",
        "na_hole_effective",
        "theta_in_deg",
        "theta_max_deg",
        "J1",
        "J2",
        "J3",
        "J1_minus_J2",
        "J1_plus_J2",
        "r_max",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in coeffs_list:
            writer.writerow(
                {
                    "medium": c.medium.name,
                    "n_medium": c.medium.n,
                    "na_out": c.na_out,
                    "hole_mode": c.hole_mode,
                    "input_hole_value": c.hole_value,
                    "rho_effective": c.rho_effective,
                    "na_hole_effective": c.na_hole_effective,
                    "theta_in_deg": math.degrees(c.theta_in),
                    "theta_max_deg": math.degrees(c.theta_max),
                    "J1": c.J1,
                    "J2": c.J2,
                    "J3": c.J3,
                    "J1_minus_J2": c.A,
                    "J1_plus_J2": c.B,
                    "r_max": c.r_max,
                }
            )


def make_theta_vs_r_plot(
    coeffs_list: list[AngularIntegrals],
    output_path: Path,
    points: int = 500,
    degrees: bool = True,
) -> None:
    """
    Plot reconstructed theta_d as a function of anisotropy radius r.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for coeffs in coeffs_list:
        r, theta = theta_curve(coeffs, points=points)
        y = np.degrees(theta) if degrees else theta
        ax.plot(r, y, label=label_for_coeffs(coeffs))

    ax.set_xlabel(r"anisotropy radius $r=\sqrt{X^2+Y^2}$")
    ax.set_ylabel(r"reconstructed polar angle $\theta_d$ / degrees" if degrees else r"$\theta_d$ / rad")
    ax.set_title(r"Dipole finite-NA correction: $\theta_d(r)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_error_plot_against_no_hole(
    coeffs_list: list[AngularIntegrals],
    output_path: Path,
    points: int = 500,
) -> None:
    """
    Plot theta_hole / theta_no_hole for each medium and hole, where the no-hole
    reference is computed for the same medium and NA_out.

    This visualizes the error from ignoring the central hole.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for coeffs in coeffs_list:
        if abs(coeffs.rho_effective) < 1e-15:
            continue

        no_hole = angular_integrals(
            na_out=coeffs.na_out,
            n_medium=coeffs.medium.n,
            hole=0.0,
            hole_mode="rho",
            medium_name=coeffs.medium.name,
        )

        # Only compare over the radius range valid for both curves.
        r_stop = min(coeffs.r_max, no_hole.r_max) * (1.0 - 1e-9)
        r = np.linspace(1e-9, r_stop, points)

        theta_hole = theta_from_r(r, coeffs)
        theta_no_hole = theta_from_r(r, no_hole)
        ratio = theta_hole / theta_no_hole

        ax.plot(r, ratio, label=label_for_coeffs(coeffs))

    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"anisotropy radius $r=\sqrt{X^2+Y^2}$")
    ax.set_ylabel(r"$\theta_d^{(\mathrm{hole})} / \theta_d^{(\mathrm{no\ hole})}$")
    ax.set_title("Effect of ignoring the central BFP hole")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def print_coefficients(coeffs_list: list[AngularIntegrals]) -> None:
    """
    Print a readable coefficient summary.
    """
    print("\nAngular coefficients")
    print("=" * 100)
    header = (
        f"{'medium':<16} {'n':>7} {'NA':>7} {'hole':>12} "
        f"{'theta_in':>10} {'theta_max':>10} {'J1':>12} {'J2':>12} {'J3':>12} {'rmax':>10}"
    )
    print(header)
    print("-" * len(header))

    for c in coeffs_list:
        if c.hole_mode == "rho":
            hole_text = f"rho={c.rho_effective:.4g}"
        else:
            hole_text = f"NA={c.na_hole_effective:.4g}"

        print(
            f"{c.medium.name:<16} {c.medium.n:>7.4f} {c.na_out:>7.3f} {hole_text:>12} "
            f"{math.degrees(c.theta_in):>10.3f} {math.degrees(c.theta_max):>10.3f} "
            f"{c.J1:>12.8f} {c.J2:>12.8f} {c.J3:>12.8f} {c.r_max:>10.6f}"
        )

    print("=" * 100)


def demo_reconstruction(coeffs: AngularIntegrals) -> None:
    """
    Small sanity check showing phi is recovered and theta is recovered from r.
    """
    theta_true = math.radians(55.0)
    phi_true = math.radians(31.0)

    I0, I90, I45, I135 = intensities_from_orientation(theta_true, phi_true, coeffs)
    X, Y, r = anisotropy_from_intensities(I0, I90, I45, I135)

    phi_rec = phi_from_xy(X, Y)
    theta_rec = theta_from_r(r, coeffs)

    print("\nSanity-check reconstruction using first configuration")
    print("=" * 100)
    print(f"true theta_d = {math.degrees(theta_true):.6f} deg")
    print(f"rec. theta_d = {float(np.degrees(theta_rec)):.6f} deg")
    print(f"true phi_d   = {math.degrees(phi_true):.6f} deg")
    print(f"rec. phi_d   = {float(np.degrees(phi_rec)):.6f} deg, modulo 180 deg rod symmetry")
    print(f"X = {float(X):.8f}, Y = {float(Y):.8f}, r = {float(r):.8f}")
    print("=" * 100)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finite-NA dipole emission model with a circular central BFP cutout."
    )

    parser.add_argument(
        "--na-out",
        type=float,
        default=1.4,
        help="Outer numerical aperture. Default: 1.4",
    )
    parser.add_argument(
        "--hole-mode",
        choices=["rho", "na"],
        default="rho",
        help=(
            "Interpretation of --holes. "
            "'rho' = fractional BFP radius, sin(theta_in)=rho sin(theta_max). "
            "'na' = physical central-hole NA, sin(theta_in)=NA_hole/n. "
            "Default: rho"
        ),
    )
    parser.add_argument(
        "--holes",
        type=float,
        nargs="+",
        default=[0.0, 0.25 / 1.4, 0.39],
        help=(
            "Central hole sizes. Default with --hole-mode rho is: "
            "0, 0.25/1.4, 0.39. "
            "Use --hole-mode na if these are physical NA values."
        ),
    )
    parser.add_argument(
        "--media",
        type=str,
        nargs="+",
        default=["water:1.333", "glycerol50:1.398"],
        help=(
            "Media as name:n pairs. Defaults: water:1.333 glycerol50:1.398. "
            "Example: --media water:1.333 oil:1.515"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("dipole_finite_na_outputs"),
        help="Output directory. Default: dipole_finite_na_outputs",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=600,
        help="Number of r samples per curve. Default: 600",
    )
    parser.add_argument(
        "--no-error-plot",
        action="store_true",
        help="Do not create the theta_hole/theta_no_hole error plot.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a simple forward/inverse reconstruction sanity check.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    media = parse_media(args.media)
    args.outdir.mkdir(parents=True, exist_ok=True)

    coeffs_list: list[AngularIntegrals] = []
    for medium in media:
        for hole in args.holes:
            coeffs_list.append(
                angular_integrals(
                    na_out=args.na_out,
                    n_medium=medium.n,
                    hole=hole,
                    hole_mode=args.hole_mode,
                    medium_name=medium.name,
                )
            )

    print_coefficients(coeffs_list)

    csv_path = args.outdir / "angular_coefficients.csv"
    theta_plot_path = args.outdir / "theta_vs_r.png"
    error_plot_path = args.outdir / "theta_ratio_vs_r_ignoring_hole.png"

    write_coefficients_csv(coeffs_list, csv_path)
    make_theta_vs_r_plot(coeffs_list, theta_plot_path, points=args.points)

    print(f"\nWrote coefficient table: {csv_path}")
    print(f"Wrote theta-vs-r plot:   {theta_plot_path}")

    if not args.no_error_plot:
        make_error_plot_against_no_hole(coeffs_list, error_plot_path, points=args.points)
        print(f"Wrote error-ratio plot:  {error_plot_path}")

    if args.demo and coeffs_list:
        demo_reconstruction(coeffs_list[0])


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Z_PERIOD_NM = 220.0
DRIVE_FREQUENCY_HZ = 900.0
DRIVE_PERIOD_S = 1.0 / DRIVE_FREQUENCY_HZ
AMPLITUDES_NM = [Z_PERIOD_NM / 4.0, Z_PERIOD_NM / 2.0, Z_PERIOD_NM]
SAMPLES = 4000
OUTPUT_DIR = Path("datasets") / "z_drive_intensity_simulation"


def triangle_wave_unit(t: np.ndarray, period: float) -> np.ndarray:
    """Triangle wave in [-1, 1], starting at 0 phase minimum."""
    phase = (t / period) % 1.0
    return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0


def sine_wave_unit(t: np.ndarray, period: float) -> np.ndarray:
    """Sine wave in [-1, 1], starting at zero and rising."""
    return np.sin(2.0 * np.pi * t / period)


def square_wave_unit(t: np.ndarray, period: float) -> np.ndarray:
    """Square wave in [-1, 1], starting high."""
    phase = (t / period) % 1.0
    return np.where(phase < 0.5, 1.0, -1.0)


def intensity_from_z(z_nm: np.ndarray, period_nm: float) -> np.ndarray:
    """Sinusoidal I(z), normalized to [0, 1]."""
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * z_nm / period_nm))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, DRIVE_PERIOD_S, SAMPLES, endpoint=True)
    drive_shapes = {
        "triangle": triangle_wave_unit(t, DRIVE_PERIOD_S),
        "sine": sine_wave_unit(t, DRIVE_PERIOD_S),
        "square": square_wave_unit(t, DRIVE_PERIOD_S),
    }

    for shape_name, drive_unit in drive_shapes.items():
        traces = []
        for amp_nm in AMPLITUDES_NM:
            z_nm = amp_nm * drive_unit
            intensity = intensity_from_z(z_nm, Z_PERIOD_NM)
            traces.append((amp_nm, z_nm, intensity))

            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axes[0].plot(t * 1e3, z_nm, color="tab:blue")
            axes[0].set_ylabel("z(t) / nm")
            axes[0].set_title(f"{shape_name.capitalize()} z drive, middle-to-peak = {amp_nm:.0f} nm")
            axes[0].grid(alpha=0.25)

            axes[1].plot(t * 1e3, intensity, color="tab:red")
            axes[1].set_xlabel("time / ms")
            axes[1].set_ylabel("normalized I(t)")
            axes[1].set_title("I(t) from sinusoidal I(z), z period = 220 nm")
            axes[1].grid(alpha=0.25)

            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"I_t_{shape_name}_amp_{amp_nm:.0f}nm.png", dpi=220)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for amp_nm, _z_nm, intensity in traces:
            ax.plot(t * 1e3, intensity, label=f"middle-to-peak = {amp_nm:.0f} nm")
        ax.set_xlabel("time / ms")
        ax.set_ylabel("normalized I(t)")
        ax.set_title(f"Intensity modulation for 900 Hz {shape_name} z drives")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"I_t_{shape_name}_all_amplitudes.png", dpi=220)
        plt.close(fig)

    for amp_nm in AMPLITUDES_NM:
        fig, ax = plt.subplots(figsize=(9, 5))
        for shape_name, drive_unit in drive_shapes.items():
            intensity = intensity_from_z(amp_nm * drive_unit, Z_PERIOD_NM)
            ax.plot(t * 1e3, intensity, label=shape_name)
        ax.set_xlabel("time / ms")
        ax.set_ylabel("normalized I(t)")
        ax.set_title(f"Intensity modulation by drive shape, amplitude = {amp_nm:.0f} nm")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"I_t_all_shapes_amp_{amp_nm:.0f}nm.png", dpi=220)
        plt.close(fig)

    print(f"Saved plots to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

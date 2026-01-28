import numpy as np
import json
import os
import imageio.v2 as imageio
from dataclasses import dataclass, asdict
from typing import List, Tuple


# =========================
# Configuration
# =========================

FPS = 80
N_FRAMES = 80

POLAR_W = 2464   # camera width
POLAR_H = 2056   # camera height

INT_W = POLAR_W // 2
INT_H = POLAR_H // 2

TARGET_MAX = 240.0

BACKGROUND_MAX = 50.0
NOISE_MAX = 5.0

I_TOT_MIN = 399.0
I_TOT_MAX = 1023.0

SIGMA_MIN = 1.0
SIGMA_MAX = 3.0

FREQ_MIN = 0.1     # Hz
FREQ_MAX = 1000.0  # Hz


# =========================
# Ground truth structure
# =========================

@dataclass
class ObjectTruth:
    id: int
    category: str   # "non_gold_static" | "gold_static" | "gold_rotating"
    center_xy: Tuple[float, float]   # intensity coords
    sigma: float
    I_tot: float
    theta0: float
    phi0: float
    frequency: float


# =========================
# Polarization coefficients
# =========================

def compute_ABC(NA: float, nw: float):
    alpha = np.arcsin(NA / nw)
    c = np.cos(alpha)

    A = 1/6  - 1/4 * c + 1/12 * c**3
    B = 1/8  * c     - 1/8  * c**3
    C = 7/48 - 1/16 * c - 1/16 * c**2 - 1/48 * c**3
    return A, B, C


# =========================
# Object generation
# =========================

def sample_frequency():
    logf = np.random.uniform(np.log10(FREQ_MIN), np.log10(FREQ_MAX))
    return 10**logf


def generate_objects(n_objects: int) -> List[ObjectTruth]:
    objects = []

    categories = (
        ["non_gold_static"] * 5 +
        ["gold_static"] * 5 +
        ["gold_rotating"] * 10
    )

    while len(categories) < n_objects:
        categories.append(np.random.choice(
            ["non_gold_static", "gold_static", "gold_rotating"]
        ))

    np.random.shuffle(categories)

    margin = 10

    for obj_id, cat in enumerate(categories):
        cx = np.random.uniform(margin, INT_W - margin)
        cy = np.random.uniform(margin, INT_H - margin)

        sigma = np.random.uniform(SIGMA_MIN, SIGMA_MAX)
        I_tot = np.random.uniform(I_TOT_MIN, I_TOT_MAX)

        theta0 = np.random.uniform(0, np.pi / 2)
        phi0 = np.random.uniform(0, 2 * np.pi)

        freq = sample_frequency() if cat == "gold_rotating" else 0.0

        objects.append(ObjectTruth(
            id=obj_id,
            category=cat,
            center_xy=(cx, cy),
            sigma=sigma,
            I_tot=I_tot,
            theta0=theta0,
            phi0=phi0,
            frequency=freq
        ))

    return objects


# =========================
# Simulation core
# =========================

def simulate_movie(
    objects: List[ObjectTruth],
    NA: float = 1.3,
    nw: float = 1.33,
):
    A, B, C = compute_ABC(NA, nw)
    t = np.arange(N_FRAMES) / FPS

    P = np.zeros((N_FRAMES, POLAR_H, POLAR_W), dtype=np.float32)

    for obj in objects:
        cx, cy = obj.center_xy
        sigma = obj.sigma
        I_tot = obj.I_tot

        r = int(np.ceil(3 * sigma))
        x0 = max(int(cx) - r, 0)
        x1 = min(int(cx) + r + 1, INT_W)
        y0 = max(int(cy) - r, 0)
        y1 = min(int(cy) + r + 1, INT_H)

        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        psf = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        psf *= I_tot

        sin2t = np.sin(obj.theta0)**2
        phi_t = 2 * np.pi * obj.frequency * t + obj.phi0

        if obj.category == "non_gold_static":
            f0 = f45 = f90 = f135 = np.ones_like(t)
        else:
            f0   = A + B * sin2t + C * sin2t * np.cos(2 * phi_t)
            f45  = A + B * sin2t + C * sin2t * np.sin(2 * phi_t)
            f90  = A + B * sin2t - C * sin2t * np.cos(2 * phi_t)
            f135 = A + B * sin2t - C * sin2t * np.sin(2 * phi_t)

        for k in range(N_FRAMES):
            for iy, y in enumerate(range(y0, y1)):
                py = 2 * y
                for ix, x in enumerate(range(x0, x1)):
                    px = 2 * x
                    val = psf[iy, ix]

                    P[k, py,     px    ] += val * f90[k]
                    P[k, py,     px + 1] += val * f45[k]
                    P[k, py + 1, px    ] += val * f135[k]
                    P[k, py + 1, px + 1] += val * f0[k]

    background = np.random.uniform(
        0, BACKGROUND_MAX, size=(POLAR_H, POLAR_W)
    ).astype(np.float32)

    P += background[None, :, :]

    scale = TARGET_MAX / P.max()
    P *= scale

    noise = np.random.uniform(
        0, NOISE_MAX, size=P.shape
    ).astype(np.float32)

    P += noise

    movie = np.clip(P, 0, 255).astype(np.uint8)
    return movie


# =========================
# Save utilities
# =========================

def save_outputs(movie, objects, seed):
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f"movie_{seed}.npy")
    json_path = os.path.join(out_dir, f"truth_{seed}.json")
    mp4_path = os.path.join(out_dir, f"movie_{seed}.mp4")

    np.save(npy_path, movie)

    with open(json_path, "w") as f:
        json.dump([asdict(o) for o in objects], f, indent=2)

    writer = imageio.get_writer(
        mp4_path,
        fps=FPS,
        codec="libx264",
        format="ffmpeg"
    )
    for frame in movie:
        writer.append_data(frame)
    writer.close()

    print(f"Saved:")
    print(f"  {npy_path}")
    print(f"  {json_path}")
    print(f"  {mp4_path}")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    seed = np.random.randint(0, 1_000_000)
    np.random.seed(seed)

    n_objects = np.random.randint(30, 51)
    objects = generate_objects(n_objects)
    movie = simulate_movie(objects)

    save_outputs(movie, objects, seed)

"""
Heart-shaped stellarator — animated rotating GIF.

Same surface design and coil optimization as heart_stellarator.py; after the
optimizer converges we render N camera angles rotating around the device and
stitch the frames into a GIF with Pillow.
"""

import os
import io
import numpy as np
import matplotlib
if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize

from simsopt.geo import (
    SurfaceRZFourier, create_equally_spaced_curves,
    CurveLength, LpCurveCurvature, MeanSquaredCurvature,
    ArclengthVariation, CurveCurveDistance,
)
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, QuadraticPenalty


# --- 1. Heart surface ------------------------------------------------------
nfp = 2
R0 = 1.0
A = 0.018

nphi, ntheta = 48, 48
s = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=5, ntor=2,
    quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
)
s.set_rc(0, 0, R0)
s.set_rc(1, 0,  A * 13.0)
s.set_rc(2, 0, -A *  5.0)
s.set_rc(3, 0, -A *  2.0)
s.set_rc(4, 0, -A *  1.0)
s.set_zs(1, 0,  A * 12.0)
s.set_zs(3, 0, -A *  4.0)
TWIST = 0.035
s.set_rc(1, 1,  TWIST)
s.set_zs(1, 1, -TWIST)


# --- 2. Coils + optimization ----------------------------------------------
ncoils = 5
base_curves = create_equally_spaced_curves(
    ncoils, nfp, stellsym=True, R0=R0, R1=0.55, order=12)
base_currents = [Current(1.0e5) for _ in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jcs = sum(LpCurveCurvature(c, p=2, threshold=25.0) for c in base_curves)
Jmsc = sum(MeanSquaredCurvature(c) for c in base_curves)
Jarc = sum(ArclengthVariation(c) for c in base_curves)
Jdist = CurveCurveDistance(base_curves, 0.12)
JF = (Jf
      + 5e-4 * sum(QuadraticPenalty(Jl, 3.0, "max") for Jl in Jls)
      + 1e-7 * Jcs + 1e-8 * Jmsc + 1e-4 * Jarc + 5e-2 * Jdist)


def fun(dofs):
    JF.x = dofs
    return float(JF.J()), JF.dJ()


print("Optimizing coils ...")
minimize(fun, JF.x, jac=True, method="L-BFGS-B",
         options={"maxiter": 800, "maxcor": 300, "gtol": 1e-12, "ftol": 1e-14})
print(f"  done — Jf = {Jf.J():.3e}")


# --- 3. Hi-res surface for rendering ---------------------------------------
NPHI_PLOT, NTHETA_PLOT = nfp * 96, 128
full = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
    quadpoints_phi=np.linspace(0, 1.0, NPHI_PLOT, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, NTHETA_PLOT, endpoint=False),
)
full.x = s.x
G = full.gamma()
coil_paths = [np.vstack([c.curve.gamma(), c.curve.gamma()[:1]]) for c in coils]


# --- 4. Render rotation frames ---------------------------------------------
N_FRAMES = 48
FPS = 18
frames = []
L = R0 + 0.45

fig = plt.figure(figsize=(7, 5.6))
ax = fig.add_subplot(111, projection="3d")

print(f"Rendering {N_FRAMES} frames ...")
for k in range(N_FRAMES):
    ax.clear()
    ax.plot_surface(G[:, :, 0], G[:, :, 1], G[:, :, 2],
                    color="#e63946", alpha=0.72, linewidth=0,
                    antialiased=True, shade=True)
    for g in coil_paths:
        ax.plot(g[:, 0], g[:, 1], g[:, 2], color="#1d3557", linewidth=1.1)

    azim = 360.0 * k / N_FRAMES
    elev = 18 + 8 * np.sin(2 * np.pi * k / N_FRAMES)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.45))
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-0.45, 0.45)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"Heart stellarator  (nfp={nfp}, R0={R0} m, "
                 f"{len(coils)} coils)")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    # Convert to 'P' (palette) mode for compact GIF
    img = Image.open(buf).convert("RGB").convert(
        "P", palette=Image.ADAPTIVE, colors=128)
    frames.append(img)
    buf.close()
    if (k + 1) % 8 == 0:
        print(f"  {k+1}/{N_FRAMES}")

plt.close(fig)


# --- 5. Save GIF -----------------------------------------------------------
out = "heart_stellarator.gif"
frames[0].save(
    out, save_all=True, append_images=frames[1:],
    duration=int(1000 / FPS), loop=0, optimize=True, disposal=2,
)
size_kb = os.path.getsize(out) / 1024
print(f"  saved {out}  ({N_FRAMES} frames, {FPS} fps, {size_kb:.0f} KB)")

"""
Heart-shaped stellarator.

The classical parametric heart curve
    x(t) = 16 sin^3(t)              = 12 sin(t) - 4 sin(3t)
    y(t) = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t
is already a tiny finite Fourier series — so it drops straight into a
SurfaceRZFourier with no fitting.  Rotated 90 deg so the heart's symmetry
axis lies in the equatorial plane (Z = 0), the cross-section becomes
stellarator-symmetric: dip + lobes face outboard (large R), tip points
toward the central axis (small R).

A small n=1 perturbation twists the heart as it goes around — that's what
turns a heart-tokamak into an actual non-axisymmetric stellarator.

Outputs:
    heart_stellarator.png   — 3-D view of plasma + optimized coils
    heart_cross_sections.png — poloidal cross-sections at several phi
"""

import os
import numpy as np
import matplotlib
if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from simsopt.geo import (
    SurfaceRZFourier,
    create_equally_spaced_curves,
    CurveLength,
    LpCurveCurvature,
    MeanSquaredCurvature,
    ArclengthVariation,
    CurveCurveDistance,
)
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, QuadraticPenalty


# ---------------------------------------------------------------------------
# 1. Heart-shaped plasma cross-section
# ---------------------------------------------------------------------------
nfp = 2
R0 = 1.0
A = 0.018          # heart scale; cross-section spans ~0.4 m

nphi, ntheta = 48, 48
s = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=5, ntor=2,
    quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
)

# R(theta) = R0 + A * (13 cos t - 5 cos 2t - 2 cos 3t - cos 4t)
# Z(theta) = A * (12 sin t - 4 sin 3t)
s.set_rc(0, 0, R0)
s.set_rc(1, 0,  A * 13.0)
s.set_rc(2, 0, -A *  5.0)
s.set_rc(3, 0, -A *  2.0)
s.set_rc(4, 0, -A *  1.0)
s.set_zs(1, 0,  A * 12.0)
s.set_zs(3, 0, -A *  4.0)

# Small toroidal twist (rotating-shape stellarator term)
TWIST = 0.035
s.set_rc(1, 1,  TWIST)
s.set_zs(1, 1, -TWIST)


# ---------------------------------------------------------------------------
# 2. Modular coils
# ---------------------------------------------------------------------------
ncoils = 5
order = 12
base_curves = create_equally_spaced_curves(
    ncoils, nfp, stellsym=True, R0=R0, R1=0.55, order=order
)
base_currents = [Current(1.0e5) for _ in range(ncoils)]
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))


# ---------------------------------------------------------------------------
# 3. Optimize
# ---------------------------------------------------------------------------
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jcs = sum(LpCurveCurvature(c, p=2, threshold=25.0) for c in base_curves)
Jmsc = sum(MeanSquaredCurvature(c) for c in base_curves)
Jarc = sum(ArclengthVariation(c) for c in base_curves)
Jdist = CurveCurveDistance(base_curves, 0.12)

LENGTH_TARGET = 3.0
LENGTH_W = 5e-4
CURV_W = 1e-7
MSC_W = 1e-8
ARC_W = 1e-4
DIST_W = 5e-2

JF = (
    Jf
    + LENGTH_W * sum(QuadraticPenalty(Jl, LENGTH_TARGET, "max") for Jl in Jls)
    + CURV_W * Jcs
    + MSC_W * Jmsc
    + ARC_W * Jarc
    + DIST_W * Jdist
)


def fun(dofs):
    JF.x = dofs
    return float(JF.J()), JF.dJ()


print(f"Heart stellarator:  nfp = {nfp},  R0 = {R0} m")
print(f"Heart scale A = {A}  (cross-section ~ "
      f"{A*(13+17):.2f} m radial x {A*32:.2f} m vertical)")
print(f"\nInitial flux objective Jf = {Jf.J():.3e}")
print("Running L-BFGS-B ...")
res = minimize(
    fun, JF.x, jac=True, method="L-BFGS-B",
    options={"maxiter": 800, "maxcor": 300, "gtol": 1e-12, "ftol": 1e-14},
)
print(f"  converged in {res.nit} iter,  J = {res.fun:.3e},  Jf = {Jf.J():.3e}")
print(f"  mean coil length = {np.mean([Jl.J() for Jl in Jls]):.3f} m")

bs.set_points(s.gamma().reshape((-1, 3)))
B = bs.B().reshape((nphi, ntheta, 3))
n = s.unitnormal()
absB = np.linalg.norm(B, axis=2)
Bn = np.sum(B * n, axis=2)
rel = np.abs(Bn) / absB
print(f"  <|B.n|/|B|> = {rel.mean():.3e},  max = {rel.max():.3e}")


# ---------------------------------------------------------------------------
# 4. 3-D plot of full torus + coils, hi-res surface so the heart shape shows
# ---------------------------------------------------------------------------
NPHI_PLOT, NTHETA_PLOT = nfp * 96, 128
full = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
    quadpoints_phi=np.linspace(0, 1.0, NPHI_PLOT, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, NTHETA_PLOT, endpoint=False),
)
full.x = s.x
G = full.gamma()

coil_paths = []
for c in coils:
    g = c.curve.gamma()
    coil_paths.append(np.vstack([g, g[:1]]))

# Half-torus cut-away so the heart cross-section is visible at the cut
mask = G[:, :, 1] >= -0.02
Gx = np.where(mask, G[:, :, 0], np.nan)
Gy = np.where(mask, G[:, :, 1], np.nan)
Gz = np.where(mask, G[:, :, 2], np.nan)

fig = plt.figure(figsize=(15, 8))

# --- Panel A: oblique 3-D, full torus + coils ---------------------------
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(G[:, :, 0], G[:, :, 1], G[:, :, 2],
                 color="#e63946", alpha=0.65, linewidth=0, antialiased=True,
                 shade=True)
for g in coil_paths:
    ax1.plot(g[:, 0], g[:, 1], g[:, 2], color="#1d3557", linewidth=1.3)
ax1.view_init(elev=18, azim=35)
ax1.set_box_aspect((1, 1, 0.45))
L = R0 + 0.45
ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-0.45, 0.45)
ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_zlabel("z [m]")
ax1.set_title("Full plasma surface + optimized coils")

# --- Panel B: half-torus cut-away so the heart pops at the cut ----------
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(Gx, Gy, Gz,
                 color="#e63946", alpha=0.85, linewidth=0, antialiased=True,
                 shade=True)
# Draw the cut cross-section curve (phi = 0 and phi = pi) heavily so you
# see the heart from inside.
for phi_norm in [0.0, 0.5]:
    cross = SurfaceRZFourier(
        nfp=nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
        quadpoints_phi=np.array([phi_norm]),
        quadpoints_theta=np.linspace(0, 1.0, 400, endpoint=True),
    )
    cross.x = s.x
    gx, gy, gz = cross.gamma()[0].T
    ax2.plot(gx, gy, gz, color="#7a0010", linewidth=2.5)
for g in coil_paths:
    ax2.plot(g[:, 0], g[:, 1], g[:, 2], color="#1d3557", linewidth=1.1,
             alpha=0.55)
ax2.view_init(elev=22, azim=-60)
ax2.set_box_aspect((1, 1, 0.45))
ax2.set_xlim(-L, L); ax2.set_ylim(-L, L); ax2.set_zlim(-0.45, 0.45)
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]"); ax2.set_zlabel("z [m]")
ax2.set_title("Cut-away — heart cross-section visible at the cut")

plt.suptitle(
    f"Heart stellarator — nfp={nfp},  R0={R0} m,  {len(coils)} optimized "
    f"coils,  $\\langle|B\\cdot n|/|B|\\rangle$ = {rel.mean()*100:.2f}%",
    fontsize=13, y=1.0,
)
plt.tight_layout()
plt.savefig("heart_stellarator.png", dpi=160, bbox_inches="tight")
print("  saved heart_stellarator.png")


# ---------------------------------------------------------------------------
# 5. Poloidal cross-sections at several phi (so the heart is visible)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 7))
phis = np.linspace(0, 0.5 / nfp, 5)
colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(phis)))
for phi_norm, col in zip(phis, colors):
    cross = SurfaceRZFourier(
        nfp=nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
        quadpoints_phi=np.array([phi_norm]),
        quadpoints_theta=np.linspace(0, 1.0, 400, endpoint=True),
    )
    cross.x = s.x
    g = cross.gamma()[0]
    R = np.sqrt(g[:, 0]**2 + g[:, 1]**2)
    Z = g[:, 2]
    ax2.plot(R, Z, color=col, linewidth=2,
             label=r"$\phi$ = " + f"{phi_norm * 360:.0f}" + r"$^\circ$/nfp")

ax2.set_aspect("equal")
ax2.set_xlabel("R [m]"); ax2.set_ylabel("Z [m]")
ax2.set_title("Heart cross-sections at several toroidal angles")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("heart_cross_sections.png", dpi=150, bbox_inches="tight")
print("  saved heart_cross_sections.png")

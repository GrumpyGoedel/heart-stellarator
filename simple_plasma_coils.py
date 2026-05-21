"""
Design a simple plasma flux surface from scratch, then optimize a fresh set of
modular coils to produce B.n = 0 on that surface.

The plasma surface is a 3-field-period rotating ellipse — a textbook minimal
stellarator boundary.  The equilibrium / coil-optimization stack runs FORTRAN
under the hood (VMEC2000, biot-savart kernels in SIMSOPT) — see the Dockerfile
for the gfortran + netCDF build environment.

Output:
    simple_plasma_coils.png   — 3-D rendering of plasma surface + coils
    simple_plasma_bn.png      — colormap of |B.n|/|B| on the surface
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
)
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, QuadraticPenalty


# ---------------------------------------------------------------------------
# 1. Design a simple plasma flux surface
#    3-period rotating ellipse: circular cross-section that rotates as it goes
#    around the torus.  R0 = 1.0 m, a = 0.25 m, elongation 1.3.
# ---------------------------------------------------------------------------
nfp = 3
R0 = 1.0
a = 0.25
elong = 1.3

nphi, ntheta = 32, 32
s = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=3, ntor=3,
    quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
)
s.set_rc(0, 0, R0)                              # major radius
s.set_rc(1, 0, a)                               # circular minor radius
s.set_zs(1, 0, a * elong)                       # vertical stretch -> elongation
s.set_rc(1, 1, 0.04)                            # rotating-ellipse coupling
s.set_zs(1, 1, -0.04)

print("Plasma surface designed:")
print(f"  nfp = {nfp},  R0 = {R0} m,  a = {a} m,  elongation = {elong}")


# ---------------------------------------------------------------------------
# 2. Initialize modular coils
# ---------------------------------------------------------------------------
ncoils = 4
order = 8
base_curves = create_equally_spaced_curves(
    ncoils, nfp, stellsym=True, R0=R0, R1=2.2 * a, order=order
)
base_currents = [Current(1.0e5) for _ in range(ncoils)]
base_currents[0].fix_all()  # remove global current-scaling gauge

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))


# ---------------------------------------------------------------------------
# 3. Optimize
# ---------------------------------------------------------------------------
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jcs = sum(LpCurveCurvature(c, p=2, threshold=12.0) for c in base_curves)
Jmsc = sum(MeanSquaredCurvature(c) for c in base_curves)
Jarc = sum(ArclengthVariation(c) for c in base_curves)

LENGTH_TARGET = 2.5      # per coil, metres
LENGTH_W = 1e-3
CURV_W = 1e-6
MSC_W = 1e-7
ARC_W = 1e-4

JF = (
    Jf
    + LENGTH_W * sum(QuadraticPenalty(Jl, LENGTH_TARGET, "max") for Jl in Jls)
    + CURV_W * Jcs
    + MSC_W * Jmsc
    + ARC_W * Jarc
)


def fun(dofs):
    JF.x = dofs
    return float(JF.J()), JF.dJ()


print(f"\nInitial flux objective Jf = {Jf.J():.3e}")
print("Running L-BFGS-B ...")
res = minimize(
    fun, JF.x, jac=True, method="L-BFGS-B",
    options={"maxiter": 400, "maxcor": 300, "gtol": 1e-12, "ftol": 1e-14},
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
# 4. Plot: 3-D view of plasma + coils
# ---------------------------------------------------------------------------
full = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
    quadpoints_phi=np.linspace(0, 1.0, nfp * nphi * 2, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
)
full.x = s.x
G = full.gamma()

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    G[:, :, 0], G[:, :, 1], G[:, :, 2],
    color="#2ca02c", alpha=0.45, linewidth=0, antialiased=True,
)
for c in coils:
    g = c.curve.gamma()
    g = np.vstack([g, g[:1]])
    ax.plot(g[:, 0], g[:, 1], g[:, 2], color="#d62728", linewidth=1.8)

ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title(f"Designed plasma (nfp={nfp} rotating ellipse) + optimized coils")
ax.set_box_aspect((1, 1, 0.35))
L = R0 + 3 * a
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-0.6, 0.6)
plt.tight_layout()
plt.savefig("simple_plasma_coils.png", dpi=150, bbox_inches="tight")
print("  saved simple_plasma_coils.png")


# ---------------------------------------------------------------------------
# 5. Plot: |B.n|/|B| heatmap on (phi, theta)
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
im = ax2.imshow(
    rel.T, origin="lower", aspect="auto",
    extent=[0, 1.0 / nfp, 0, 1.0], cmap="magma",
)
ax2.set_xlabel(r"$\phi$ / (1/nfp)")
ax2.set_ylabel(r"$\theta$ / (2$\pi$)")
ax2.set_title(r"$|B\cdot n|/|B|$ on optimized plasma surface")
fig2.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig("simple_plasma_bn.png", dpi=150, bbox_inches="tight")
print("  saved simple_plasma_bn.png")

plt.show()

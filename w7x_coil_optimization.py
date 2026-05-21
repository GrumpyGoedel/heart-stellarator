"""
Stage-two coil optimization for Wendelstein 7-X using SIMSOPT.

Loads the real W7-X coils + Biot-Savart field bundled with SIMSOPT, builds a
W7-X-shaped target plasma boundary, then optimizes a fresh set of modular
coils to produce B.n = 0 on that boundary, and finally plots plasma + coils.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from simsopt.configs import get_data
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
# 1. Load the W7-X reference configuration (coils + Biot-Savart + axis)
# ---------------------------------------------------------------------------
print("Loading Wendelstein 7-X reference configuration ...")
ref_curves, ref_currents, magnetic_axis, nfp, bs_ref = get_data(
    "w7x", coil_order=24, points_per_period=4, magnetic_axis_order=10
)
ref_coils = coils_via_symmetries(ref_curves, ref_currents, nfp, True)
print(f"  nfp = {nfp}, {len(ref_curves)} unique coils, {len(ref_coils)} total")


# ---------------------------------------------------------------------------
# 2. Build a W7-X-shaped target plasma boundary
#    (SurfaceRZFourier with W7-X major radius ~5.5 m, minor radius ~0.5 m,
#     bean-shaped cross-section through field-period rotation).
# ---------------------------------------------------------------------------
nphi, ntheta = 32, 32
s = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=4, ntor=4,
    quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
)
# Major / minor radius
s.set_rc(0, 0, 5.5)
s.set_rc(1, 0, 0.49)
s.set_zs(1, 0, 0.49)
# W7-X-like bean shaping (toroidal modulation of cross-section)
s.set_rc(0, 1, 0.30)
s.set_zs(0, 1, -0.30)
s.set_rc(1, 1, -0.06)
s.set_zs(1, 1, 0.06)


# ---------------------------------------------------------------------------
# 3. Initialize a fresh set of modular coils to be optimized
# ---------------------------------------------------------------------------
ncoils = 5  # unique coils per half-period
order = 8
base_curves = create_equally_spaced_curves(
    ncoils, nfp, stellsym=True, R0=5.5, R1=1.3, order=order
)
base_currents = [Current(1.4e6) for _ in range(ncoils)]
# Fix one current to remove the global current-scaling gauge freedom
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))


# ---------------------------------------------------------------------------
# 4. Build the objective and run L-BFGS-B
# ---------------------------------------------------------------------------
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = sum(MeanSquaredCurvature(c) for c in base_curves)
Jcs = sum(LpCurveCurvature(c, p=2, threshold=5.0) for c in base_curves)
Jarcs = sum(ArclengthVariation(c) for c in base_curves)

LENGTH_TARGET = 11.0
LENGTH_WEIGHT = 1e-5
CURV_WEIGHT = 1e-6
MSC_WEIGHT = 1e-7
ARC_WEIGHT = 1e-4

JF = (
    Jf
    + LENGTH_WEIGHT * sum(QuadraticPenalty(Jl, LENGTH_TARGET, "max") for Jl in Jls)
    + CURV_WEIGHT * Jcs
    + MSC_WEIGHT * Jccdist
    + ARC_WEIGHT * Jarcs
)


def fun(dofs):
    JF.x = dofs
    return float(JF.J()), JF.dJ()


print("\nInitial flux objective:  Jf = %.3e" % Jf.J())
print("Optimizing coils ...")
res = minimize(
    fun, JF.x, jac=True, method="L-BFGS-B",
    options={"maxiter": 400, "maxcor": 300, "gtol": 1e-12, "ftol": 1e-12},
)
print(f"  converged in {res.nit} iterations, final J = {res.fun:.3e}")
print(f"  final flux Jf = {Jf.J():.3e}")
print(f"  mean coil length = {np.mean([Jl.J() for Jl in Jls]):.2f} m")

# Surface-averaged normal-field error after optimization
bs.set_points(s.gamma().reshape((-1, 3)))
B = bs.B().reshape((nphi, ntheta, 3))
n = s.unitnormal()
absB = np.linalg.norm(B, axis=2)
Bn = np.sum(B * n, axis=2)
print(f"  <|B.n|/|B|> on plasma surface = {np.mean(np.abs(Bn) / absB):.3e}")


# ---------------------------------------------------------------------------
# 5. Visualization: full-torus plasma surface + optimized coils + W7-X coils
# ---------------------------------------------------------------------------
print("\nRendering matplotlib visualization ...")

# Expand the half-period plasma surface to the full torus for plotting
full = SurfaceRZFourier(
    nfp=nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
    quadpoints_phi=np.linspace(0, 1.0, nfp * nphi * 2, endpoint=False),
    quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
)
full.x = s.x
G = full.gamma()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    G[:, :, 0], G[:, :, 1], G[:, :, 2],
    color="#1f77b4", alpha=0.35, linewidth=0, antialiased=True,
    rstride=1, cstride=1,
)

for c in coils:
    g = c.curve.gamma()
    g = np.vstack([g, g[:1]])  # close the loop
    ax.plot(g[:, 0], g[:, 1], g[:, 2], color="#d62728", linewidth=1.6)

for c in ref_coils:
    g = c.curve.gamma()
    g = np.vstack([g, g[:1]])
    ax.plot(g[:, 0], g[:, 1], g[:, 2], color="0.45", linewidth=0.7, alpha=0.6)

ma_g = magnetic_axis.gamma()
ma_g = np.vstack([ma_g, ma_g[:1]])
ax.plot(ma_g[:, 0], ma_g[:, 1], ma_g[:, 2], "k--", linewidth=1.2, label="magnetic axis")

ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("W7-X plasma boundary + optimized coils (red) vs. reference W7-X coils (grey)")
ax.set_box_aspect((1, 1, 0.35))
lim = 7.0
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-2, 2)
ax.legend(loc="upper right")

out = "w7x_coils_optimized.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"  saved {out}")
plt.show()

"""
Heart-shaped stellarator.

The classical parametric heart curve
    x(t) = 16 sin^3(t)              = 12 sin(t) - 4 sin(3t)
    y(t) = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t
is already a tiny finite Fourier series, so it drops straight into a
SurfaceRZFourier with no fitting.  Rotated 90 deg so the heart's symmetry
axis lies in the equatorial plane (Z = 0), the cross-section becomes
stellarator-symmetric.  A small n=1 perturbation twists the heart as it
goes around — that's what turns a heart-tokamak into a real, non-
axisymmetric, 2-field-period stellarator.

Run:
    python heart_stellarator.py            # write static plots
    python heart_stellarator.py --gif      # also render a rotating GIF
"""

import argparse
import io
import os

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


def build_heart_surface(nfp=2, R0=1.0, A=0.018, twist=0.035,
                        nphi=48, ntheta=48):
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
    s.set_rc(1, 1,  twist)
    s.set_zs(1, 1, -twist)
    return s


def optimize_coils(s, ncoils=5, order=12, R0=1.0, R1=0.55, current=1.0e5):
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    base_currents = [Current(current) for _ in range(ncoils)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
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

    print(f"Initial flux objective Jf = {Jf.J():.3e}")
    print("Running L-BFGS-B ...")
    res = minimize(fun, JF.x, jac=True, method="L-BFGS-B",
                   options={"maxiter": 800, "maxcor": 300,
                            "gtol": 1e-12, "ftol": 1e-14})
    print(f"  converged in {res.nit} iter,  Jf = {Jf.J():.3e}")
    print(f"  mean coil length = {np.mean([Jl.J() for Jl in Jls]):.3f} m")

    bs.set_points(s.gamma().reshape((-1, 3)))
    nphi, ntheta = s.quadpoints_phi.size, s.quadpoints_theta.size
    B = bs.B().reshape((nphi, ntheta, 3))
    n = s.unitnormal()
    rel = np.abs(np.sum(B * n, axis=2)) / np.linalg.norm(B, axis=2)
    print(f"  <|B.n|/|B|> = {rel.mean():.3e},  max = {rel.max():.3e}")
    return coils, rel


def render_full_surface(s, nphi=None, ntheta=128):
    if nphi is None:
        nphi = s.nfp * 96
    full = SurfaceRZFourier(
        nfp=s.nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
        quadpoints_phi=np.linspace(0, 1.0, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
    )
    full.x = s.x
    return full


def cross_section(s, phi_norm, ntheta=400):
    cross = SurfaceRZFourier(
        nfp=s.nfp, stellsym=True, mpol=s.mpol, ntor=s.ntor,
        quadpoints_phi=np.array([phi_norm]),
        quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=True),
    )
    cross.x = s.x
    return cross.gamma()[0]


def plot_static(s, coils, rel, R0=1.0):
    full = render_full_surface(s)
    G = full.gamma()
    coil_paths = [np.vstack([c.curve.gamma(), c.curve.gamma()[:1]])
                  for c in coils]

    # half-torus cut-away so the heart cross-section is visible
    mask = G[:, :, 1] >= -0.02
    Gx = np.where(mask, G[:, :, 0], np.nan)
    Gy = np.where(mask, G[:, :, 1], np.nan)
    Gz = np.where(mask, G[:, :, 2], np.nan)

    fig = plt.figure(figsize=(15, 8))
    L = R0 + 0.45

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(G[:, :, 0], G[:, :, 1], G[:, :, 2],
                     color="#e63946", alpha=0.65, linewidth=0,
                     antialiased=True, shade=True)
    for g in coil_paths:
        ax1.plot(g[:, 0], g[:, 1], g[:, 2], color="#1d3557", linewidth=1.3)
    ax1.view_init(elev=18, azim=35)
    ax1.set_box_aspect((1, 1, 0.45))
    ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-0.45, 0.45)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_zlabel("z [m]")
    ax1.set_title("Full plasma surface + optimized coils")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(Gx, Gy, Gz,
                     color="#e63946", alpha=0.85, linewidth=0,
                     antialiased=True, shade=True)
    for phi_norm in [0.0, 0.5]:
        gx, gy, gz = cross_section(s, phi_norm).T
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
        f"Heart stellarator — nfp={s.nfp},  R0={R0} m,  {len(coils)} optimized"
        f" coils,  $\\langle|B\\cdot n|/|B|\\rangle$ = {rel.mean()*100:.2f}%",
        fontsize=13, y=1.0,
    )
    plt.tight_layout()
    plt.savefig("heart_stellarator.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  saved heart_stellarator.png")

    fig2, ax2d = plt.subplots(figsize=(7, 7))
    phis = np.linspace(0, 0.5 / s.nfp, 5)
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(phis)))
    for phi_norm, col in zip(phis, colors):
        g = cross_section(s, phi_norm)
        R = np.sqrt(g[:, 0]**2 + g[:, 1]**2)
        Z = g[:, 2]
        ax2d.plot(R, Z, color=col, linewidth=2,
                  label=r"$\phi$ = " + f"{phi_norm * 360:.0f}"
                        + r"$^\circ$/nfp")
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("R [m]"); ax2d.set_ylabel("Z [m]")
    ax2d.set_title("Heart cross-sections at several toroidal angles")
    ax2d.grid(alpha=0.3)
    ax2d.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("heart_cross_sections.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  saved heart_cross_sections.png")


def render_gif(s, coils, R0=1.0, n_frames=48, fps=18):
    full = render_full_surface(s)
    G = full.gamma()
    coil_paths = [np.vstack([c.curve.gamma(), c.curve.gamma()[:1]])
                  for c in coils]

    frames = []
    L = R0 + 0.45
    fig = plt.figure(figsize=(7, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    print(f"Rendering {n_frames} frames ...")
    for k in range(n_frames):
        ax.clear()
        ax.plot_surface(G[:, :, 0], G[:, :, 1], G[:, :, 2],
                        color="#e63946", alpha=0.72, linewidth=0,
                        antialiased=True, shade=True)
        for g in coil_paths:
            ax.plot(g[:, 0], g[:, 1], g[:, 2], color="#1d3557",
                    linewidth=1.1)
        azim = 360.0 * k / n_frames
        elev = 18 + 8 * np.sin(2 * np.pi * k / n_frames)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.45))
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-0.45, 0.45)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_title(f"Heart stellarator  (nfp={s.nfp}, R0={R0} m, "
                     f"{len(coils)} coils)")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=85, bbox_inches="tight",
                    facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB").convert(
            "P", palette=Image.ADAPTIVE, colors=128)
        frames.append(img)
        buf.close()
        if (k + 1) % 8 == 0:
            print(f"  {k+1}/{n_frames}")
    plt.close(fig)

    frames[0].save(
        "heart_stellarator.gif", save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True, disposal=2,
    )
    size_kb = os.path.getsize("heart_stellarator.gif") / 1024
    print(f"  saved heart_stellarator.gif "
          f"({n_frames} frames, {fps} fps, {size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true",
                        help="also render a rotating GIF")
    args = parser.parse_args()

    R0 = 1.0
    s = build_heart_surface(R0=R0)
    print(f"Heart stellarator:  nfp={s.nfp}, R0={R0} m\n")
    coils, rel = optimize_coils(s, R0=R0)
    plot_static(s, coils, rel, R0=R0)
    if args.gif:
        render_gif(s, coils, R0=R0)


if __name__ == "__main__":
    main()

# heart-stellarator

A stellarator with the cross section of a heart.

![Heart stellarator — rotating view](heart_stellarator.gif)

The plasma boundary's poloidal cross-section traces the classical
parametric heart curve

```
x(t) = 16 sin³(t)
y(t) = 13 cos t − 5 cos 2t − 2 cos 3t − cos 4t
```

Rotated 90° in the (R, Z) plane so the heart's symmetry axis lies on the
equatorial plane, the curve is *already* a finite Fourier series and drops
straight into a `SurfaceRZFourier` with no fitting — five `R_{m,0}` modes
and two `Z_{m,0}` modes give the shape exactly. A small `n=1` modulation
adds the toroidal twist that turns this from a heart-tokamak into a real
2-field-period stellarator. Five modular coils per half-period are then
optimized against the surface with L-BFGS-B until `⟨|B·n|/|B|⟩ ≈ 1%`.

## Result

| | |
|---|---|
| ![3-D view with cut-away](heart_stellarator.png) | ![Cross-sections](heart_cross_sections.png) |

The cut-away on the upper right shows the heart cross-section at the slice
plane. The cross-sections plot shows how the heart morphs as you move
through one half-field-period — the `n=1` twist rotates and deforms it.

## How it's built

| File | Purpose |
|---|---|
| `heart_stellarator.py` | Designs the heart surface, optimizes coils, plots the static figures |
| `heart_stellarator_gif.py` | Same pipeline + renders 48 rotation frames into a GIF |
| `simple_plasma_coils.py` | Warm-up: 3-period rotating-ellipse plasma + coils |
| `w7x_coil_optimization.py` | Stage-two coil optimization against the real W7-X boundary |
| `Dockerfile` | Debian + gfortran/OpenMPI/netCDF/HDF5/LAPACK, SIMSOPT built from source |
| `docker-compose.yml` | One service per script, working dir bind-mounted so PNGs land back on the host |

`simsopt` is compiled from source (the PyPI aarch64 wheel uses CPU
extensions Colima's VM doesn't expose); `pybind11` is pinned `<3` so the
vendored xtensor compiles.

## Run it

```bash
docker compose build                          # one-time
docker compose run --rm heart-stellarator     # static plots
docker compose run --rm heart-gif             # rotating GIF
```

Outputs land in the working directory:
`heart_stellarator.png`, `heart_cross_sections.png`, `heart_stellarator.gif`.

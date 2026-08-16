# opt-arena

Race classical minimization methods on **the same function** and see who actually reached the well — by path, `f`-count, and `f*`.

Modernized from NTU «KhPI» MIIO labs (dichotomy, cubic interpolation, conjugate gradients, Levenberg–Marquardt). The Python package is the source of truth. The TypeScript page is a live sketch of the same methods.

```bash
pip install -e ".[dev]"
opt-arena run --function rosenbrock
opt-arena run --function quad1d --json
```

| Method | Class |
|---|---|
| dichotomy, golden section, cubic (Hermite/Davidon) | 1D interval |
| Fletcher–Reeves CG, BFGS, Levenberg–Marquardt | nD |

Functions: `(x-2)²`, a steep 1D well, Rosenbrock, a skinny ellipse `100x²+y²`, and a sphere.

```bash
cd web && npm install && npm run dev
```

No SciPy in the core — otherwise this would be a wrapper, not an arena.

Provenance: NTU KhPI coursework, rewritten. MIT.

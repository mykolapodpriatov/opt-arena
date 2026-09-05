# opt-arena

Race classical minimization methods on **the same function** and see who actually reached the well — by path, `f`-count, and `f*`.

Modernized from NTU «KhPI» MIIO labs (dichotomy, cubic interpolation, conjugate gradients, Levenberg–Marquardt). The Python package is the source of truth. The TypeScript page is a live sketch of the same methods.

```bash
pip install -e ".[dev]"
opt-arena run --function rosenbrock
opt-arena run --function quad1d --json
```

The table is ranked and the race has a winner:

```
rank,method,converged,reason,n_f,n_grad,steps,f_final,err_x,x_final
1,levenberg,1,tol_g,27,19,19,1.65733e-17,9.11034e-09,1 1
2,newton_quasi,1,tol_g,90,36,36,7.66456e-25,1.16778e-12,1 1
3,conjugate_grad,0,budget,201,17,17,2.58184e+11,256.45,225.14 -123.611
Winner: levenberg, converged in 27 function evaluation(s) to f=1.65733e-17.
```

Converged methods rank above unconverged ones whatever their final `f`: being stopped by
a budget is not the same as stopping on a tolerance. Among the converged, fewer function
evaluations wins, because iterations are not comparable across method classes (one CG step
costs a whole line search, one dichotomy step costs two evaluations). Ties go to distance
from the known `x*`, then to the method name, so two runs agree. Unconverged methods are
ordered by how low they got.

Every `Budget` field has a flag, defaulting to the dataclass value:

```bash
# Who gets furthest on 200 function evaluations?
opt-arena run --function rosenbrock --max-f 200
opt-arena run --function sphere --tol-g 1e-10 --max-grad 100
```

`--max-f` is the interesting one. It is the fair axis for comparing a derivative-free
method against a gradient one.

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

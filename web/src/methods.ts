import type { Objective } from "./functions";

export type Trace = {
  method: string;
  color: string;
  path: number[][];
  nF: number;
  nG: number;
  fFinal: number;
};

export const COLORS: Record<string, string> = {
  dichotomy: "#c23b22",
  golden: "#1f6f4a",
  cubic: "#b58105",
  conjugate_grad: "#1d4e89",
  newton_quasi: "#6b2d5c",
  levenberg: "#0f7a8a",
};

const PHI = 0.5 * (Math.sqrt(5) - 1);

function add(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + b[i]);
}
function scale(a: number[], s: number): number[] {
  return a.map((v) => v * s);
}
function dot(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

type Counted = { nF: number; nG: number; f: (x: number[]) => number; g: (x: number[]) => number[] };

function wrap(obj: Objective): Counted {
  const c: Counted = {
    nF: 0,
    nG: 0,
    f: (x) => {
      c.nF += 1;
      return obj.f(x);
    },
    g: (x) => {
      c.nG += 1;
      return obj.g(x);
    },
  };
  return c;
}

function dichotomy(obj: Objective): Trace {
  const c = wrap(obj);
  const [a0, b0] = obj.domain[0];
  let a = a0;
  let b = b0;
  const path: number[][] = [];
  for (let i = 0; i < 80 && b - a > 1e-6; i++) {
    const probe = Math.max((b - a) * 0.25, 1e-7);
    const x1 = 0.5 * (a + b) - probe;
    const x2 = 0.5 * (a + b) + probe;
    if (c.f([x1]) < c.f([x2])) {
      b = x2;
      path.push([x1]);
    } else {
      a = x1;
      path.push([x2]);
    }
  }
  const x = 0.5 * (a + b);
  path.push([x]);
  return pack("dichotomy", c, path, obj.f([x]));
}

function golden(obj: Objective): Trace {
  const c = wrap(obj);
  let [a, b] = obj.domain[0];
  let p = b - PHI * (b - a);
  let q = a + PHI * (b - a);
  let fp = c.f([p]);
  let fq = c.f([q]);
  const path: number[][] = [[p]];
  for (let i = 0; i < 80 && b - a > 1e-6; i++) {
    if (fp < fq) {
      b = q;
      q = p;
      fq = fp;
      p = b - PHI * (b - a);
      fp = c.f([p]);
      path.push([p]);
    } else {
      a = p;
      p = q;
      fp = fq;
      q = a + PHI * (b - a);
      fq = c.f([q]);
      path.push([q]);
    }
  }
  const x = fp < fq ? p : q;
  return pack("golden", c, path, obj.f([x]));
}

function cubic(obj: Objective): Trace {
  const c = wrap(obj);
  let [a, b] = obj.domain[0];
  let fa = c.f([a]);
  let fb = c.f([b]);
  let ga = c.g([a])[0];
  let gb = c.g([b])[0];
  const path: number[][] = [[a]];
  for (let i = 0; i < 40 && Math.abs(b - a) > 1e-6; i++) {
    const z = (3 * (fa - fb)) / (b - a) + ga + gb;
    const disc = z * z - ga * gb;
    const w = disc > 0 ? Math.sqrt(disc) : 0;
    const denom = gb - ga + 2 * w;
    let xmin = Math.abs(denom) < 1e-12 ? 0.5 * (a + b) : b - ((b - a) * (gb + w - z)) / denom;
    xmin = Math.min(Math.max(xmin, a + 1e-9), b - 1e-9);
    const fx = c.f([xmin]);
    const gx = c.g([xmin])[0];
    path.push([xmin]);
    if (gx > 0) {
      b = xmin;
      fb = fx;
      gb = gx;
    } else {
      a = xmin;
      fa = fx;
      ga = gx;
    }
  }
  const last = path[path.length - 1];
  return pack("cubic", c, path, obj.f(last));
}

function backtrack(c: Counted, x: number[], f0: number, g: number[], d: number[]): number {
  const slope = dot(g, d);
  if (slope >= 0) return 0;
  let alpha = 1;
  for (let i = 0; i < 24; i++) {
    const trial = add(x, scale(d, alpha));
    if (c.f(trial) <= f0 + 1e-4 * alpha * slope) return alpha;
    alpha *= 0.5;
  }
  return alpha;
}

function cg(obj: Objective): Trace {
  const c = wrap(obj);
  let x = obj.x0.slice();
  let f0 = c.f(x);
  let g = c.g(x);
  let d = scale(g, -1);
  const path = [x.slice()];
  for (let i = 0; i < 80 && norm(g) > 1e-6; i++) {
    const alpha = backtrack(c, x, f0, g, d);
    if (alpha <= 0) break;
    const nxt = add(x, scale(d, alpha));
    const gn = c.g(nxt);
    f0 = c.f(nxt);
    path.push(nxt);
    const beta = dot(g, g) < 1e-18 ? 0 : dot(gn, gn) / dot(g, g);
    d = add(scale(gn, -1), scale(d, beta));
    if (dot(d, gn) >= 0) d = scale(gn, -1);
    x = nxt;
    g = gn;
  }
  return pack("conjugate_grad", c, path, obj.f(x));
}

function bfgs(obj: Objective): Trace {
  const c = wrap(obj);
  let x = obj.x0.slice();
  const n = x.length;
  let H = ident(n);
  let g = c.g(x);
  let f0 = c.f(x);
  const path = [x.slice()];
  for (let k = 0; k < 80 && norm(g) > 1e-6; k++) {
    let d = scale(matvec(H, g), -1);
    if (dot(d, g) >= 0) {
      d = scale(g, -1);
      H = ident(n);
    }
    const alpha = backtrack(c, x, f0, g, d);
    if (alpha <= 0) break;
    const s = scale(d, alpha);
    const nxt = add(x, s);
    const gn = c.g(nxt);
    f0 = c.f(nxt);
    path.push(nxt);
    const y = gn.map((v, i) => v - g[i]);
    H = bfgsUpdate(H, s, y);
    x = nxt;
    g = gn;
  }
  return pack("newton_quasi", c, path, obj.f(x));
}

function lm(obj: Objective): Trace {
  const c = wrap(obj);
  if (!obj.r || !obj.j) {
    return { method: "levenberg", color: COLORS.levenberg, path: [], nF: 0, nG: 0, fFinal: NaN };
  }
  let x = obj.x0.slice();
  let r = obj.r(x);
  c.nF += 1;
  let fval = dot(r, r);
  let mu = 1e-3;
  const path = [x.slice()];
  for (let k = 0; k < 80; k++) {
    const J = obj.j(x);
    c.nG += 1;
    const jt = transpose(J);
    const jtj = matmul(jt, J);
    const g = matvec(jt, r);
    if (norm(g) < 1e-6) break;
    let ok = false;
    for (let inner = 0; inner < 6; inner++) {
      const a = jtj.map((row, i) => row.map((v, j) => (i === j ? v + mu : v)));
      const step = scale(solve(a, g), -1);
      const trial = add(x, step);
      const rt = obj.r(trial);
      c.nF += 1;
      const ft = dot(rt, rt);
      if (ft < fval) {
        x = trial;
        r = rt;
        fval = ft;
        path.push(x.slice());
        mu = Math.max(mu / 3, 1e-12);
        ok = true;
        break;
      }
      mu *= 10;
    }
    if (!ok) break;
  }
  return pack("levenberg", c, path, obj.f(x));
}

function pack(method: string, c: Counted, path: number[][], fFinal: number): Trace {
  return { method, color: COLORS[method], path, nF: c.nF, nG: c.nG, fFinal };
}

function ident(n: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}
function matvec(m: number[][], v: number[]): number[] {
  return m.map((row) => dot(row, v));
}
function transpose(m: number[][]): number[][] {
  return m[0].map((_, j) => m.map((row) => row[j]));
}
function matmul(a: number[][], b: number[][]): number[][] {
  const bt = transpose(b);
  return a.map((row) => bt.map((col) => dot(row, col)));
}
function bfgsUpdate(H: number[][], s: number[], y: number[]): number[][] {
  const ys = dot(y, s);
  if (Math.abs(ys) < 1e-16) return H;
  const rho = 1 / ys;
  const n = s.length;
  const A = ident(n).map((row, i) => row.map((v, j) => v - rho * s[i] * y[j]));
  const B = ident(n).map((row, i) => row.map((v, j) => v - rho * y[i] * s[j]));
  const tmp = matmul(A, H);
  const out = matmul(tmp, B);
  return out.map((row, i) => row.map((v, j) => v + rho * s[i] * s[j]));
}
function solve(aIn: number[][], b: number[]): number[] {
  const n = b.length;
  const a = aIn.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    [a[col], a[piv]] = [a[piv], a[col]];
    const diag = a[col][col];
    if (Math.abs(diag) < 1e-18) return scale(b, 0);
    for (let r = col + 1; r < n; r++) {
      const f = a[r][col] / diag;
      for (let c = col; c <= n; c++) a[r][c] -= f * a[col][c];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let acc = a[i][n];
    for (let j = i + 1; j < n; j++) acc -= a[i][j] * x[j];
    x[i] = acc / a[i][i];
  }
  return x;
}

export function runSelected(obj: Objective, names: string[]): Trace[] {
  const runners: Record<string, (o: Objective) => Trace> = {
    dichotomy,
    golden,
    cubic,
    conjugate_grad: cg,
    newton_quasi: bfgs,
    levenberg: lm,
  };
  return names.filter((n) => runners[n]).map((n) => runners[n](obj));
}

export function defaultMethods(dim: 1 | 2): string[] {
  return dim === 1
    ? ["dichotomy", "golden", "cubic"]
    : ["conjugate_grad", "newton_quasi", "levenberg"];
}

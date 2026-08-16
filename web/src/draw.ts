import type { Objective } from "./functions";
import type { Trace } from "./methods";

export function draw(cv: HTMLCanvasElement, obj: Objective, traces: Trace[]): void {
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  const w = cv.width;
  const h = cv.height;
  ctx.clearRect(0, 0, w, h);
  paper(ctx, w, h);
  if (obj.dim === 1) draw1d(ctx, w, h, obj, traces);
  else draw2d(ctx, w, h, obj, traces);
}

function paper(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.fillStyle = "#f3ead3";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(40, 55, 40, 0.08)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 24) {
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 24) {
    ctx.beginPath();
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
    ctx.stroke();
  }
}

function draw1d(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  obj: Objective,
  traces: Trace[],
): void {
  const [lo, hi] = obj.domain[0];
  const pad = 48;
  const xs = Array.from({ length: 240 }, (_, i) => lo + ((hi - lo) * i) / 239);
  const ys = xs.map((x) => obj.f([x]));
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const X = (x: number) => pad + ((x - lo) / (hi - lo)) * (w - 2 * pad);
  const Y = (y: number) => h - pad - ((y - ymin) / (ymax - ymin || 1)) * (h - 2 * pad);

  ctx.strokeStyle = "#2b2416";
  ctx.lineWidth = 2;
  ctx.beginPath();
  xs.forEach((x, i) => (i === 0 ? ctx.moveTo(X(x), Y(ys[i])) : ctx.lineTo(X(x), Y(ys[i]))));
  ctx.stroke();

  ctx.fillStyle = "#2b2416";
  ctx.beginPath();
  ctx.arc(X(obj.star[0]), Y(obj.f(obj.star)), 5, 0, Math.PI * 2);
  ctx.fill();

  for (const t of traces) {
    ctx.strokeStyle = t.color;
    ctx.fillStyle = t.color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    t.path.forEach((p, i) => {
      const px = X(p[0]);
      const py = Y(obj.f(p));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    t.path.forEach((p, i) => {
      ctx.globalAlpha = 0.25 + (0.75 * i) / Math.max(1, t.path.length - 1);
      ctx.beginPath();
      ctx.arc(X(p[0]), Y(obj.f(p)), i === t.path.length - 1 ? 5 : 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }
}

function draw2d(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  obj: Objective,
  traces: Trace[],
): void {
  const [xlo, xhi] = obj.domain[0];
  const [ylo, yhi] = obj.domain[1] ?? obj.domain[0];
  const pad = 16;
  const X = (x: number) => pad + ((x - xlo) / (xhi - xlo)) * (w - 2 * pad);
  const Y = (y: number) => h - pad - ((y - ylo) / (yhi - ylo)) * (h - 2 * pad);

  const nx = 90;
  const ny = 64;
  const vals: number[] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = xlo + ((xhi - xlo) * i) / (nx - 1);
      const y = ylo + ((yhi - ylo) * j) / (ny - 1);
      vals.push(Math.log10(1 + obj.f([x, y])));
    }
  }
  const vmin = Math.min(...vals);
  const vmax = Math.max(...vals);
  const cw = (w - 2 * pad) / nx;
  const ch = (h - 2 * pad) / ny;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const t = (vals[j * nx + i] - vmin) / (vmax - vmin || 1);
      ctx.fillStyle = `rgba(47, 72, 52, ${0.04 + t * 0.34})`;
      ctx.fillRect(X(xlo) + i * cw, Y(yhi) + j * ch, cw + 0.5, ch + 0.5);
    }
  }

  ctx.fillStyle = "#2b2416";
  ctx.beginPath();
  ctx.arc(X(obj.star[0]), Y(obj.star[1]), 6, 0, Math.PI * 2);
  ctx.fill();

  for (const tr of traces) {
    ctx.strokeStyle = tr.color;
    ctx.fillStyle = tr.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    tr.path.forEach((p, i) => (i === 0 ? ctx.moveTo(X(p[0]), Y(p[1])) : ctx.lineTo(X(p[0]), Y(p[1]))));
    ctx.stroke();
    const last = tr.path[tr.path.length - 1];
    if (last) {
      ctx.beginPath();
      ctx.arc(X(last[0]), Y(last[1]), 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

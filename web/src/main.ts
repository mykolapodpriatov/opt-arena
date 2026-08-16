import { draw } from "./draw";
import { OBJECTIVES, byName } from "./functions";
import { COLORS, defaultMethods, runSelected } from "./methods";

const fnSel = document.querySelector<HTMLSelectElement>("#fn")!;
const methodsBox = document.querySelector<HTMLDivElement>("#methods")!;
const board = document.querySelector<HTMLTableSectionElement>("#board tbody")!;
const cv = document.querySelector<HTMLCanvasElement>("#cv")!;
const hint = document.querySelector<HTMLParagraphElement>("#hint")!;

function currentFn() {
  return byName(fnSel.value);
}

function fillFunctions(): void {
  fnSel.innerHTML = OBJECTIVES.map(
    (o) => `<option value="${o.name}">${o.label}</option>`,
  ).join("");
  fnSel.value = "rosenbrock";
}

function fillMethods(): void {
  const obj = currentFn();
  const on = new Set(defaultMethods(obj.dim));
  const names =
    obj.dim === 1
      ? ["dichotomy", "golden", "cubic"]
      : ["conjugate_grad", "newton_quasi", "levenberg"];
  methodsBox.innerHTML = names
    .map(
      (n) =>
        `<label class="chk"><input type="checkbox" value="${n}" ${on.has(n) ? "checked" : ""}/>
           <i style="background:${COLORS[n]}"></i>${n.replaceAll("_", " ")}</label>`,
    )
    .join("");
}

function selected(): string[] {
  return [...methodsBox.querySelectorAll<HTMLInputElement>("input:checked")].map((el) => el.value);
}

function race(): void {
  const obj = currentFn();
  const traces = runSelected(obj, selected());
  draw(cv, obj, traces);
  const best = Math.min(
    ...traces.map((x) => (Number.isFinite(x.fFinal) ? x.fFinal : Infinity)),
  );
  board.innerHTML = traces
    .map((t) => {
      const win = Number.isFinite(t.fFinal) && t.fFinal === best;
      return `<tr>
        <td><i style="background:${t.color}"></i></td>
        <td>${win ? "★ " : ""}${t.method.replaceAll("_", " ")}</td>
        <td>${t.nF}</td>
        <td>${t.nG}</td>
        <td>${Number.isFinite(t.fFinal) ? t.fFinal.toExponential(2) : "—"}</td>
      </tr>`;
    })
    .join("");
  const winner = traces.find((t) => t.fFinal === best);
  hint.textContent = winner
    ? `Lowest f* this run: ${winner.method.replaceAll("_", " ")} = ${winner.fFinal.toExponential(3)} after ${winner.nF} f-calls.`
    : "";
}

function reset(): void {
  board.innerHTML = "";
  hint.textContent = currentFn().label;
  draw(cv, currentFn(), []);
}

fnSel.addEventListener("change", () => {
  fillMethods();
  reset();
});
document.querySelector("#run")!.addEventListener("click", race);
document.querySelector("#reset")!.addEventListener("click", reset);

fillFunctions();
fillMethods();
reset();

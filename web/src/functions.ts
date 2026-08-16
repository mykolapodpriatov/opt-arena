export type Objective = {
  name: string;
  dim: 1 | 2;
  label: string;
  f: (x: number[]) => number;
  g: (x: number[]) => number[];
  r?: (x: number[]) => number[];
  j?: (x: number[]) => number[][];
  domain: [[number, number], [number, number]?];
  x0: number[];
  star: number[];
};

export const OBJECTIVES: Objective[] = [
  {
    name: "quad1d",
    dim: 1,
    label: "1D  (x−2)²",
    f: (x) => (x[0] - 2) ** 2,
    g: (x) => [2 * (x[0] - 2)],
    domain: [[-2, 6]],
    x0: [5],
    star: [2],
  },
  {
    name: "valley1d",
    dim: 1,
    label: "1D  100(x−1)²",
    f: (x) => 100 * (x[0] - 1) ** 2,
    g: (x) => [200 * (x[0] - 1)],
    domain: [[-2, 4]],
    x0: [3.5],
    star: [1],
  },
  {
    name: "rosenbrock",
    dim: 2,
    label: "Rosenbrock banana",
    f: (x) => (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
    g: (x) => [
      -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
      200 * (x[1] - x[0] ** 2),
    ],
    r: (x) => [1 - x[0], 10 * (x[1] - x[0] ** 2)],
    j: (x) => [
      [-1, 0],
      [-20 * x[0], 10],
    ],
    domain: [
      [-2, 2],
      [-1, 3],
    ],
    x0: [-1.2, 1],
    star: [1, 1],
  },
  {
    name: "stretched",
    dim: 2,
    label: "Stretched  100x² + y²",
    f: (x) => 100 * x[0] ** 2 + x[1] ** 2,
    g: (x) => [200 * x[0], 2 * x[1]],
    r: (x) => [10 * x[0], x[1]],
    j: (x) => [
      [10, 0],
      [0, 1],
    ],
    domain: [
      [-2, 2],
      [-2, 2],
    ],
    x0: [1.5, 1.5],
    star: [0, 0],
  },
  {
    name: "sphere",
    dim: 2,
    label: "Sphere  x² + y²",
    f: (x) => x[0] ** 2 + x[1] ** 2,
    g: (x) => [2 * x[0], 2 * x[1]],
    r: (x) => [x[0], x[1]],
    j: (x) => [
      [1, 0],
      [0, 1],
    ],
    domain: [
      [-2, 2],
      [-2, 2],
    ],
    x0: [1.5, -1.2],
    star: [0, 0],
  },
];

export function byName(name: string): Objective {
  const found = OBJECTIVES.find((o) => o.name === name);
  if (!found) throw new Error(name);
  return found;
}

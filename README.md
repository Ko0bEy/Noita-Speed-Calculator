# Noita Speed Calculator

CLI tool to find builds for arbitrary teleports in **Noita**.

---

## Angle Solver: Arbitrary teleport

```bash
angle_solver.exe x0 y0 x1 y1 [options]
```

Also runs speed\_calc for the distance unless `--skip-speed-calc` is set.

| option               | default | meaning                                                |
| -------------------- | ------- | ------------------------------------------------------ |
| `-a, --shot-angle °` | `90`    | base facing direction (0° = +X, clockwise)             |
| `-n, --max-n`        | `100`   | upper limit for projectile count that will be searched |
| `-t, --tolerance`    | `0.01`  | angular tolerance (radians)                            |
| `--top`              | `3`     | how many rows to print per pattern degree              |
| `--skip-speed-calc`  |         | do not run speed\_calc for the distance                |
| `--visualize`        |         | show projectile pattern plot (with matplotlib)         |

### Example

```bash
angle_solver.exe 0 0 6400 15000 -t 0.02 --visualize --skip-speed-calc
```

Prints the most accurate + fewest-projectile formations to hit `(6400, 15000)` from the origin.
Also finds suitable speed modifier solutions, unless `--skip-speed-calc` is used.

## Speed solver - PW Travel

```bash
speed_calc.exe DISTANCE [options]
```

| option           | default  | meaning                                                |
| ---------------- | -------- | ------------------------------------------------------ |
| `-c, --coefs`    | see code | speed multipliers per modifier / perk                  |
| `-t, --tol`      | `5e-3`   | relative error tolerance                               |
| `-n, --top-n`    | `50`     | number of solutions to print                           |
| `-u, --uncapped` |          | indices whose multipliers are **uncapped** (besides 0) |

### Examples

```bash
# Closest match for 12 000 px flight-path
speed_calc.exe 12000

# Same, but allow multiplier index 2 to exceed the 20× cap
speed_calc.exe 12000 -u 2
```

Both commands print the top solutions in `nz_other / sum_other / x0 / rel_err` order.

---

\*All CLI flags have `--help`.

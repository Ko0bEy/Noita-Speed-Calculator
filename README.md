# Noita Speed Calculator

CLI tool to find builds for arbitrary teleports in **Noita**.

---

## Angle Solver: Arbitrary teleport

```
angle_solver.exe x0 y0 x1 y1 [options]
```

Also runs speed\_calc for the distance unless `--skip-speed-calc` is set.

| Option/Flag               | Default             | Meaning                                                                            |
|---------------------------| ------------------- |------------------------------------------------------------------------------------|
| `x0`                      | (required)          | Shooter X coordinate (float)                                                       |
| `y0`                      | (required)          | Shooter Y coordinate (float)                                                       |
| `x1`                      | (required)          | Target X coordinate (float)                                                        |
| `y1`                      | (required)          | Target Y coordinate (float)                                                        |
| `-a`, `--shot-angle`      | `90.0`              | Shot centreline angle in degrees (CW from +X axis)                                 |
| `-n`, `--max-n`           | `100`               | Max projectiles to test                                                            |
| `-t`, `--tolerance`       | `0.01`              | Max allowed perpendicular error as a fraction of the distance (e.g., 0.01 = 1%)    |
| `-p`, `--pattern-options` | `5 20 30 45 90 180` | Comma/space-separated list of pattern degrees to test (integers between 1 and 180) |
| `--show-few`              | `3`                 | How many small solutions to show per pattern degree                                |
| `--show-accurate`         | `3`                 | How many accurate solutions to show per pattern degree                             |
| `-v`, `--visualize`       |                     | Show projectile pattern plot with matplotlib                                       |
| `--skip-speed-calc`       |                     | Do not run speed_calc for the distance                                             |

**Options passed on to speed_calc:**

| Option/Flag        | Default                          | Meaning                                                     |
|--------------------|----------------------------------|-------------------------------------------------------------|
| `-c`, `--coefs`    | `[1.2 ... 7.5]` (see speed_calc) | Override speed multipliers for speed_calc                   |
| `-u`, `--uncapped` |                                  | Indices whose multipliers are **uncapped** (besides index 0)    |
| `--sort`           | `nz,sum,rel_err,max_exp`         | Sort priorities for speed_calc solutions (comma-separated). |
| `--top-n`          | `25`                             | Number of solutions to print                                                            |


### Example

```
angle_solver.exe 0 0 6400 15000 -t 0.02 --visualize --skip-speed-calc
```

Prints the most accurate + fewest-projectile formations to hit `(6400, 15000)` from the origin.
Also finds suitable speed modifier solutions, unless `--skip-speed-calc` is used.

# speed\_calc.exe Usage

```
speed_calc.exe DISTANCE [options]
```

| Option / Flag      | Default                                             | Meaning                                                                                 |
|--------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------|
| `-c`, `--coefs`    | `[1.2, 0.3, 0.32, 0.33, 0.75, 1.68, 2.0, 2.5, 7.5]` | Speed multipliers per modifier/perk                                                     |
| `-t`, `--tol`      | `5e-3`                                              | Relative error tolerance                                                                |
| `--top-n`          | `50`                                                | Number of solutions to print                                                            |
| `-u`, `--uncapped` | *(none)*                                            | Indices whose multipliers are **uncapped** (besides index 0)                            |
| `--sort`           | `nz,sum,rel_err,max_exp`                            | Solution sort priority (comma-separated). Supported: `nz`, `sum`, `rel_err`, `max_exp`. |

**Note:**

* The `--sort` option allows partial override. If you specify e.g. `--sort max_exp`, the remaining default sort order (`nz`, `sum`, `rel_err`) is appended automatically.
* Sorting priorities:
    * `nz`: amount of non-zero entries
    * `sum`: sum of non-`x[0]` entries
    * `rel_err`: relative error
    * `max_exp`: maximum exponent (probably `x[0]`'s exponent)

### Examples

```
# Closest match for 1 PW (35840px)
speed_calc.exe 35840

# Same, but only use heavy shot and speed up. note that the first coef is always considered uncapped.
speed_calc.exe 35840 -c 1.2 0.3 2.5
```

---

\_All CLI flags have \`--help

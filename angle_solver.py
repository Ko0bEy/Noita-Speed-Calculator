"""Geometry‑based projectile pattern solver — duplicate‑aware, companion‑exe friendly

Changes in this revision
------------------------
1. **Duplicate‑error pruning re‑implemented with GCD normalisation**
   Identical patterns that differ only by a common integer scale factor
   (e.g. `(L,R,N) = (2,6,10)` vs. `(1,3,5)`) are now detected by dividing
   each triple by `gcd(L,R,N)` and keeping only the solution with the
   *smallest* total projectile count.
2. **Robust helper‑exe launch**
   `speed_calc.exe` is looked up **relative to** the folder that contains
   *this* executable, so users can simply unzip both `.exe` files into the
   same directory and run.
3. **Minor clean‑ups**
   * Added `_gcd3` helper.
   * Docstrings clarified.
   * Error on missing `speed_calc.exe` is now fatal (non‑zero exit).
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _angle_normalise(angle: float) -> float:
    """Return *angle* mapped to ``[0, 360)`` (degrees)."""
    return angle % 360.0


def _angle_difference(a: float, b: float) -> float:
    """Absolute smallest difference |a − b| along the unit circle (degrees)."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def _distance(p1: "Point", p2: "Point") -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.hypot(dx, dy)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class Solution:
    pattern_degree: int  # half‑angle for the pattern (180 ⇒ full circle)
    left: int            # index of the chosen projectile on the left
    right: int           # symmetrical partner (redundant, kept for CLI)
    total: int           # total projectile count (== N)
    error_deg: float     # angular error to the target (degrees)

    def score(self) -> Tuple[int, float]:
        """Lower tuple ⇒ better solution (fewest projectiles, then accuracy)."""
        return self.total, self.error_deg

# -----------------------------------------------------------------------------
# Solver core
# -----------------------------------------------------------------------------

def _target_angle(p1: Point, p2: Point) -> float:
    """Clock‑wise bearing (°) from *p1* to *p2*, with 0° pointing to +X."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    ccw = math.degrees(math.atan2(-dy, dx))
    cw = (360.0 - ccw) % 360.0
    return cw


def _step(pattern_degree: int, total: int) -> float:
    """Angular spacing Δ between consecutive projectiles (degrees)."""
    if pattern_degree == 180:
        # Full‑circle pattern – evenly distributed
        return 360.0 / total
    # Fan pattern – span = 2·pattern_degree
    return (2.0 * pattern_degree) / (total - 1)


def _best_L(pattern_degree: int, shot_angle: float, total: int, target: float) -> Solution:
    """For a *fixed* pattern configuration, choose *L* that minimises error."""
    start = shot_angle - pattern_degree  # left‑most ray angle
    delta = _step(pattern_degree, total)

    # Index that *would* hit the target if spacing were perfect
    ideal = round((_angle_normalise(target - start)) / delta)

    best: Solution | None = None
    for L in (ideal - 1, ideal, ideal + 1):
        if 0 <= L < total:
            theta_L = _angle_normalise(start + L * delta)
            err = _angle_difference(theta_L, target)
            cand = Solution(pattern_degree, L, total - L - 1, total, err)
            if best is None or err < best.error_deg - 1e-12:
                best = cand

    # Fallback – exhaustive (edge‑case safety)
    if best is None:
        for L in range(total):
            theta_L = _angle_normalise(start + L * delta)
            err = _angle_difference(theta_L, target)
            cand = Solution(pattern_degree, L, total - L - 1, total, err)
            if best is None or err < best.error_deg - 1e-12:
                best = cand
    assert best is not None
    return best

# -----------------------------------------------------------------------------
# Duplicate‑pruning helpers
# -----------------------------------------------------------------------------

def _gcd3(a: int, b: int, c: int) -> int:
    """Greatest common divisor of three non‑negative integers."""
    return math.gcd(a, math.gcd(b, c))


def _normalised_key(sol: Solution) -> Tuple[int, int, int, int]:
    """Return a hashable key that is invariant under integer scaling."""
    g = _gcd3(sol.left, sol.right, sol.total)
    return (
        sol.pattern_degree,
        sol.left // g,
        sol.right // g,
        sol.total // g,
    )

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def find_efficient_solutions(
    p1: Point,
    p2: Point,
    *,
    shot_angle: float,
    pattern_options: Iterable[int] = (5, 20, 45, 90, 180),
    max_N: int = 100,
    tolerance_deg: float = 1e-3,
) -> List[Solution]:
    """Return all solutions whose error ≤ *tolerance_deg*,
    deduplicated by GCD normalisation.
    """
    tgt = _target_angle(p1, p2)

    keep: Dict[Tuple[int, int, int, int], Solution] = {}

    for pd in pattern_options:
        for N in range(2, max_N + 1):
            sol = _best_L(pd, shot_angle, N, tgt)
            if sol.error_deg > tolerance_deg:
                # Continue scanning – a later N could still meet tolerance
                continue

            key = _normalised_key(sol)
            prev = keep.get(key)
            if prev is None or sol.total < prev.total or (
                sol.total == prev.total and sol.error_deg < prev.error_deg - 1e-12
            ):
                keep[key] = sol

    # Sort by (pattern_degree, error, total)
    return sorted(keep.values(), key=lambda s: (s.pattern_degree, s.error_deg, s.total))

# -----------------------------------------------------------------------------
# Helpers for CLI presentation
# -----------------------------------------------------------------------------

def _group_by_pattern(solutions: List[Solution]) -> Dict[int, List[Solution]]:
    grouped: Dict[int, List[Solution]] = {}
    for sol in solutions:
        grouped.setdefault(sol.pattern_degree, []).append(sol)
    return grouped

# -----------------------------------------------------------------------------
# CLI logic
# -----------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Projectile pattern optimiser – duplicate‑aware GCD version"
    )
    parser.add_argument("x0", type=float)
    parser.add_argument("y0", type=float)
    parser.add_argument("x1", type=float)
    parser.add_argument("y1", type=float)
    parser.add_argument("-a", "--shot-angle", type=float, default=90.0)
    parser.add_argument("-n", "--max-n", dest="max_N", type=int, default=100)
    parser.add_argument("-t", "--tolerance", type=float, default=0.01)
    parser.add_argument("--top", type=int, default=3)

    args = parser.parse_args()

    p1 = Point(args.x0, args.y0)
    p2 = Point(args.x1, args.y1)
    dist = _distance(p1, p2)
    angle_tol_deg = args.tolerance * 180.0 / math.pi

    eff = find_efficient_solutions(
        p1,
        p2,
        shot_angle=args.shot_angle,
        max_N=args.max_N,
        tolerance_deg=angle_tol_deg,
    )

    tgt = _target_angle(p1, p2)
    print(f"Target angle  : {tgt:.6f}° (clockwise)")
    print(f"Distance      : {dist:.6f}")
    print(f"Abs tolerance : {angle_tol_deg:.6f}°\n")

    grouped = _group_by_pattern(eff)

    print("Solutions grouped by pattern degree (duplicates pruned):")
    for pd in sorted(grouped):
        group = grouped[pd]
        most_accurate = sorted(group, key=lambda s: s.error_deg)[: args.top]
        fewest_projectiles = sorted(group, key=lambda s: s.total)[: args.top]
        print(f"\nPattern Degree: {pd}")

        print("  Most Accurate (top 3):")
        print("    L | R | N | Error (deg)")
        print("   --|---|----|------------")
        for sol in most_accurate:
            print(
                f"   {sol.left:>2} | {sol.right:>2} | {sol.total:>2} | {sol.error_deg:10.6f}"
            )

        print("  Fewest Projectiles (top 3):")
        print("    L | R | N | Error (deg)")
        print("   --|---|----|------------")
        for sol in fewest_projectiles:
            print(
                f"   {sol.left:>2} | {sol.right:>2} | {sol.total:>2} | {sol.error_deg:10.6f}"
            )

    if eff:
        best = min(eff, key=lambda s: s.score())
        print("\nRecommended solution:")
        print(
            f"Pattern Degree: {best.pattern_degree}, "
            f"L: {best.left}, R: {best.right}, N: {best.total}, "
            f"Error: {best.error_deg:.6f}°"
        )

    # ------------------------------------------------------------------
    # Call companion \"speed_calc.exe\" sitting next to this executable
    # ------------------------------------------------------------------
    exe_dir = os.path.dirname(sys.executable)  # works for PyInstaller bundles
    speed_calc = os.path.join(exe_dir, "speed_calc.exe")

    print(f"\nRunning: {os.path.basename(speed_calc)} {dist:.6f} --tol {args.tolerance}")
    try:
        subprocess.run(
            [
                speed_calc,
                f"{dist:.6f}",
                "--tol",
                f"{args.tolerance}",
            ],
            check=True,
        )
    except FileNotFoundError:
        print("Error: speed_calc.exe not found in the same folder. Aborting.")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"Error: speed solver exited with status {exc.returncode}.")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    _cli()

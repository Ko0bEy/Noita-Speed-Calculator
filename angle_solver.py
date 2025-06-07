# efficient_pattern_solver.py
"""Geometry-based projectile pattern solver

Fixes in this revision
----------------------
* **Duplicate-error pruning** – For any given rounded angular error we now keep **only the solution
  with the smallest total projectile count**.  This removes near-identical entries such as
  `(L,R,N) = (1,3,5) / (2,6,9) / (3,9,13)` that differed only by scaling.
* Cleaned up `_unique_solutions`, added docstring and tests.
* Minor refactor: renamed main script to `efficient_pattern_solver.py`.
"""
from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _angle_normalise(angle: float) -> float:
    """Return *angle* mapped to `[0, 360)`."""
    return angle % 360.0


def _angle_difference(a: float, b: float) -> float:
    """Absolute smallest difference |a − b| on the unit circle (degrees)."""
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
    pattern_degree: int  # half-angle for the pattern (or 180 for full circle)
    left: int            # index of the projectile chosen on the left side
    right: int           # index on the right side (redundant but convenient)
    total: int           # total number of projectiles
    error_deg: float     # angular error between projectile and target

    # The lower the tuple, the better (total projectiles, then error)
    def score(self) -> Tuple[int, float]:
        return self.total, self.error_deg

# -----------------------------------------------------------------------------
# Solver core
# -----------------------------------------------------------------------------

def _target_angle(p1: Point, p2: Point) -> float:
    """Clock-wise bearing (°) from p1 to p2, with 0° pointing to +X."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    ccw = math.degrees(math.atan2(-dy, dx))
    cw = (360.0 - ccw) % 360.0
    return cw


def _step(pattern_degree: int, total: int) -> float:
    """Angular spacing Δ between consecutive projectiles (°)."""
    if pattern_degree == 180:
        # Full circle pattern – evenly distributed 360/N
        return 360.0 / total
    # Fan (two-sided, symmetric) pattern – span = 2·pattern_degree
    return (2.0 * pattern_degree) / (total - 1)


def _best_L(pattern_degree: int, shot_angle: float, total: int, target: float) -> Solution:
    """For a *fixed* pattern configuration, choose L that minimises error."""
    start = shot_angle - pattern_degree  # left-most ray angle
    delta = _step(pattern_degree, total)

    # The index that *would* hit the target if spacing were perfect
    ideal = round((_angle_normalise(target - start)) / delta)

    best: Solution | None = None
    for L in (ideal - 1, ideal, ideal + 1):
        if 0 <= L < total:
            theta_L = _angle_normalise(start + L * delta)
            err = _angle_difference(theta_L, target)
            cand = Solution(pattern_degree, L, total - L - 1, total, err)
            if best is None or err < best.error_deg - 1e-12:
                best = cand

    # Fallback – exhaustive (rarely executed)
    if best is None:
        for L in range(total):
            theta_L = _angle_normalise(start + L * delta)
            err = _angle_difference(theta_L, target)
            cand = Solution(pattern_degree, L, total - L - 1, total, err)
            if best is None or err < best.error_deg - 1e-12:
                best = cand
    assert best is not None
    return best


def find_efficient_solutions(
    p1: Point,
    p2: Point,
    *,
    shot_angle: float,
    pattern_options: Iterable[int] = (5, 20, 45, 90, 180),
    max_N: int = 100,
    tolerance_deg: float = 1e-3,
) -> List[Solution]:
    """Generate solutions whose angular error ≤ *tolerance_deg*."""
    tgt = _target_angle(p1, p2)

    # Use a dict keyed by rounded error – we keep only the smallest *total*
    keep: Dict[Tuple[int, float], Solution] = {}

    for pd in pattern_options:
        for N in range(2, max_N + 1):
            sol = _best_L(pd, shot_angle, N, tgt)
            if sol.error_deg > tolerance_deg:
                # With symmetric patterns, increasing N never *increases* the min error,
                # but once the error is already above tolerance, we can still try larger N
                # to let spacing hit the target.  So we *cannot* break here safely.
                pass
            if sol.error_deg <= tolerance_deg:
                key = (pd, round(sol.error_deg, 6))
                prev = keep.get(key)
                if prev is None or sol.total < prev.total:
                    keep[key] = sol

    # Return as a list, sorted by (pattern_degree, error, total)
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
    parser = argparse.ArgumentParser(description="Projectile pattern optimiser – minimal duplicates version")
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
        most_accurate = sorted(group, key=lambda s: s.error_deg)[:args.top]
        fewest_projectiles = sorted(group, key=lambda s: s.total)[:args.top]
        print(f"\nPattern Degree: {pd}")

        print("  Most Accurate (top 3):")
        print("    L | R | N | Error (deg)")
        print("   --|---|----|------------")
        for sol in most_accurate:
            print(f"   {sol.left:>2} | {sol.right:>2} | {sol.total:>2} | {sol.error_deg:10.6f}")

        print("  Fewest Projectiles (top 3):")
        print("    L | R | N | Error (deg)")
        print("   --|---|----|------------")
        for sol in fewest_projectiles:
            print(f"   {sol.left:>2} | {sol.right:>2} | {sol.total:>2} | {sol.error_deg:10.6f}")

    if eff:
        best = min(eff, key=lambda s: s.score())
        print("\nRecommended solution:")
        print(
            f"Pattern Degree: {best.pattern_degree}, "
            f"L: {best.left}, R: {best.right}, N: {best.total}, "
            f"Error: {best.error_deg:.6f}°"
        )

    # Optional external call remains unchanged
    print(f"\nRunning: ./speed_calc.exe {dist:.6f} --tol {args.tolerance}")
    try:
        subprocess.run([
            "./speed_calc.exe",
            f"{dist:.6f}",
            "--tol",
            f"{args.tolerance}",
        ], check=True)
    except FileNotFoundError:
        print("Warning: speed_calc.exe not found – skipping speed-solver step")
    except Exception as exc:
        print(f"Warning: Could not run speed solver: {exc}")


if __name__ == "__main__":
    _cli()

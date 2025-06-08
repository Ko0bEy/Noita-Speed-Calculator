from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
from functools import lru_cache
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


@lru_cache(maxsize=None)
def _angle_normalise(angle: float) -> float:
    return angle % 360.0


@lru_cache(maxsize=None)
def _angle_difference(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def _distance(p1: "Point", p2: "Point") -> float:
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class Solution:
    pattern_degree: int
    left: int
    right: int
    total: int
    error_deg: float

    def score(self) -> Tuple[int, float]:
        return self.total, self.error_deg


def _target_angle(p1: Point, p2: Point) -> float:
    return (360.0 - math.degrees(math.atan2(-(p2.y - p1.y), p2.x - p1.x))) % 360.0


def _step(pattern_degree: int, total: int) -> float:
    return (
        360.0 / total if pattern_degree == 180 else (2.0 * pattern_degree) / (total - 1)
    )


def _best_L(
    pattern_degree: int, shot_angle: float, total: int, target: float
) -> Solution:
    start = shot_angle - pattern_degree
    delta = _step(pattern_degree, total)
    ideal = round((_angle_normalise(target - start)) / delta)
    best: Solution | None = None
    for L in (ideal - 1, ideal, ideal + 1):
        if 0 <= L < total:
            theta_L = _angle_normalise(start + L * delta)
            err = _angle_difference(theta_L, target)
            cand = Solution(pattern_degree, L, total - L - 1, total, err)
            if best is None or err < best.error_deg - 1e-12:
                best = cand
    if best is None:
        for L in range(total):
            theta_L = _angle_normalise(start + L * delta)
            err = _angle_difference(theta_L, target)
            cand = Solution(pattern_degree, L, total - L - 1, total, err)
            if best is None or err < best.error_deg - 1e-12:
                best = cand
    return best


def _gcd3(a: int, b: int, c: int) -> int:
    return math.gcd(a, math.gcd(b, c))


def _normalised_key(sol: Solution) -> Tuple[int, ...]:
    if sol.left == 0:
        return (sol.pattern_degree, 0)
    g = _gcd3(sol.left, sol.right, sol.total)
    return (sol.pattern_degree, sol.left // g, sol.right // g, sol.total // g)


def find_efficient_solutions(
    p1: Point,
    p2: Point,
    *,
    shot_angle: float,
    pattern_options: Tuple[int, ...] = (5, 20, 30, 45, 90, 180),
    max_n: int = 100,
    tolerance: float = 0.01,
    distance: float,
) -> List[Solution]:
    tgt = _target_angle(p1, p2)
    keep: Dict[Tuple[int, ...], Solution] = {}
    for pd in pattern_options:
        for N in range(2, max_n + 1):
            sol = _best_L(pd, shot_angle, N, tgt)
            perp_error = math.sin(math.radians(sol.error_deg)) * distance
            if perp_error > tolerance * distance:
                continue
            key = _normalised_key(sol)
            prev = keep.get(key)
            if (
                prev is None
                or sol.total < prev.total
                or (sol.total == prev.total and sol.error_deg < prev.error_deg - 1e-12)
            ):
                keep[key] = sol
    return sorted(keep.values(), key=lambda s: (s.pattern_degree, s.error_deg, s.total))


def _group_by_pattern(solutions: List[Solution]) -> Dict[int, List[Solution]]:
    grouped: Dict[int, List[Solution]] = {}
    for sol in solutions:
        grouped.setdefault(sol.pattern_degree, []).append(sol)
    return grouped


def _visualize_solution(
    p1: Point, p2: Point, shot_angle: float, sol: Solution, tolerance: float = 0.01
) -> None:
    center_x, center_y = p1.x, p1.y
    dist = math.hypot(p2.x - p1.x, p2.y - p1.y)

    # Compute projectile angles
    if sol.total == 1:
        angles = [shot_angle]
    else:
        start = shot_angle - sol.pattern_degree
        step = (
            360.0 / sol.total
            if sol.pattern_degree == 180
            else (2.0 * sol.pattern_degree) / (sol.total - 1)
        )
        angles = [(start + i * step) % 360.0 for i in range(sol.total)]

    proj_x = [center_x + dist * np.cos(np.deg2rad(-a)) for a in angles]
    proj_y = [center_y - dist * np.sin(np.deg2rad(-a)) for a in angles]

    # Plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.scatter(center_x, center_y, c="blue", s=100, label="Player")
    plt.scatter(p2.x, p2.y, c="red", marker="X", s=120, label="Target")
    plt.scatter(proj_x, proj_y, c="green", s=80, marker="*", label="Projectiles")
    for x, y in zip(proj_x, proj_y):
        plt.plot([center_x, x], [center_y, y], c="gray", lw=1, ls="--")
    plt.plot(
        [center_x, p2.x], [center_y, p2.y], c="red", lw=2, label="Target Direction"
    )

    # Plot the shot angle (centreline)
    shot_line_x = [center_x, center_x + dist * np.cos(np.deg2rad(-shot_angle))]
    shot_line_y = [center_y, center_y - dist * np.sin(np.deg2rad(-shot_angle))]
    plt.plot(shot_line_x, shot_line_y, c="orange", lw=2, label="Shot Angle")

    # --- Draw tolerance circle around the target ---
    tol_radius = tolerance * dist
    tol_circle = Circle(
        (p2.x, p2.y),
        tol_radius,
        color="gold",
        alpha=0.25,
        label=f"Tolerance Area ({tol_radius:.0f}px)",
    )
    ax.add_patch(tol_circle)

    # Labels
    N = len(proj_x)
    offset = max(dist * 0.04, 25)

    def label_proj(idx, text, color):
        dx = proj_x[idx] - center_x
        dy = proj_y[idx] - center_y
        norm = np.hypot(dx, dy)
        if norm == 0:
            lx, ly = proj_x[idx] + offset, proj_y[idx] + offset
        else:
            lx = proj_x[idx] + (dx / norm) * offset
            ly = proj_y[idx] + (dy / norm) * offset
        ax.annotate(
            text,
            (lx, ly),
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            color=color,
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, lw=0),
        )

    if N > 0:
        label_proj(0, "1", "black")
        label_proj(N - 1, str(N), "black")
        # Optional: label the closest one to target direction in blue, as in the original
    plt.axis("equal")
    plt.xlabel("+X (right)")
    plt.ylabel("+Y (down)")
    plt.title(
        f"Projectile Pattern Visualization\nPattern Degree: {sol.pattern_degree}°"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ax.invert_yaxis()
    # Optionally: ScalarFormatter for axes
    from matplotlib.ticker import ScalarFormatter

    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def _parse_pattern_list(value: str) -> Tuple[int, ...]:
    if not value:
        raise argparse.ArgumentTypeError("pattern list must not be empty")
    try:
        parts = [int(part) for part in value.replace(",", " ").split() if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pattern degrees must be integers") from exc
    if any(p <= 0 or p > 180 for p in parts):
        raise argparse.ArgumentTypeError("pattern degrees must be in the range 1‑180")
    return tuple(set(parts))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Projectile pattern optimiser")
    parser.add_argument("x0", type=float, help="Shooter X coordinate")
    parser.add_argument("y0", type=float, help="Shooter Y coordinate")
    parser.add_argument("x1", type=float, help="Target X coordinate")
    parser.add_argument("y1", type=float, help="Target Y coordinate")
    parser.add_argument(
        "-a",
        "--shot-angle",
        type=float,
        default=90.0,
        help="Shot centreline angle (deg, CW from +X)",
    )
    parser.add_argument(
        "-n", "--max-n", type=int, default=100, help="Max projectiles to test"
    )
    parser.add_argument(
        "-t",
        "--tol",
        type=float,
        default=0.01,
        help="Max allowed perpendicular error as a fraction of the distance",
    )
    parser.add_argument(
        "-c",
        "--coefs",
        type=float,
        nargs="+",
        help="Override speed multipliers for speed_calc",
    )
    parser.add_argument(
        "-u",
        "--uncapped",
        type=int,
        nargs="*",
        help="Indices whose multipliers are uncapped for speed_calc",
        default=[],
    )

    parser.add_argument(
        "-p",
        "--pattern-options",
        type=str,
        nargs="+",
        default=["5", "20", "30", "45", "90", "180"],
        help="Space- or comma-separated list of pattern degrees (default: 5 20 30 45 90 180)",
    )
    parser.add_argument(
        "--skip-speed-calc",
        action="store_true",
        help="Do not invoke external speed calculator",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Show matplotlib visualization of the recommended solution",
    )
    parser.add_argument(
        "--show-few",
        type=int,
        default=3,
        help="How many small solutions to show per category",
    )
    parser.add_argument(
        "--show-accurate",
        type=int,
        default=3,
        help="How many accurate solutions to show per category",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help="Comma-separated sort priorities to pass to speed_calc "
        "(e.g. rel_err,nz,sum). Supported: nz, sum, rel_err, max_exp.",
    )
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    if not all(math.isfinite(coord) for coord in (args.x0, args.x1, args.y0, args.y1)):
        parser.error("invalid coordinates.")

    if args.max_n < 2:
        parser.error("--max-n must be at least 2")

    pattern_degrees = []
    for arg in args.pattern_options:
        for val in arg.split(","):
            val = val.strip()
            if not val:
                continue
            try:
                iv = int(val)
            except ValueError:
                parser.error(f"Invalid pattern degree: {val}")
            if iv < 1 or iv > 180:
                parser.error(f"Pattern degrees must be in range 1-180, got: {iv}")
            pattern_degrees.append(iv)
    pattern_degrees = tuple(pattern_degrees)
    args.pattern_options = pattern_degrees

    p1 = Point(args.x0, args.y0)
    p2 = Point(args.x1, args.y1)
    dist = _distance(p1, p2)
    if dist == 0:
        print("Distance is zero; nothing to solve.")
        return

    eff = find_efficient_solutions(
        p1,
        p2,
        shot_angle=args.shot_angle,
        pattern_options=args.pattern_options,
        max_n=args.max_n,
        tolerance=args.tol,
        distance=dist,
    )
    tgt = _target_angle(p1, p2)
    print(f"Target angle  : {tgt:.6f}° (clockwise)")
    print(f"Distance      : {dist:.6f}")
    print(f"Abs tolerance : {args.tol:.6f} × distance = {args.tol * dist:.0f}px\n")
    grouped = _group_by_pattern(eff)
    print("Solutions grouped by pattern degree (duplicates pruned):")
    for pd in sorted(grouped):
        group = grouped[pd]
        most_accurate = heapq.nsmallest(
            args.show_accurate, group, key=lambda s: s.error_deg
        )
        fewest_projectiles = heapq.nsmallest(
            args.show_few, group, key=lambda s: s.total
        )
        print(f"\nPattern Degree: {pd}")
        print(f"  Most Accurate (top {args.show_accurate}):")
        print("    L | R  | N  | Error (deg)")
        print("   ---|----|----|------------")
        for sol in most_accurate:
            print(
                f"   {sol.left:>2} | {sol.right:>2} | {sol.total:>2} | {sol.error_deg:10.6f}"
            )
        print(f"  Fewest Projectiles (top {args.show_few}):")
        print("    L | R  | N  | Error (deg)")
        print("   ---|----|----|------------")
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
            f"Error: {best.error_deg:.6f}° ({math.sin(math.radians(best.error_deg)) * dist:.0f}px)"
        )
        if args.visualize:
            _visualize_solution(p1, p2, args.shot_angle, best, args.tol)
    if args.skip_speed_calc:
        print("\nSpeed calculation skipped (--skip-speed-calc specified).")
    else:
        exe_dir = os.path.dirname(sys.executable)
        speed_calc = os.path.join(exe_dir, "speed_calc.exe")
        cmd = [speed_calc, f"{dist:.6f}", "--tol", f"{args.tol}"]
        if args.coefs:
            cmd.extend(["--coefs"] + [str(x) for x in args.coefs])
        if args.uncapped:
            cmd.extend(["--uncapped"] + [str(u) for u in args.uncapped])
        if args.sort:
            cmd.extend(["--sort", args.sort])
        if args.top_n:
            cmd.extend(["--top-n", str(args.top_n)])
        print(
            f"\nRunning: {' '.join(os.path.basename(x) if i == 0 else x for i, x in enumerate(cmd))}"
        )
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("Error: speed_calc.exe not found in the same folder. Aborting.")
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            print(f"Error: speed solver exited with status {exc.returncode}.")
            sys.exit(exc.returncode)

    if args.visualize:
        plt.show()


if __name__ == "__main__":
    _cli()

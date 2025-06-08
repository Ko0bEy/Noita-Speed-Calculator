import argparse
import bisect
import itertools
import math
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Set, Optional, Dict, Any
from functools import lru_cache

DEFAULT_COEFS: List[float] = [1.2, 0.3, 0.32, 0.33, 0.75, 1.68, 2.0, 2.5, 7.5]
BASE_SPEED: float = 7.92
PROD_CAP: float = 20.0
DEFAULT_TOP_N: int = 50
LABELS: List[str] = ["flyup", "heavy", "accel", "phasing", "explo", "decel", "slither", "speed", "light"]
DEFAULT_SORTING_PRIORITY: Tuple[str, ...] = ('nz', 'sum', 'rel_err', 'max_exp')

@lru_cache(maxsize=None)
def _cached_log(x: float) -> float:
    return math.log(x)

@dataclass(slots=True, frozen=True)
class _HalfVector:
    log_cap: float
    log_full: float
    nz: int
    s: int
    vec: Tuple[int, ...]

@dataclass(slots=True)
class Solution:
    vec: Tuple[int, ...]
    rel_err: float
    signed_err: float
    nz: int
    sum: int
    max_exp: int

    def __init__(self, vec: Tuple[int, ...], rel_err: float, signed_err: float, nz: int, sum_: int):
        self.vec = vec
        self.rel_err = rel_err
        self.signed_err = signed_err
        self.nz = nz
        self.sum = sum_
        self.max_exp = max(vec)

    def key(self, sort_priority: Sequence[str] = DEFAULT_SORTING_PRIORITY) -> Tuple[Any, ...]:
        key_tuple: List[Any] = []
        for k in sort_priority:
            val = getattr(self, k)
            if k == 'rel_err':
                val = abs(val)
            key_tuple.append(val)
        return tuple(key_tuple)

def _upper_bounds(
    a: Sequence[float],
    uncapped: Set[int],
    log_c_hi: float,
    prod_cap: float,
    x0_margin: int,
) -> List[int]:
    ln = _cached_log
    bounds = [int(math.ceil(log_c_hi / ln(a[0])) + x0_margin)]
    for i, ai in enumerate(a[1:], 1):
        if i in uncapped or ai <= 1.0:
            bounds.append(int(math.floor(log_c_hi / abs(math.log(ai)))))
        else:
            bounds.append(int(math.floor(math.log(prod_cap) / math.log(ai))))
    return bounds

def _enumerate_half(
    idxs: List[int],
    ub: List[int],
    log_a: List[float],
    uncapped: Set[int],
    log_cap: float,
) -> List[_HalfVector]:
    res: List[_HalfVector] = []
    for xs in itertools.product(*(range(ub[i] + 1) for i in idxs)):
        log_full = sum(x * log_a[i] for x, i in zip(xs, idxs))
        log_cap_sum = sum(x * log_a[i] for x, i in zip(xs, idxs) if i not in uncapped)
        if log_cap_sum < log_cap:
            res.append(_HalfVector(log_cap_sum, log_full, sum(x > 0 for x in xs), sum(xs), xs))
    return res

def _split(n: int) -> Tuple[List[int], List[int]]:
    mid = n // 2
    return list(range(1, 1 + mid)), list(range(1 + mid, 1 + n))

def find_sparse_solutions(
    coefs: Sequence[float],
    distance: float,
    *,
    base_speed: float = BASE_SPEED,
    rel_tol: float = 5e-3,
    prod_cap: float = PROD_CAP,
    top_n: int = DEFAULT_TOP_N,
    uncapped_indices: Optional[Sequence[int]] = None,
    x0_margin: int = 10,
    sort_priority: Sequence[str] = DEFAULT_SORTING_PRIORITY
) -> List[Solution]:
    """
    Main function to find sparse solutions to the given distance and coefficients.
    Returns a list of Solution objects sorted according to the sort_priority.
    """
    if len(coefs) < 1 or coefs[0] <= 1:
        raise ValueError("At least 1 coefficient required, with coefs[0] > 1.")
    if min(coefs) <= 0:
        raise ValueError("All coefficients must be positive.")
    uncapped: Set[int] = {0} | set(map(int, uncapped_indices or []))
    if any(i < 0 or i >= len(coefs) for i in uncapped):
        raise ValueError
    target_c = distance / base_speed
    log_c = _cached_log(target_c)
    log_c_hi = _cached_log(target_c * (1 + rel_tol))
    log_cap = _cached_log(prod_cap)
    log_a = [_cached_log(a) for a in coefs]
    ub = _upper_bounds(coefs, uncapped, log_c_hi, prod_cap, x0_margin)
    n_other = len(coefs) - 1
    idx_A, idx_B = _split(n_other)
    list_A = _enumerate_half(idx_A, ub, log_a, uncapped, log_cap)
    list_B = _enumerate_half(idx_B, ub, log_a, uncapped, log_cap)
    list_B.sort(key=lambda h: h.log_cap)
    caps_B = [h.log_cap for h in list_B]
    ln_a0 = log_a[0]
    sols: List[Solution] = []
    for ha in list_A:
        rem = log_cap - ha.log_cap
        cutoff = bisect.bisect_left(caps_B, rem)
        for hb in list_B[:cutoff]:
            log_full = ha.log_full + hb.log_full
            x0 = int(round((log_c - log_full) / ln_a0))
            if x0 < 0:
                continue
            total_log = log_full + x0 * ln_a0
            rel_err = abs(math.expm1(total_log - log_c))
            if rel_err > rel_tol:
                continue
            dist_est = math.exp(total_log) * base_speed
            err = dist_est - distance
            vec = (x0,) + ha.vec + hb.vec
            sols.append(Solution(vec, rel_err, err, ha.nz + hb.nz, ha.s + hb.s))
    sol_map: Dict[Tuple[int, ...], Solution] = {}
    for s in sols:
        if s.vec not in sol_map or s.rel_err < sol_map[s.vec].rel_err:
            sol_map[s.vec] = s
    return sorted(sol_map.values(), key=lambda s: s.key(sort_priority))[: top_n]

def _labels(coefs: Sequence[float]) -> List[str]:
    if list(coefs) == DEFAULT_COEFS:
        return LABELS
    return [f"{a}" for a in coefs]

def _col_w(sols: List[Solution], labels: List[str]) -> int:
    if not sols:
        return 4
    return max(4, len(str(max(max(s.vec) for s in sols))), max(len(l) for l in labels))

def _header(labels: List[str], w: int) -> str:
    cols = " ".join(l.rjust(w) for l in labels)
    return f"{cols} | multiplier |   distance  |   error | rel_error"

def _row(sol: Solution, coefs: Sequence[float], w: int) -> str:
    exps = " ".join(f"{x:>{w}d}" if x else " " * w for x in sol.vec)
    multiplier = math.prod(a ** x for a, x in zip(coefs, sol.vec))
    dist_est = multiplier * BASE_SPEED
    return (
        f"{exps} | {multiplier:10.2f} | {dist_est:11.5g} | "
        f"{sol.signed_err:+7.0f}px | {sol.rel_err:9.0e}"
    )

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("distance", type=float)
    p.add_argument("--coefs", "-c", type=float, nargs="+", default=DEFAULT_COEFS)
    p.add_argument("--tol", "-t", type=float, dest="rel_tol", default=5e-3)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--uncapped", "-u", type=int, nargs="*", default=[])
    p.add_argument(
        "--sort", type=str, default="nz,sum,rel_err,max_exp",
        help=(
            "Sort priority, comma-separated. Use -field for descending. non-provided are inferred from default."
            "Supported: nz, sum, rel_err, max_exp. "
            "Example: --sort -nz,rel_err"
        ),
    )
    return p
def _run_cli(argv: Optional[List[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.distance <= 0:
        print(f"invalid distance: {args.distance}")
        return
    if any(c <= 0 for c in args.coefs):
        print(f"non-positive coefficient found.")
        return
    input_sort = [s.strip() for s in args.sort.split(",") if s.strip()]
    # infer rest from default ordering
    all_fields = [field.lstrip('-') for field in input_sort]
    inferred = [s for s in DEFAULT_SORTING_PRIORITY if s not in all_fields]
    sort_priority = input_sort + inferred
    abs_tol = args.rel_tol * args.distance
    start = time.perf_counter()
    sols = find_sparse_solutions(
        coefs=args.coefs,
        distance=args.distance,
        base_speed=BASE_SPEED,
        rel_tol=args.rel_tol,
        prod_cap=PROD_CAP,
        top_n=args.top_n,
        uncapped_indices=args.uncapped,
        sort_priority=sort_priority,
    )
    ms = (time.perf_counter() - start) * 1_000
    if not sols:
        print("No solution found.")
        return
    labels = _labels(args.coefs)
    w = _col_w(sols, labels)
    print(f"Target distance = {args.distance}  (Error tolerance={abs_tol:.0f}px)")
    head = _header(labels, w)
    print(head)
    print("-" * len(head))
    for s in sols:
        print(_row(s, args.coefs, w))
    print(f"\n{len(sols)} solution(s) shown in {ms:.2f} ms")
    print("Best solution at the top!")

if __name__ == "__main__":
    _run_cli()

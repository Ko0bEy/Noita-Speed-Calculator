from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

from utils import (
    get_starting_nodes,
    calculate_fly_up_multipliers,
    check_solution,
    print_results,
    extend_coefs,
    choose_increase_decrease,
    choose_wisely,
    set_globals,
    get_multiplier
)


def initialize(args):
    strategy = args.s
    coefs = tuple(sorted(float(coef) for coef in args.coefs))
    uncapped_coef = args.u
    specials_idx, extended_coefs = extend_coefs(coefs, uncapped_coef)
    max_specials = args.c
    v0 = args.v0
    set_globals(coefs, uncapped_coef, _v0=v0)
    # Search parameters
    max_iter, budget_weight = 0, 0
    mode = args.m
    if mode == "shallow":
        max_iter = 3
    elif mode == "normal":
        max_iter = 6
    elif mode == "deep":
        max_iter = 9
    elif mode == "verydeep":
        max_iter = 18

    if args.iter is not None:
        max_iter = args.iter

    budget = args.b
    max_rel_err = None

    if budget == "budget":
        budget_weight = 2.0
        max_rel_err = 0.035
    elif budget == "normal":
        budget_weight = 1.0
        max_rel_err = 0.01
    elif budget == "accurate":
        budget_weight = 0.5
        max_rel_err = 0.001
    elif budget == "ignorebudget":
        budget_weight = 0.0
        max_rel_err = 0.001

    if args.w is not None:
        budget_weight = args.w

    if args.e is not None:
        max_rel_err = args.e

    distance = args.distance
    if args.min is None:
        dmin = (1 - max_rel_err) * distance
    else:
        dmin = args.min
    if args.max is None:
        dmax = (1 + max_rel_err) * distance
    else:
        dmax = args.max

    target_multiplier = distance / v0
    min_multiplier = dmin / v0
    max_multiplier = dmax / v0

    return (
        distance,
        coefs,
        dmin,
        dmax,
        v0,
        max_rel_err,
        max_iter,
        target_multiplier,
        min_multiplier,
        max_multiplier,
        budget,
        budget_weight,
        strategy,
        specials_idx,
        extended_coefs,
        max_specials
    )


def _search(
    target,
    target_min,
    target_max,
    coefs=(0.3, 0.32, 0.33, 0.75, 1.68, 2, 2.5, 7.5),
    max_iter=10,
    max_specials=200,
    strategy="complete",
):
    def _step(_node: float, _path: tuple[int]):
        candidates = candidate_chooser(_node, target)
        new_solutions = set()
        for a, c in candidates:
            _new_path = list(_path)
            _new_path[coefs.index(c)] += a
            # new_solution = (node * c**a, tuple(new_path))
            new_solution = (get_multiplier(_new_path), tuple(_new_path))
            new_solutions.add(
                (new_solution, check_solution(new_solution[0], target_min, target_max))
            )
        return new_solutions

    min_flyups, fly_up_multipliers = calculate_fly_up_multipliers(target, max_len=max_specials)
    candidate_chooser = None
    if strategy == "complete":
        candidate_chooser = choose_increase_decrease
    else:
        assert strategy == "wise"
        candidate_chooser = choose_wisely

    last_nodes = get_starting_nodes(min_flyups, fly_up_multipliers)
    current_nodes = set()
    for _iter in tqdm(range(max_iter), total=max_iter):
        solutions = defaultdict(set)
        for node, path in last_nodes:
            for (new_node, new_path), check in _step(node, path):
                current_nodes.add((new_node, new_path))
                if check:
                    solutions[new_node].add(new_path)
        last_nodes = current_nodes
        current_nodes = set()
        yield solutions
    # return solutions


def find_solutions(*args):
    solutions = defaultdict(set)
    for solution_batch in _search(*args):
        for multiplier, paths in solution_batch.items():
            node = multiplier
            solutions[node] = solutions[node].union(paths)
    return solutions


def main():
    parser = ArgumentParser()
    parser.add_argument("distance", default=35840, type=int)
    parser.add_argument(
        "-mode",
        "--m",
        default="shallow",
        required=False,
        type=str,
        choices=["normal", "shallow", "deep", "verydeep"],
    )
    parser.add_argument(
        "-budget",
        "--b",
        default="normal",
        required=False,
        type=str,
        choices=["accurate", "normal", "budget", "ignorebudget"],
    )
    parser.add_argument("-distance_min", "--min", type=int)
    parser.add_argument("-distance_max", "--max", type=int)
    parser.add_argument("-v0", "--v0", default=7.92, type=float)
    parser.add_argument("-err", "--e", type=float, default=0.035)
    parser.add_argument("-weight", "--w", type=float)
    parser.add_argument("-uncapped_coef", "--u", type=float, default=1.2)
    parser.add_argument("-cap_uncap", "--c", type=int, default=200)
    parser.add_argument(
        "-strategy", "--s", type=str, choices=["complete", "wise"], default="wise"
    )
    parser.add_argument(
        "-coefs", nargs="+", default=(0.3, 0.32, 0.33, 0.75, 1.68, 2, 2.5, 7.5)
    )
    parser.add_argument("-iter_max", "--iter", type=int)

    args = parser.parse_args()
    print(args)

    (
        distance,
        coefs,
        dmin,
        dmax,
        v0,
        max_rel_err,
        max_iter,
        target_multiplier,
        min_multiplier,
        max_multiplier,
        budget,
        budget_weight,
        strategy,
        specials_idx,
        extended_coefs,
        max_specials
    ) = initialize(args)

    solutions = find_solutions(
        target_multiplier,
        min_multiplier,
        max_multiplier,
        coefs,
        max_iter,
        max_specials,
        strategy,
    )

    if solutions:
        print_results(
            solutions,
            50,
            distance,
            target_multiplier,
            budget_weight,
        )


if __name__ == "__main__":
    main()

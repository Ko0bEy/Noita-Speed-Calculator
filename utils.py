import numpy as np
import pickle
import functools

# floating point precisiion (i think)
epsilon = 1.175494351e-38

# somewhat arbitrary measure of how hard numbers are to divide by
divide_chain_difficulty = [0, 1, 2, 2, 3, 4, 3, 4, 4, 4, 3, 3, 4, 4, 6, 7, 4, 4, 5, 5, 4, 4, 5, 6, 5, 5, 6, 5, 5, 7, 4, 4, 4, 5, 6, 7, 5, 5, 7, 7, 4, 4, 6, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 4, 4, 6, 7, 5, 5, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 6, 6, 7, 7, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

# initialize some globals to be set by set_globals. Makes some things easier and/or faster.
coefs = []
log_coefs = []
increasers = []
decreasers = []
increaser_candidates = set()
decreaser_candidates = set()
extended_coefs = []
extension_idx = 0
uncapped_coef = 0
v0 = 7.92

modifier_names = {
    0.3: "HeavyS",
    0.32: "Accel",
    0.33: "Phasing",
    0.5: "GNuke",
    0.75: "ExploP",
    1.1: "SlimeB",
    1.2: "FlyUp",
    1.68: "Decel",
    1.75: "FasterP",
    2.0: "Chaotic",
    2.5: "Speed+",
    7.5: "LightS",
}


# call this once to set up the right values for the utility functions
def set_globals(_coefs: tuple[float], special_coef: float = 1.2, _v0=7.92) -> None:
    """
    call this once to set up the right values for the utility functions.
    :param _coefs: modifiers to use
    :param special_coef: uncapped modifier to use (Fly Upwards/Downwards, Faster Projectiles)
    :return: None
    """
    global log_coefs, increasers, decreasers, coefs, extended_coefs, extension_idx, increaser_candidates, decreaser_candidates, uncapped_coef, v0
    coefs = _coefs
    extension_idx, extended_coefs = extend_coefs(coefs, special_coef)
    log_coefs = np.log(_coefs).tolist()
    increasers = tuple(x for x in _coefs if x > 1)
    decreasers = tuple(x for x in _coefs if x < 1)
    increaser_candidates = set((1, c) for c in increasers)
    decreaser_candidates = set((1, c) for c in decreasers)
    uncapped_coef = special_coef
    v0 = _v0


def rel_error(val: float, tar: float) -> float:
    """
    relative error between val and tar
    :param val: value
    :param tar: target
    :return: relative error
    """
    return np.abs(val / tar - 1)


@functools.cache
def difficulty(path, budget_weight, specials_idx):
    """
    Somewhat arbitrary measure of how hard a solution is to build. Used to sort solutions.
    :param path: list of modifier amounts
    :param budget_weight: how much weight should be given to the difficulty for the overall score
    :param specials_idx: where do capped modifiers stop, and uncapped modifiers start
    :return:
    """
    return np.round(
        budget_weight
        * (
            +sum(
                divide_chain_difficulty[p] ** 1.5
                if i < specials_idx
                else divide_chain_difficulty[p]
                for i, p in enumerate(path)
            )
            + (np.count_nonzero(path) - 1) * np.count_nonzero(path) / 2
        ),
        2,
    )


@functools.cache
def error_bonus(value, target, cap=np.inf):
    """
    Rewards solutions for accuracy
    :param value: speed multiplier of the solution
    :param target: target speed multiplier
    :param cap: max bonus awarded for accuracy
    :return:
    """
    score = np.round(np.log(1 / rel_error(value, target) ** 1.8), 2)
    return min(score, cap)


@functools.cache
# @profile
def choose_wisely(node, target) -> set[tuple[int]]:
    """
    from a given path and speed multiplier, choose modifiers and corresponding amounts to add, and check in the next iteration
    :param node: speed multiplier
    :param target: target speed multiplier
    :return: a set of tuples containing the modifier and amount to add
    """
    relation = target / node
    amounts_candidates = set()
    if coefs[0] < relation < coefs[-1]:
        return choose_increase_decrease(node, target)
    else:
        for amount, candidate in choose_increase_decrease(node, target):
            amount = int(np.log(relation) / log_coefs[coefs.index(candidate)])
            amounts_candidates = amounts_candidates.union(
                {(amount, candidate), (amount + 1, candidate)}
            )
    return amounts_candidates


@functools.cache
# @profile
def choose_increase_decrease(node, target) -> set[tuple[int]]:
    """
    chooses modifiers via a simple heuristic, and adds 1 of each for the next iteration.
    :param node: speed multiplier
    :param target: target speed multiplier
    :return: a set of tuples containing the modifier and amount to add
    """
    if node < target:
        return increaser_candidates
    elif node > target:
        return decreaser_candidates


@functools.cache
def path_cost(
    value: float,
    path: tuple[int],
    target: float,
    budget_weight: float,
    specials_idx: int,
    acc_cap: float = np.inf,
) -> float:
    """
    How good is the solution, regarding difficulty to build, and accuracy
    :param value: speed multiplier
    :param path: modifier amounts
    :param target: target speed multiplier
    :param budget_weight: weight of the difficulty score
    :param specials_idx: where do uncapped modifiers start
    :param acc_cap: max points awarded for accuracy
    :return:
    """
    diff = difficulty(path, budget_weight, specials_idx)
    err = error_bonus(value, target, cap=acc_cap)
    return diff - err


def with_sign(s: [float, np.array]) -> str:
    """
    adds a plus sign to positive values
    :param s: value
    :return: string, for example +0.43
    """
    s = float(s)
    if s >= 0:
        return "+" + str(s)
    else:
        return str(s)


def to_equation(
    sol: tuple[int, tuple[int]],
    distance: int,
    target: float,
    budget_weight: float,
    specials_idx: int,
) -> str:
    """
    Returns a solution in a human-readable form
    :param sol: solution (speed_multiplier, path)
    :param distance: desired distance to travel
    :param target: target speed multiplier
    :param budget_weight: weight of difficulty score
    :param specials_idx: where do uncapped modifiers start
    :return:
    """
    message = ""
    for i, c in enumerate(extended_coefs):
        if sol[1][i] != 0:
            term = f"{c:.2f}^{sol[1][i]}"
            term = term + " " * (8 - len(term)) + " * "
            message += term
        else:
            message += " " * 9 + "* "
    message = (
        message[:-2]
        + f"=\t{np.round(sol[0], 5)}\t({with_sign(int(get_distance(sol[0]) - distance))}px) \t({with_sign(np.round(100 * rel_error(sol[0], target), 7))}%)\t({difficulty(sol[1], budget_weight, specials_idx)})\t\t({error_bonus(sol[0],target)})\t\t({path_cost(*sol, target, budget_weight, specials_idx):.2f})"
    )
    return message


def print_headline(spacing=11):
    message = ""
    for c in extended_coefs:
        sub_message = modifier_names[c]
        message += sub_message + " " * (spacing - len(sub_message))

    print(message + "\tSpeedMul\tError\t\t%-Error\t\tDifficulty\tAccuracy")


def print_results(
    solutions: dict, n: int, distance: int, target: float, budget_weight: float
) -> None:
    """
    sorts the results, and prints the output
    :param solutions: solutions
    :param n: number of solutions to print
    :param distance: desired distance
    :param target: target speed mutiplier
    :param budget_weight: weight of the difficulty
    :return: None
    """
    solutions = sort_solutions(
        solutions,
        lambda val, path: path_cost(val, path, target, budget_weight, extension_idx),
    )
    print()
    print_headline()
    for sol in solutions[:n]:
        print(to_equation(sol, distance, target, budget_weight, extension_idx))


def extend_coefs(_coefs: tuple[float], special_coefs: float = 1.2) -> tuple[int, tuple]:
    """
    attaches uncapped modifiers to the end of the list
    :param _coefs: capped modifiers
    :param special_coefs: uncapped modifiers
    :return:
    """
    extended_coefs = tuple(list(_coefs) + [special_coefs])
    return len(_coefs), extended_coefs


def sort_solutions(
    tree: dict[float: set[tuple[int]]], scorer: callable, reverse: bool = False
) -> list[tuple]:
    """
    sorts the solutions based on the provided scorer
    :param tree: solutions
    :param scorer: function to use for scoring
    :param reverse: sort in reverse?
    :return: sorted list of sollutions
    """
    return sorted(
        (
            (multiplier, path)
            for multiplier in tree
            for path in tree[multiplier]
            # if sum(1 for s in sol if s != 0) <= 3
        ),
        key=lambda x: scorer(*x),
        reverse=reverse,
    )


@functools.cache
def check_solution(multiplier: float, target_min: float, target_max: float) -> bool:
    """
    check whether a given multiplier falls in the desired range
    :param multiplier: speed multiplier
    :param target_min: minimum speed multiplier
    :param target_max: maximum speed multiplier
    :return: True/False
    """
    return target_min < multiplier < target_max


def get_starting_nodes(
    min_flyups: int, fly_up_multipliers: list[float]
) -> set[tuple[float, tuple]]:
    """
    return the set of potential solutions to start the search with.
    :param min_flyups: how many of the uncapped modifier do we have to use at least?
    :param fly_up_multipliers: pre-calculated list of possible multipliers using the uncapped modifier
    :return:
    """
    nodes = set()
    for i, m in enumerate(fly_up_multipliers):
        flyup_count = min_flyups + i
        path = tuple([0 for _ in extended_coefs[:-1]] + [flyup_count])
        m = get_multiplier(path)
        if m < np.inf:
            nodes.add((get_multiplier(path), path))
    return nodes


def get_multiplier(path: [tuple[int], list[int]]) -> float:
    """
    calculates speed multiplier from a given list of modifier amounts
    :param path: modifier amounts
    :return: speed multiplier
    """
    return float(
        np.prod([c ** path[i] for i, c in enumerate(extended_coefs)], dtype=np.float32)
    )


def get_distance(multiplier: float) -> float:
    """
    calculates the distance traveled, using a give speed multiplier
    :param multiplier: speed multiplier
    :param v0: effective base speed (tentacle: 7.92)
    :return: distance traveled
    """
    return multiplier * v0


def calculate_fly_up_multipliers(
    target_multiplier: float, max_len: int = 200
) -> tuple[int, np.array]:
    """
    Pre-calculates all possible multipliers reached by the uncapped modfier
    :param target_multiplier: target multiplier
    :param max_len: maximum amount of entries to be generated
    :return: minimum amount of uncapped modifiers needed, array of multipliers
    """
    uncapped_multipliers = []
    cnt = 0
    res = 1
    min_count = None
    while True:
        cnt += 1
        res = res * uncapped_coef
        if 1 / res < epsilon:
            break
        if 20 >= target_multiplier / res:
            if min_count is None:
                min_count = cnt
                max_len -= min_count
            uncapped_multipliers.append(res)
        elif target_multiplier / res < epsilon:
            break
        if len(uncapped_multipliers) >= max_len:
            break
    uncapped_multipliers = np.flip(np.asarray(uncapped_multipliers))
    return min_count, uncapped_multipliers

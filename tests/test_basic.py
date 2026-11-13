"""Run with:  python -m pytest -q

The checker tests pass from the start. The projection and gradient tests fail until
project_simplex and objective_and_grad are written (they are skipped while those
functions raise NotImplementedError)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.objective import (check_gradient, constraint_violation, init_weights,  # noqa: E402
                           objective_and_grad, project_simplex, round_grid)
from src.sudoku import is_solved, is_valid_solution, parse_grid  # noqa: E402

PUZZLE = "000009000200080105080560930020000600900000000000103079610307008390005000005612700"
SOLUTION = "561239487239784165784561932123978654978456321456123879612397548397845216845612793"


def _skip_if_todo(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not written yet")


def test_checker_accepts_solution():
    p, s = parse_grid(PUZZLE), parse_grid(SOLUTION)
    assert is_valid_solution(s) and is_solved(s, p)


def test_checker_rejects_swap():
    s = parse_grid(SOLUTION)
    s[0, 0], s[0, 1] = s[0, 1], s[0, 0]
    assert not is_valid_solution(s)


def test_one_hot_solution_has_zero_violation():
    s = parse_grid(SOLUTION)
    x = np.zeros((9, 9, 9))
    for r in range(9):
        for c in range(9):
            x[r, c, s[r, c] - 1] = 1
    assert constraint_violation(x) == 0
    assert (round_grid(x) == s).all()


def test_projection_lands_on_simplex():
    v = np.random.default_rng(0).normal(size=(9, 9, 9))
    w = _skip_if_todo(project_simplex, v)
    assert w.shape == v.shape
    assert (w >= -1e-12).all()
    assert np.allclose(w.sum(axis=-1), 1)


def test_projection_keeps_points_already_on_simplex():
    x, _ = init_weights(parse_grid(PUZZLE))
    w = _skip_if_todo(project_simplex, x)
    assert np.allclose(w, x)


def test_gradient_matches_finite_differences():
    x, _ = init_weights(parse_grid(PUZZLE), noise=0.1, seed=1)
    _skip_if_todo(objective_and_grad, x, 0.3)
    err = check_gradient(lambda z: objective_and_grad(z, 0.3), x)
    assert err < 1e-5


# ---------- Douglas-Rachford helpers ----------
from src.methods import (douglas_rachford, from_groups, project_onehot,  # noqa: E402
                         to_groups)


@pytest.mark.parametrize("N", [4, 9, 16])
def test_groups_roundtrip(N):
    x = np.random.default_rng(0).random((N, N, N))
    for kind in ["cell", "row", "col", "box"]:
        assert np.allclose(from_groups(to_groups(x, kind), kind, N), x)


def test_box_group_is_a_box():
    x = np.zeros((9, 9, 9))
    x[0:3, 3:6, 4] = 1                      # top-middle box, digit 5
    G = to_groups(x, "box")
    assert (G.sum(axis=1) == 9).sum() == 1  # exactly one group covers the whole box
    assert G.sum() == 9


def test_onehot_projection_one_per_group():
    x = np.random.default_rng(1).random((9, 9, 9))
    p = parse_grid(PUZZLE)
    for kind in ["cell", "row", "col", "box"]:
        y = project_onehot(x, kind, p)
        assert (to_groups(y, kind).sum(axis=1) == 1).all()
    y = project_onehot(x, "cell", p)
    mask = p > 0
    assert ((y.argmax(axis=2) + 1)[mask] == p[mask]).all()


def test_douglas_rachford_solves_example():
    p = parse_grid(PUZZLE)
    res = douglas_rachford(p, max_iter=3000)
    assert res.solved
    assert (res.grid == parse_grid(SOLUTION)).all()


@pytest.mark.parametrize("n_sets,order,start", [(5, "paper", "paper"), (4, "paper", "paper"),
                                                (4, "rules_first", "uniform")])
def test_douglas_rachford_variants_solve_example(n_sets, order, start):
    p = parse_grid(PUZZLE)
    res = douglas_rachford(p, n_sets=n_sets, order=order, start=start)
    assert res.solved and (res.grid == parse_grid(SOLUTION)).all()


# ---------- N x N support ----------
from src.sudoku import grid_to_string, make_puzzle, random_solution  # noqa: E402
from src.objective import box_sums  # noqa: E402


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_random_solution_is_valid(n):
    sol = random_solution(n, np.random.default_rng(n))
    assert sol.shape == (n * n, n * n)
    assert is_valid_solution(sol)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_grid_string_roundtrip(n):
    rng = np.random.default_rng(0)
    puz = make_puzzle(random_solution(n, rng), 0.5, rng)
    assert (parse_grid(grid_to_string(puz)) == puz).all()


def test_kaggle_format_still_parses():
    assert parse_grid(PUZZLE).shape == (9, 9)
    assert grid_to_string(parse_grid(SOLUTION)) == SOLUTION


@pytest.mark.parametrize("n", [3, 4])
def test_one_hot_solution_box_sums_are_one(n):
    sol = random_solution(n, np.random.default_rng(1))
    N = n * n
    x = np.zeros((N, N, N))
    r, c = np.indices((N, N))
    x[r, c, sol - 1] = 1
    assert np.allclose(box_sums(x), 1)
    assert (round_grid(x) == sol).all()
    assert constraint_violation(x) == 0


def test_douglas_rachford_solves_16x16():
    rng = np.random.default_rng(3)
    puz = make_puzzle(random_solution(4, rng), 0.5, rng)
    res = douglas_rachford(puz)
    assert res.solved and is_solved(res.grid, puz)

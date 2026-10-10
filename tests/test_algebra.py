from fractions import Fraction

import pytest

from nkiequiv import algebra as A
from nkiequiv import z3check


@pytest.fixture(autouse=True)
def fresh():
    A.reset_tables()
    yield


def xs(n, name="x"):
    return [A.var(name, (i,)) for i in range(n)]


def test_polynomial_reassociation():
    x, y = xs(4), xs(4, "y")
    left = A.add_many(A.mul(a, b) for a, b in zip(x, y))
    right = A.add(A.add_many(A.mul(a, b) for a, b in zip(x[2:], y[2:])),
                  A.add_many(A.mul(a, b) for a, b in zip(x[:2], y[:2])))
    assert left == right


def test_softmax_shift_invariance():
    x = xs(5)
    m = A.maximum(x)
    naive = [A.exp(a) * A.power(A.add_many(A.exp(b) for b in x), -1) for a in x]
    shifted = [A.exp(a - m) / A.add_many(A.exp(b - m) for b in x) for a in x]
    assert naive == shifted


def test_online_softmax_recurrence():
    x = xs(6)
    m1 = A.maximum(x[:3])
    l1 = A.add_many(A.exp(a - m1) for a in x[:3])
    m2 = A.maximum([m1, A.maximum(x[3:])])
    l2 = l1 * A.exp(m1 - m2) + A.add_many(A.exp(a - m2) for a in x[3:])
    assert m2 == A.maximum(x)
    assert all(A.exp(a - m2) / l2 == A.exp(a) / A.add_many(A.exp(b) for b in x) for a in x)


def test_max_normalization():
    x = xs(3)
    assert A.maximum([A.scale(a, Fraction(1, 8)) for a in x]) == A.scale(A.maximum(x), Fraction(1, 8))
    assert A.maximum([a + 1 for a in x]) == A.maximum(x) + 1
    assert A.maximum([x[0], A.const(-1e30)]) == x[0]  # -inf-like init values are dropped


def test_sqrt_rsqrt_reciprocal_share_a_base():
    v = A.add_many(a * a for a in xs(4)) + A.const(Fraction(1, 1000))
    assert A.power(v, Fraction(-1, 2)) * A.power(v, Fraction(1, 2)) == A.ONE
    assert A.power(v, Fraction(-1, 2)) * A.power(v, Fraction(-1, 2)) == A.power(v, -1)
    assert A.power(A.scale(v, 4), Fraction(-1, 2)) == A.scale(A.power(v, Fraction(-1, 2)), Fraction(1, 2))


def test_sigmoid_forms():
    x = xs(1)[0]
    assert A.sigmoid(x) == A.exp(x) / (A.exp(x) + 1)
    assert A.tanh(x) == A.scale(A.sigmoid(A.scale(x, 2)), 2) - 1


def test_variance_forms():
    x = xs(4)
    mean = A.scale(A.add_many(x), Fraction(1, 4))
    v1 = A.scale(A.add_many((a - mean) * (a - mean) for a in x), Fraction(1, 4))
    v2 = A.scale(A.add_many(a * a for a in x), Fraction(1, 4)) - mean * mean
    assert v1 == v2


def test_masked_exp_is_zero():
    x = xs(1)[0]
    assert A.exp(x - 9984) == A.ZERO
    assert any("exp" in a for a in A.ASSUMPTIONS)


def test_z3_proves_piecewise_identities():
    a, b = xs(2)
    assert z3check.prove_equal(A.maximum([a, b]) + A.minimum([a, b]), a + b)[0] == "proved"
    assert z3check.prove_equal(A.relu(a), (a + A.absolute(a)) * Fraction(1, 2))[0] == "proved"
    status, model = z3check.prove_equal(A.relu(a), (A.absolute(a) - a) * Fraction(1, 2))
    assert status == "refuted" and model


def test_z3_unknown_with_uninterpreted_functions_is_not_refuted():
    a = xs(1)[0]
    status, _ = z3check.prove_equal(A.fn("gelu", a), A.fn("erf", a))
    assert status in ("unknown", "refuted") and status != "proved"

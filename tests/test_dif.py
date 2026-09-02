"""DIF tests, written against ground truth rather than against the code.

Every case here plants a known defect and asks whether the procedure recovers it.
A test that only asserts the function returns a float would pass on a broken
implementation, which is the failure mode these are built to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspect_invariance.dif import ETS_A, logistic_dif, mantel_haenszel

SEED = 20260902


def simulate(
    n_per_group: int = 600,
    n_items: int = 20,
    dif_item: int | None = 7,
    dif_size: float = 1.0,
    nonuniform: float = 0.0,
    focal_ability_shift: float = 0.0,
    seed: int = SEED,
):
    """One-parameter logistic generation with a planted DIF item.

    ``dif_size`` shifts the focal group's difficulty on ``dif_item`` only, holding
    ability equal, which is the definition of uniform DIF.
    ``focal_ability_shift`` moves the focal group's whole ability distribution,
    which is impact and must NOT be flagged.
    """
    rng = np.random.default_rng(seed)
    b = np.linspace(-1.5, 1.5, n_items)
    blocks, groups = [], []
    for g, shift in ((0, 0.0), (1, focal_ability_shift)):
        theta = rng.normal(shift, 1.0, n_per_group)
        eta = theta[:, None] - b[None, :]
        if g == 1 and dif_item is not None:
            eta[:, dif_item] -= dif_size
            eta[:, dif_item] += nonuniform * theta
        p = 1.0 / (1.0 + np.exp(-eta))
        blocks.append((rng.random(p.shape) < p).astype(int))
        groups.append(np.full(n_per_group, g))
    return np.vstack(blocks), np.concatenate(groups)


def test_mantel_haenszel_recovers_planted_uniform_dif():
    responses, group = simulate(dif_item=7, dif_size=1.0)
    result = mantel_haenszel(responses, group, 7, "q007")
    assert result.flagged, "a 1.0 logit shift should not go unnoticed"
    assert result.classification == "C"
    # Negative delta means the item favours the reference group, which is what
    # making it harder for the focal group should produce.
    assert result.delta < 0


def test_logistic_recovers_planted_uniform_dif_and_names_its_form():
    responses, group = simulate(dif_item=7, dif_size=1.0)
    result = logistic_dif(responses, group, 7, "q007")
    assert result.flagged
    assert result.kind == "uniform"
    assert result.p_uniform < 0.001


def test_no_dif_produces_no_flags():
    responses, group = simulate(dif_item=None)
    flagged = [i for i in range(responses.shape[1])
               if logistic_dif(responses, group, i).flagged]
    assert flagged == [], f"clean data should flag nothing, flagged {flagged}"


def test_impact_is_not_flagged_as_bias():
    """The single most important property.

    The focal group is genuinely less able, so it scores lower on every item.
    That is impact, not bias, and conditioning on the matching variable is
    supposed to remove it entirely. If this test fails the tool would tell a
    regulator that a model is being measured unfairly when it is simply worse.
    """
    responses, group = simulate(dif_item=None, focal_ability_shift=-0.8)
    ref_mean = responses[group == 0].sum(axis=1).mean()
    foc_mean = responses[group == 1].sum(axis=1).mean()
    assert ref_mean - foc_mean > 1.0, "the simulation should produce a real score gap"

    flagged = [i for i in range(responses.shape[1])
               if mantel_haenszel(responses, group, i).flagged]
    assert flagged == [], f"a pure ability difference must flag no items, flagged {flagged}"


def test_mantel_haenszel_is_blind_to_crossing_dif():
    """A documented limitation, asserted so it cannot be forgotten.

    MH conditions on total score and averages the odds ratio over strata, so DIF
    that reverses direction across the ability range cancels. Logistic regression
    is the reason this package runs both.
    """
    responses, group = simulate(dif_item=7, dif_size=0.0, nonuniform=1.2)
    mh = mantel_haenszel(responses, group, 7)
    lr = logistic_dif(responses, group, 7)
    assert mh.classification == ETS_A, "MH is expected to miss crossing DIF"
    assert lr.p_nonuniform < 0.01, "logistic regression must detect it"


def test_purification_changes_the_matching_variable_without_losing_sensitivity():
    """Purification excludes the studied item from its own matching score.

    The original version of this test asserted that purifying always increases
    the absolute delta. That is not a theoretical guarantee: purification removes
    the item's contamination of the variable it is conditioned on, and on any one
    sample the estimate can move either way. What IS guaranteed is that the
    matching variable differs and that sensitivity to real DIF survives.
    """
    responses, group = simulate()
    purified = mantel_haenszel(responses, group, 7, purify=True)
    unpurified = mantel_haenszel(responses, group, 7, purify=False)

    assert purified.delta != unpurified.delta, "purification must change the estimate"
    assert purified.flagged and unpurified.flagged, "both must still see real DIF"
    assert purified.delta < 0 and unpurified.delta < 0, "both must agree on direction"


def test_purification_does_not_add_false_positives_on_clean_data():
    responses, group = simulate(dif_item=None)
    purified = sum(mantel_haenszel(responses, group, i, purify=True).flagged
                   for i in range(responses.shape[1]))
    assert purified == 0


def test_constant_item_is_handled_not_crashed():
    responses, group = simulate()
    responses[:, 3] = 1
    lr = logistic_dif(responses, group, 3)
    assert lr.classification == ETS_A
    assert "constant" in lr.note


@pytest.mark.parametrize(
    "bad_group, message",
    [
        (np.zeros(1200, dtype=int), "reference"),
        (np.full(1200, 2), "only 0 and 1"),
    ],
)
def test_invalid_groups_are_rejected(bad_group, message):
    responses, _ = simulate()
    with pytest.raises(ValueError, match=message):
        mantel_haenszel(responses, bad_group, 0)


def test_non_binary_responses_are_rejected():
    responses, group = simulate()
    responses = responses.astype(float)
    responses[0, 0] = 0.5
    with pytest.raises(ValueError, match="0/1 scored"):
        mantel_haenszel(responses, group, 0)

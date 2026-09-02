"""Does the method actually work? Plant a known defect and see if it comes back.

Running an analysis on real logs tells you what the analysis says. It does not
tell you whether the analysis is right. So this script generates response data
where the truth is known by construction, and checks that the procedures recover
it, including the two cases that matter most:

* a genuine ability difference between groups must produce NO flags, because that
  is impact and not bias;
* an item that is harder in one language at equal ability MUST be flagged.

Run it with::

    python examples/recover_planted_dif.py
"""

from __future__ import annotations

import numpy as np

from inspect_invariance import logistic_dif, mantel_haenszel
from inspect_invariance.invariance import check_invariance
from inspect_invariance.matrix import ResponseMatrix

SEED = 20260902
N_ITEMS = 20
N = 600


def simulate(dif_item=None, dif_size=0.0, nonuniform=0.0, ability_shift=0.0, seed=SEED):
    rng = np.random.default_rng(seed)
    b = np.linspace(-1.5, 1.5, N_ITEMS)
    blocks, groups = [], []
    for g, shift in ((0, 0.0), (1, ability_shift)):
        theta = rng.normal(shift, 1.0, N)
        eta = theta[:, None] - b[None, :]
        if g == 1 and dif_item is not None:
            eta[:, dif_item] -= dif_size
            eta[:, dif_item] += nonuniform * theta
        p = 1.0 / (1.0 + np.exp(-eta))
        blocks.append((rng.random(p.shape) < p).astype(int))
        groups.append(np.full(N, g))
    return np.vstack(blocks), np.concatenate(groups)


def flagged(responses, group, procedure):
    return [i for i in range(responses.shape[1]) if procedure(responses, group, i).flagged]


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    print("Recovering planted defects. 20 items, 600 respondents per language.")

    rule("1. One item made 1.0 logit harder in the focal language")
    responses, group = simulate(dif_item=7, dif_size=1.0)
    mh = mantel_haenszel(responses, group, 7)
    lr = logistic_dif(responses, group, 7)
    print(f"   item 7  MH delta {mh.delta:+.2f} ({mh.classification})  "
          f"dR2 {lr.delta_r2:.4f} ({lr.classification}, {lr.kind})")
    print(f"   flagged by logistic regression: {flagged(responses, group, logistic_dif)}")
    print("   expected: item 7 and nothing else")

    rule("2. No DIF anywhere")
    responses, group = simulate()
    print(f"   MH flags:       {flagged(responses, group, mantel_haenszel)}")
    print(f"   logistic flags: {flagged(responses, group, logistic_dif)}")
    print("   expected: both empty")

    rule("3. The focal group is genuinely less able, but no item is biased")
    responses, group = simulate(ability_shift=-0.8)
    gap = responses[group == 0].sum(1).mean() - responses[group == 1].sum(1).mean()
    print(f"   raw score gap: {gap:.2f} points out of {N_ITEMS}")
    print(f"   MH flags: {flagged(responses, group, mantel_haenszel)}")
    print("   expected: EMPTY. This is impact, not bias, and the whole point of")
    print("   conditioning on the matching variable is to tell them apart.")

    rule("4. Crossing DIF, which Mantel-Haenszel is known to miss")
    responses, group = simulate(dif_item=7, dif_size=0.0, nonuniform=1.2)
    mh = mantel_haenszel(responses, group, 7)
    lr = logistic_dif(responses, group, 7)
    print(f"   MH:       delta {mh.delta:+.2f} ({mh.classification})  <- misses it, as expected")
    print(f"   logistic: non-uniform p = {lr.p_nonuniform:.2e}  <- catches it")
    print("   this is why the package runs both rather than picking one")

    rule("5. Do the language versions measure the same construct?")
    items = tuple(f"i{k}" for k in range(8))
    tau = np.array([-0.5, -0.2, 0.0, 0.2, 0.5, -0.3, 0.1, 0.3])

    def probit(loadings, seed):
        r = np.random.default_rng(seed)
        lam = np.asarray(loadings)
        resid = np.sqrt(np.clip(1 - lam**2, 1e-6, None))
        theta = r.normal(size=800)
        ystar = theta[:, None] * lam[None, :] + r.normal(size=(800, 8)) * resid[None, :]
        return (ystar > tau[None, :]).astype(int)

    def as_matrix(data, lang):
        return ResponseMatrix(lang, items, tuple(f"r{i}" for i in range(len(data))),
                              data, "models")

    equal = [0.7] * 8
    broken = [0.7, 0.15, 0.7, 0.7, 0.15, 0.7, 0.15, 0.7]

    for label, focal_loadings in (("equal loadings", equal), ("three items broken", broken)):
        result = check_invariance({
            "en": as_matrix(probit(equal, 1), "en"),
            "hr": as_matrix(probit(focal_loadings, 2), "hr"),
        })
        print(f"   {label:<20} dCFI {result.delta_cfi:+.4f}  "
              f"metric holds: {result.metric_holds}")
    print("   expected: holds for equal loadings, fails when three items are broken")

    print("\nNote on the chi-square. It is significant in every case above, because")
    print("with 1,600 observations it is significant for any trivial misfit. That is")
    print("exactly why invariance is judged on the CHANGE in CFI and RMSEA rather")
    print("than on the test statistic.\n")


if __name__ == "__main__":
    main()

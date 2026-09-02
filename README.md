# inspect-invariance

Measurement invariance and differential item functioning for multilingual LLM
benchmarks, as an [Inspect](https://inspect.aisi.org.uk/) extension.

Translating a benchmark does not preserve what it measures. This package tests
whether it did, and names the items that broke.

## The problem

A benchmark is translated into six languages, run against a model, and the model
scores nine points lower in Croatian than in English. The obvious reading is that
the model is worse in Croatian. The other reading is that the Croatian form is
harder, or measures something slightly different, and the model is fine.

Nothing in the score distinguishes those. They are different claims with
different consequences, and only one of them is about the model.

This is not a hypothetical worry imported from another field. Cross-national
assessment has been dealing with it for forty years, because it is the failure
mode that shows up whenever an instrument crosses a language boundary, and it
shows up often. Translated items shift in difficulty when an idiom has no
equivalent, when a distractor becomes implausible, when a quantifier is rendered
loosely, or when a term is more familiar in one culture than another. The
published analysis of PISA 2009 found that a widely used international benchmark
was not fully comparable across the countries that were nonetheless being ranked
on it.

The regulatory version of the problem is sharper. From August 2026 the EU AI
Office can act on evidence of harmful manipulation, and the AI Act leaves the
evaluation method to the state of the art. A multilingual finding that rests on
an uninterrogated translation is not state of the art. It is a finding about the
instrument that has been mistaken for a finding about the model, and it will not
survive the first competent challenge.

## What this does

Two analyses, run together, answering two different questions.

**Do the language versions measure the same construct at all?** A multi-group
one-factor model is fitted to the tetrachoric correlation matrix of each language
version, first with all loadings free (configural), then with loadings held equal
across languages (metric). If the configural model does not fit, the forms are
not measuring a common dimension and no comparison between them is defined. If
metric invariance fails, the forms share a dimension but not a scale, so
differences between them are not on a common metric.

**Which individual items behave differently?** Every item is tested for
differential item functioning by two procedures: Mantel-Haenszel with the ETS
delta scale and the A/B/C classification used for operational item review, and
logistic regression with the Nagelkerke effect size and the Jodoin and Gierl
classification. Both condition on total score, so an item is flagged when one
language group gets it wrong *relative to their overall performance*, not merely
when they get it wrong.

That conditioning is the part that matters, and it is what separates this from
comparing accuracy by language. A model that is genuinely weaker in Croatian will
score lower on every Croatian item. That is impact, and it is a real property of
the model. DIF is what remains after impact is removed, and it is a property of
the instrument.

Both procedures run because they fail differently. Mantel-Haenszel is the more
familiar and has the better-understood effect-size scale, but it averages the
odds ratio across score strata, so DIF that reverses direction across the ability
range cancels out and it sees nothing. Logistic regression catches that case
through the interaction term. Running one and calling it a DIF analysis would be
a choice about which failures to be blind to.

## Does it work?

`examples/recover_planted_dif.py` generates data where the truth is known by
construction and checks that the procedures recover it. Output, abridged:

```
1. One item made 1.0 logit harder in the focal language
   item 7  MH delta -1.62 (C)  dR2 0.0427 (B, uniform)
   flagged by logistic regression: [7]

2. No DIF anywhere
   MH flags: []   logistic flags: []

3. The focal group is genuinely less able, but no item is biased
   raw score gap: 2.68 points out of 20
   MH flags: []

4. Crossing DIF, which Mantel-Haenszel is known to miss
   MH:       delta +0.34 (A)  <- misses it, as expected
   logistic: non-uniform p = 1.38e-10  <- catches it

5. Do the language versions measure the same construct?
   equal loadings       dCFI -0.0005  metric holds: True
   three items broken   dCFI -0.1520  metric holds: False
```

Case 3 is the one to look at. The focal group is two and a half points worse
across a twenty-item form, and not a single item is flagged, because the
difference is in the respondents rather than in the items. A tool that flagged
items there would be telling a regulator that a benchmark is broken every time a
model is simply worse in a language.

The demo benchmark in `data/demo/` carries a positive control for the same
reason: item `q007` has a deliberately degraded Croatian stem, which asks what
*can* be true where the English asks what *must* be true. A detector that stays
silent on a clean benchmark and a detector that is broken look identical until
you give it something to find.

## Using it

```bash
pip install -e ".[eval]"
```

Run the benchmark in each language. Each sample carries `item_id` and `language`
in its metadata, which is what links an item to its translations:

```bash
inspect eval src/inspect_invariance/task.py --model openai/gpt-4o --epochs 20
```

Then analyse the logs:

```bash
inspect-invariance analyse ./logs --reference en
```

Or from Python, on scores from anywhere:

```python
from inspect_invariance import from_records, analyse, render

matrices = from_records(records, design="models")
print(render(analyse(matrices, reference="en")))
```

### What counts as a respondent

Classical test theory wants a matrix of respondents by items, and a model is not
obviously a respondent. Two designs are supported and the choice changes what you
can conclude, so it is worth making deliberately.

With `design="epochs"`, one model runs the benchmark repeatedly and each run is a
respondent. This is cheap, and adequate for catching gross translation failures,
but the runs are not independent in the way the statistics assume and the ability
range is narrow.

With `design="models"`, a panel of models each contributes one response vector.
The variation being conditioned on is then genuine variation in capability, which
is much closer to what the procedures were built for. Prefer this where you can
afford it. A dozen models spanning a real capability range give the matching
variable something to match on.

## Limits

Stated here rather than discovered later.

The invariance half is a screening procedure. It fits by maximum likelihood to
the tetrachoric correlation matrix, where a full treatment would estimate
thresholds and loadings jointly by WLSMV with a mean-and-variance-adjusted test
statistic, as lavaan or Mplus would. That is enough to decide whether a benchmark
deserves a closer look. It is not enough to be the final word in a report to a
regulator, and it should not be cited as though it were.

The sequence stops at metric invariance. Scalar invariance, which is what
licenses comparing means, requires equal item intercepts, and for binary items
the intercept is the threshold. A threshold difference at equal ability is
precisely what uniform DIF is, so the scalar question is answered item by item by
the DIF half instead, on a scale that item review can act on. A single scalar
chi-square would be less informative than a list of items, not more.

Items must be binary. Polytomous scoring needs the ordinal generalisations
(Mantel or the standardised Liu-Agresti estimator) and those are not implemented.

DIF requires complete data. A respondent who did not answer every item is
refused rather than imputed, because imputing would invent the observations the
whole analysis is conditioned on.

Finally, the procedures find items that *behave* differently. They do not explain
why, and they cannot tell a mistranslation from a genuine cultural difference in
familiarity. That judgement needs someone who reads the language, which is the
step this tooling exists to direct rather than to replace.

## References

Holland, P. W., and Thayer, D. T. (1988). Differential item performance and the
Mantel-Haenszel procedure. In H. Wainer and H. I. Braun (eds.), *Test Validity*.

Swaminathan, H., and Rogers, H. J. (1990). Detecting differential item
functioning using logistic regression procedures. *Journal of Educational
Measurement*, 27(4).

Cheung, G. W., and Rensvold, R. B. (2002). Evaluating goodness-of-fit indexes for
testing measurement invariance. *Structural Equation Modeling*, 9(2).

Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates
using an effect size measure with the logistic regression procedure for DIF
detection. *Applied Measurement in Education*, 14(4).

Chen, F. F. (2007). Sensitivity of goodness of fit indexes to lack of measurement
invariance. *Structural Equation Modeling*, 14(3).

Kankaras, M., and Moors, G. (2014). Analysis of cross-cultural comparability of
PISA 2009 scores. *Journal of Cross-Cultural Psychology*, 45(3), 381-399.

## Licence

MIT.

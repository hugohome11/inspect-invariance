# inspect-invariance

Measurement invariance and differential item functioning for multilingual LLM
benchmarks, as an [Inspect](https://inspect.aisi.org.uk/) extension.

Translating a benchmark does not preserve what it measures. This package tests
whether it did, and names the items that broke.

## The evidence behind it

This implements the method of a published measurement audit of a real
multilingual suite: the same 22 models answering the same 3,080 items in eleven
languages of Africa, from the HELM archive.

> Kankaras, M. (2026). *Ranks without resolution: how much of a multilingual
> benchmark's language ordering is estimable?*
> [doi:10.5281/zenodo.22128037](https://doi.org/10.5281/zenodo.22128037)

Three findings from that audit, which are what the tooling here is for:

- On the Winogrande half, **only 18 of the 55 language pairs are distinguishable**
  at all. Ranks two through eleven are one undifferentiated block, and the rank
  intervals reach eight places wide. On the medical half, four of ten adjacent
  gaps in the published ordering separate.
- **19.4 per cent of the medical and 48.4 per cent of the Winogrande reported
  scores** have a confidence interval that includes the chance level of the item
  format. They are published numbers that carry no information about the model.
- Cross-language differential item functioning affects **14.1 per cent of medical
  items** after equating scale as well as location, against essentially none in a
  permutation null.

The deposit behind that DOI reproduces every number in the paper. Note that the
paper's own estimation is separate code and is not this package; the package
implements the item-level half of the method for reuse on other benchmarks.

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
pip install -e .
```

You also need the client for whichever model you intend to run. `inspect_ai`
treats these as optional, so installing the framework alone leaves `inspect eval`
failing with a request for one:

```bash
pip install -e ".[openai]"     # or [anthropic], [google], [mistral], [groq]
```

Run the benchmark in each language. Each sample carries `item_id` and `language`
in its metadata, which is what links an item to its translations:

```bash
inspect eval src/inspect_invariance/task.py --model openai/gpt-4o --epochs 20
```

### On a real multilingual benchmark

`afrimmlu.py` runs AfriMMLU, a parallel translation of a 500-item MMLU subset
into sixteen African languages plus English and French, pinned to an immutable
dataset revision. One task covers every language, so a single run produces a log
the analysis can read directly:

```bash
inspect eval src/inspect_invariance/afrimmlu.py@afrimmlu --model openai/gpt-5-nano
inspect-invariance analyse ./logs --reference eng
```

The default is all eighteen language versions of the 500-item test split, which
is 9,000 samples.

**Avoid a reasoning model here.** Measured on the full 9,000 samples: a
non-reasoning model spent 1.39M input and 57k output tokens, while a reasoning
model of the same family spent 1.38M input and 6.39M output, of which 6.02M were
reasoning tokens. That is about 110 times the output volume for the same task and
a 10x longer wall clock, and it buys little here, because the items are
four-option multiple choice. Reasoning happens regardless of the `cot` setting, so
the task cannot switch it off. Pick a small non-reasoning model.

Narrow the run further if you want it cheaper still:

```bash
inspect eval src/inspect_invariance/afrimmlu.py@afrimmlu   -T languages=eng,swa,yor,zul -T split=val --model openai/gpt-5-nano
```

The item sets are checked for parallelism on load. If a language version's answer
key, subject or option count departs from the reference, it raises rather than
proceeding, because the linkage every analysis here depends on would be an
illusion.

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

The invariance half is fitted by **diagonally weighted least squares** with a
mean-and-variance adjusted test statistic, which is the WLSMV estimator lavaan and
Mplus use for ordinal indicators. Each residual correlation is weighted by the
reciprocal of its own sampling variance, computed in closed form from the delta
method, and the statistic is then corrected using the full sampling covariance so
that it is distributed as its degrees of freedom claim.

That matters because the previous implementation used a normal-theory
maximum-likelihood fit, which treats a tetrachoric correlation matrix as though it
were a sample covariance matrix. It is not one, and the resulting statistic ran
four- to sevenfold inflated and rejected correctly specified models across the
whole range of item counts and sample sizes a real benchmark occupies. Measured on
data simulated from exactly the fitted model, with no differential functioning
anywhere, the current estimator rejects a correct model on **0 of 30 replications
at 100, 200, 400 and 800 respondents**, and still detects a real loading
difference on 26 of 30 at 200 and 30 of 30 at 400. The full account, with the
before-and-after numbers, is in the description of
[pull request #1](https://github.com/hugohome11/inspect-invariance/pull/1).

Two limits on it, both stated in the code. Metric invariance is decided by the
scaled difference test rather than by a change in CFI or RMSEA: those cutoffs
assume nested models share their degrees of freedom, which a mean-and-variance
adjusted statistic does not, and on correct models the delta-RMSEA rule rejected 8
of 30 where the difference test rejected 0. And the corrected statistic needs the
joint covariance of every residual correlation, which caps the problem size at 60
items and 4,000 stacked language-by-correlation residuals. About 20 items across
18 languages fits, as does 60 items across 2. Above that it refuses and says so,
which means invariance should be tested within a subscale or a subject rather than
across a whole item bank. That is the right unit anyway: a single common factor
over hundreds of heterogeneous items is not a model worth fitting.

**The item-level DIF half is unaffected and is the part to act on.** It controls
false discovery across items by Benjamini-Hochberg, refuses to report an effect
size where the logistic fit separates completely or has too few events per
parameter, and distinguishes "no differential functioning found" from "nothing
was testable".

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

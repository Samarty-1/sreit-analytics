# Singapore REIT Analytics

Fundamental analysis of the S-REIT sector — distribution yields, yield spreads
over bonds, empirical duration, and distribution sustainability — plus an
honest test of whether the sector's standard valuation signal actually
predicts returns.

It doesn't. The details of *why* are the interesting part.

## Why S-REITs

Singapore is one of Asia's largest REIT markets, and the sector behaves
nothing like the US equities most retail quant work is built on. S-REITs are
yield-driven, structurally levered against a regulatory gearing cap, and
unusually sensitive to interest rates. None of the metrics that matter here —
distribution yield, yield spread, gearing, NAV premium — are technical
indicators, which makes it a genuinely different modelling problem from
moving averages on price.

22 S-REITs across 7 sub-sectors, 6 years of daily data.

## Finding 1: rate sensitivity is not uniform, and the gradient makes sense

"REITs are rate-sensitive" is the sector's most repeated claim and almost
never a number. Regressing each REIT's daily return on the SGD bond fund's
return, controlling for the Straits Times Index:

| Sub-sector | n | Mean rate beta | Mean market beta |
|---|---|---|---|
| Data Centre | 1 | **0.666** | 0.523 |
| Industrial | 5 | **0.570** | 0.590 |
| Diversified | 3 | 0.496 | 0.689 |
| Healthcare | 1 | 0.398 | 0.336 |
| Office | 6 | 0.339 | 0.748 |
| Retail | 4 | 0.306 | 0.561 |
| Hospitality | 2 | **0.242** | 0.630 |

The spread is more than 2:1 between industrial/data-centre REITs and
hospitality, and the strongest betas carry t-statistics of 6–9, so this is not
noise. The economics line up: industrial and data-centre REITs have long
leases with fixed escalations — contractually bond-like cash flows, which
discount like bonds. Hospitality has no leases at all in the usual sense;
room rates reprice nightly and rise with inflation, so the cash flows are far
less bond-like and the units trade more on occupancy than on rates.

The practical read: **into a rate move, the sub-sector matters more than the
name.** Industrial and data-centre S-REITs are where duration risk
concentrates; hospitality is closer to an operating business that happens to
own hotels.

Controlling for the equity market is what makes this meaningful. Without the
market term, `rate_beta` absorbs the general risk-on/risk-off correlation
between bonds and equities and overstates true rate sensitivity across the
board.

## Finding 2: the standard valuation signal fails out of sample

The sector's standard framework says a REIT whose distribution yield spread
over bonds is unusually wide is cheap and should outperform. Entirely
plausible — so it was tested, with the data split by date into a selection
half and a confirmation half up front.

**Selection set (2021-08 to 2024-01), choosing the horizon here and only here:**

| Horizon | Mean IC | t-stat (HAC) | naive t | lag-1 autocorr | independent draws | Significant |
|---|---|---|---|---|---|---|
| **21d** | 0.071 | **1.87** | 1.87 | −0.16 | 30 | no |
| 63d | 0.084 | 1.75 | 2.07 | +0.36 | 10 | no |
| 126d | 0.081 | 1.57 | 1.98 | +0.46 | 5 | no |

**Confirmation set (2024-02 to 2026-08), looked at once, horizon fixed above:**

| Horizon | Mean IC | t-stat (HAC) | Significant |
|---|---|---|---|
| 21d | **−0.015** | −0.38 | no |

**The sign flipped.** The signal that scored +0.07 on the selection half
scored −0.015 on data it had never touched. Three horizons were tested and
none of them survived confirmation.

> **Correction (2026-08-30).** This table used to report a **significant**
> 126-day result — mean IC 0.113, t = 2.37 — and to describe it as "what a real
> finding looks like before it dies". That t-statistic was wrong, and it was
> the only significant result in the study.
>
> Observation dates are 21 trading days apart while the horizon is 126 days, so
> **six consecutive observations share most of their return window**. The
> lag-1 autocorrelation of the IC series is +0.46, which is the overlap
> signature. The plain `mean / (sd / √n)` treats 24 overlapping observations as
> 24 independent draws; they are worth about **four**. The Newey-West
> statistic on the original split was **1.83**, not 2.37 — never significant.
>
> Two smaller defects came out with it. The 252-day rolling z-score warmup left
> the first 12 sampled dates empty, and the split ran *before* they were
> dropped — so the "selection half" was 24 usable dates against the
> confirmation half's 30, over a window a year longer than the one actually
> used. And the horizon was chosen by highest mean IC, which rises mechanically
> with horizon because longer windows are smoother. It is now chosen on the
> corrected t-statistic, the halves are 30/30 on usable dates, and the
> quantile-spread Sharpe annualises on independent periods rather than on
> sampling frequency.
>
> **The conclusion did not change** — the signal still fails confirmation, and
> the sign still flips. What changed is that the study no longer offers a
> significant result it was not entitled to claim. Both statistics are now
> reported side by side (`ic_tstat` and `ic_tstat_naive`) so the gap stays
> visible.

### Why it fails — look at the top of the ranking

The cheapness ranking is not sorted by value. Its top three:

| | Yield | Spread z-score | 3y distribution growth | 6y total return |
|---|---|---|---|---|
| Prime US REIT | 6.6% | +4.32 | **−42.6%** | **−59.3%** |
| KORE US REIT | 3.8% | +3.85 | **−58.2%** | **−63.0%** |
| CapitaLand India Trust | 12.2% | +3.22 | +13.5% | +8.7% |

The two widest spreads in the universe belong to US-office S-REITs that have
cut their distributions by 43% and 58% and lost roughly 60% of their value.
Their spreads are wide because the market correctly expects further cuts, not
because they are mispriced.

That is the mechanism behind the failure. A **trailing** yield divided by a
**falling** price mechanically produces a high yield right up until the
distribution is cut, so a naive spread screen systematically buys value traps.
The distribution-growth column is what separates the two cases, and it is not
in the signal — which points at the obvious next version rather than at
abandoning the framework.

## Method notes

**Trailing yields, never forward.** `distribution_yield` sums only
distributions with an ex-date on or before each observation date. A forward
yield embeds an analyst's estimate of a future payment; using one in a
historical study imports a forecast that did not exist at the time.

**Rolling z-scores, never full-sample.** Whether a spread is wide is a
question about its own history, not its level: 5.5% is generous against a 2%
risk-free rate and thin against 5%. A full-sample z-score is the most common
way a valuation signal becomes accidentally clairvoyant, so standardisation
uses a trailing window only.

**Cross-sectional, not time-series.** REITs are ranked against each other on
each date, which nets out whatever the whole sector did — which is dominated
by rates, and would otherwise be most of the measured "signal".

**Corrected sub-sector labels.** Yahoo Finance classifies **Keppel DC REIT as
"REIT – Office"**; it is a pure data-centre REIT whose lease structure and
tenant base have nothing in common with an office landlord. It also files
CapitaLand Integrated Commercial Trust as pure Retail when it is roughly half
office. Sub-sector is the axis every comparison here cuts along, so the
vendor's errors would have corrupted Finding 1 silently. Labels are curated in
`src/universe.py` with the disagreements noted.

## The main data limitation, stated plainly

The correct risk-free anchor is the **MAS SGS 10-year yield**. MAS's API was
unreliable from this environment — repeated 404s on the documented
bonds-and-bills path, timeouts and truncated responses on the datastore path.
`rates.fetch_sgs_10y` implements it and is tried first; the fallback is
automatic and *reported in the analysis output*, not silently swapped in.

Two proxies are used, for two different jobs:

- **Yield levels: the US 10-year (^TNX).** SGS yields track Treasuries closely
  because the SGD is managed against a basket rather than an independent
  policy rate, so domestic long rates import most of their variation from US
  rates. The *level* differs — SGS 10y has generally traded modestly below UST
  10y — so absolute spreads here are biased by that gap. This is exactly why
  every signal is built on a **z-score of the spread**, in which a roughly
  constant level offset largely cancels. Read the rankings and z-scores; treat
  the absolute `yield_spread_pp` column as indicative.
- **Rate changes: the ABF Singapore Bond Index Fund (A35.SI).** Genuinely
  Singaporean, and the rate-sensitivity regressions need changes rather than
  levels, so no proxy compromise there.

## Usage

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python analysis.py --period 6y
pytest tests/ -q                 # 19 tests
```

```
src/universe.py          the 22 S-REITs, curated sub-sectors
src/metrics.py           distribution yield, yield spread, gearing, rolling z-score
src/rates.py             SG rate inputs and the proxy accounting
src/rate_sensitivity.py  empirical duration regressions
src/signal_test.py       IC, quantile spreads, selection/confirmation split
analysis.py              the full run
```

Tests focus on the look-ahead controls, because those are what a signal study
gets silently wrong: a distribution paid tomorrow must not appear in today's
trailing yield, a z-score must be NaN before its window fills, and a forward
return must never overlap its own signal date. The rate-beta regression is
checked against a constructed series with a known beta of 1.5, and against
unrelated series where it must find nothing.

## What I'd do next

Add distribution sustainability to the signal rather than dropping the
framework. The failure is specific and diagnosable — wide spreads on REITs
about to cut — so gating the cheapness rank on non-negative distribution
growth, gearing headroom against the MAS cap, and interest-coverage is the
obvious next version. That is a new hypothesis and would need its own
confirmation set; the one used here has been spent.

Worth stating: three horizons were tested. None crossed significance once the
overlap correction was applied, but the search is still a search, and the
deflated Sharpe machinery in
[backtest-overfit-audit](https://github.com/Samarty-1/backtest-overfit-audit)
is the right tool to put a number on what it cost.

Also worth stating, because it cuts the other way: the universe is the 22
S-REITs listed **today**, so it is survivorship-biased. The names missing from
it — Eagle Hospitality Trust among them — are disproportionately the ones that
traded at very wide yield spreads and then collapsed, which is to say
precisely the names this signal would have called cheapest. The bias flatters
the yield-spread signal, and it fails confirmation anyway.

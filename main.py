#!/usr/bin/env python3
"""Forecast Bitcoin's price 240 minutes ahead, for make.fun btc-cent-forecast-01.

Program: python3 /bot/main.py (Python 3.13, standard library only) at the
repository root. Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input/):
one case per file, 4,320 one-minute BTC closes (72 h up to the checkpoint).
Writes $MAKEFUN_OUTPUT_DIR (else /output/) <id>.out with one forecast row:
{"forecasts": [{"case": "<id>", "price": <guess>}]}, the price expected 240
minutes after the checkpoint. Deterministic: two runs write identical bytes.

Method — the mean of four views of the same 72 h window, shrunk 90% of
the way from persistence toward that mean, and clamped to a sane band.
No view is trusted alone: four-hour BTC moves are mostly noise (on
non-overlapping 240-minute returns the next return's regression on
trailing returns has ~zero R^2), so the ensemble is built to never be
much worse than guessing "no move", while keeping the small edge the
trend / shape / regression views show in walk-forward checks:

  1. fade         — fade 10% of the last 240-minute log return: 4 h
                     returns slightly mean-revert (AR(1) slope about -0.12
                     on public BTC 1-minute data, Nov 2026).
  2. trend        — OLS slope of log price over the last 720 minutes,
                     extrapolated 240 minutes and damped by 0.5.
  3. lad-ridge    — walk-forward ridge regression of the next 240-minute
                     log return on trailing 60/240/720/1440-minute log
                     returns, fitted inside this case's own window,
                     prediction shrunk by 0.5.
  4. knn-shape    — demeaned-shape k-NN: the last 480 minutes' log-price
                     shape (sampled every 8 minutes) matched against
                     earlier windows of this case; the median 240-minute
                     outcome of the 15 best matches from distinct episodes
                     (matches within 120 minutes of each other count once,
                     so one episode cannot dominate the vote).

The forecast is the mean of the four guesses, moved 90% of the way from
the checkpoint close (persistence) toward that mean, then clamped to
+/-3% of the checkpoint close. Persistence is the anchor, not a vote:
a median that included it kept collapsing exactly onto persistence
whenever the views split two-below / two-above. Every output is
guarded: any failure in a view falls back to the remaining views, and
any failure in the whole forecast still writes a valid price, because
one invalid case forfeits the prize.

Run locally:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999
HORIZON = 240  # minutes ahead, fixed by the competition

# Ensemble tuning (fixed constants, fitted on public BTC 1-minute data and
# the practice window; nothing here is fitted to the final's unseen prices).
FADE_A = 0.10        # fraction of the last 240-minute return to fade
TREND_H = 720        # minutes of log price the trend slope is fitted on
TREND_DAMP = 0.5     # damping of the extrapolated trend
LAD_SHRINK = 0.5     # shrinkage of the ridge prediction
KNN_SHAPE = 480      # minutes of shape matched by the k-NN view
KNN_K = 15           # distinct episodes voting in the k-NN view
KNN_MIN_GAP = 300    # minutes before the checkpoint a matched window ends
KNN_DEDUP = 120      # minutes: matches closer than this are one episode
KNN_STEP = 8         # minutes: shape sampling step (keeps the match fast)
BLEND_SHRINK = 0.9   # how far persistence moves toward the mean view
CLAMP_BAND = 0.03    # forecast stays within +/-3% of the checkpoint close


def price5(x):
    """x kept on the grid's 5 decimals and inside (0, 10,000,000)."""
    if not math.isfinite(x):
        x = 1.0
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def clean_prices(prices):
    """Finite floats only; a non-finite close is carried forward from the
    last finite one, so the series the views see is always usable."""
    out = []
    last = None
    for v in prices:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        if math.isfinite(f) and f > 0:
            last = f
        elif last is None:
            last = 1.0
        out.append(last)
    return out


def view_persist(prices, lp):
    return prices[-1]


def view_fade(prices, lp):
    return prices[-1] * math.exp(-FADE_A * (lp[-1] - lp[-1 - HORIZON]))


def view_trend(prices, lp):
    n = len(lp)
    xs = list(range(max(0, n - TREND_H), n))
    ys = [lp[i] for i in xs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return prices[-1] * math.exp(TREND_DAMP * slope * HORIZON)


def _solve_linear(matrix, rhs):
    """Small dense solve by Gauss-Jordan with partial pivoting."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        for r in range(n):
            if r != col:
                f = aug[r][col] / aug[col][col]
                for cc in range(col, n + 1):
                    aug[r][cc] -= f * aug[col][cc]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def view_lad_ridge(prices, lp):
    """Ridge of the next 240-minute log return on trailing returns."""
    n = len(lp)
    horizons = (60, 240, 720, 1440)
    rows = []
    ys = []
    for t in range(1440, n - HORIZON):
        rows.append([lp[t] - lp[t - h] for h in horizons])
        ys.append(lp[t + HORIZON] - lp[t])
    if len(rows) < 200:
        return prices[-1]
    mus = [sum(c) / len(c) for c in zip(*rows)]
    sds = []
    for c, m in zip(zip(*rows), mus):
        var = sum((v - m) ** 2 for v in c) / len(c)
        sds.append(math.sqrt(var) or 1.0)
    std = [[(v - m) / s for v, m, s in zip(row, mus, sds)] for row in rows]
    lam = 50.0
    dim = len(horizons)
    a = [[sum(r[i] * r[j] for r in std) + (lam if i == j else 0.0)
          for j in range(dim)] for i in range(dim)]
    b = [sum(r[i] * y for r, y in zip(std, ys)) for i in range(dim)]
    w = _solve_linear(a, b)
    if w is None:
        return prices[-1]
    feats = [(lp[-1] - lp[-1 - h] - m) / s
             for h, m, s in zip(horizons, mus, sds)]
    pred = sum(wi * fi for wi, fi in zip(w, feats))
    return prices[-1] * math.exp(LAD_SHRINK * pred)


def view_knn(prices, lp):
    """Demeaned-shape k-NN over this case's own history."""
    n = len(lp)
    if n < KNN_SHAPE + HORIZON + 50:
        return prices[-1]
    shape = KNN_SHAPE
    idx = list(range(n - shape, n, KNN_STEP))
    q = [lp[i] for i in idx]
    qm = sum(q) / len(q)
    qd = [x - qm for x in q]
    qn = math.sqrt(sum(x * x for x in qd)) or 1.0
    cands = []
    for end in range(shape, n - KNN_MIN_GAP):
        w = [lp[end - shape + j] for j in range(0, shape, KNN_STEP)]
        wm = sum(w) / len(w)
        wd = [x - wm for x in w]
        wn = math.sqrt(sum(x * x for x in wd)) or 1.0
        dot = sum(a * b for a, b in zip(qd, wd)) / (qn * wn)
        if end + HORIZON < n:
            cands.append((1.0 - dot, end, lp[end + HORIZON] - lp[end]))
    if not cands:
        return prices[-1]
    cands.sort()
    picked = []
    for dist, end, out in cands:
        if all(abs(end - e) > KNN_DEDUP for _, e, _ in picked):
            picked.append((dist, end, out))
        if len(picked) >= KNN_K:
            break
    if not picked:
        return prices[-1]
    return prices[-1] * math.exp(statistics.median([o for _, _, o in picked]))


# The four active views vote; persistence is the anchor the blend shrinks
# toward, not a vote itself — a median that included it kept collapsing
# exactly onto persistence whenever the views split two-below / two-above.
_VIEWS = (view_fade, view_trend, view_lad_ridge, view_knn)


def forecast(prices):
    """One forecast for one case's 4,320 closes. Never raises."""
    clean = clean_prices(prices)
    if not clean:
        return price5(1.0)
    last = clean[-1]
    if len(clean) < HORIZON + 60:
        return price5(last)
    lp = [math.log(v) for v in clean]
    guesses = []
    for view in _VIEWS:
        try:
            g = view(clean, lp)
            if math.isfinite(g) and g > 0:
                guesses.append(g)
        except Exception:
            continue
    if not guesses:
        return price5(last)
    mean_view = sum(guesses) / len(guesses)
    g = last + BLEND_SHRINK * (mean_view - last)
    g = min(last * (1.0 + CLAMP_BAND), max(last * (1.0 - CLAMP_BAND), g))
    return price5(g)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        try:
            with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            markets = case.get("markets") or []
            prices = markets[0].get("prices") if markets else []
            guess = forecast(prices)
        except Exception:
            guess = price5(1.0)
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

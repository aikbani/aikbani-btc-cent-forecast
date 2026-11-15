# BTC 4-hour-ahead forecaster — make.fun btc-cent-forecast-01

Forecasts Bitcoin's price 240 minutes after each checkpoint, from the
4,320 one-minute closes (72 h) in the case file.

## How it works

`main.py` (Python 3.13, standard library only) reads every `<id>.in` in
`$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `<id>.out` with
`{"forecasts": [{"case": "<id>", "price": <guess>}]}` to
`$MAKEFUN_OUTPUT_DIR` (else `/output/`).

The guess is the mean of four views of the case's own 72 h window,
moved 90% of the way from the checkpoint close toward that mean, then
clamped to ±3% of the checkpoint close:

1. **fade** — fade 10% of the last 240-minute log return (4 h returns
   slightly mean-revert).
2. **trend** — OLS slope of log price over the last 720 minutes,
   extrapolated 240 minutes, damped by 0.5.
3. **lad-ridge** — walk-forward ridge regression of the next 240-minute
   log return on trailing 60/240/720/1440-minute log returns, fitted
   inside the case's own window, prediction shrunk by 0.5.
4. **knn-shape** — demeaned-shape k-NN: the last 480 minutes matched
   against earlier windows of the case; median 240-minute outcome of the
   15 best matches from distinct episodes.

Four-hour BTC moves are mostly noise, so the design goal is to never be
much worse than persistence ("no move") while keeping the small edge the
views show in walk-forward checks. Every output is guarded: a failing
view is skipped, and a failing forecast still writes a valid price.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs write identical bytes. About 0.1 s per case.

## Validation (measured locally, 2026-11-15)

Practice cases p00–p04 (targets are the next case's last price):

| case | guess | target | error |
|------|-------|--------|-------|
| p00 | 74.742,30 | 74.447,80 | $294,50 |
| p01 | 74.386,16 | 73.680,84 | $705,32 |
| p02 | 73.693,56 | 73.210,50 | $483,06 |
| p03 | 73.148,26 | 73.777,35 | $629,09 |
| p04 | 73.625,85 | 73.545,88 | $79,97 |

Practice MAE p00–p04: **$438,39** vs persistence $470,55 (beats the
baseline on 3 of 5 cases; worse on p02/p03).

Out-of-sample: 42 checkpoints on public Binance BTCUSDT 1-minute closes,
8–14 Nov 2026 (72 h windows, +4 h targets), scored through this
program: MAE **$517,28** vs persistence $526,17, worst case $2.505,59
vs $2.609,99. Run time ~0,1 s per case (cap 8 s); two runs byte-identical.

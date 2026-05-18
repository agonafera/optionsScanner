# 📈 Options Trading Opportunity Scanner

> **For Learning & Research Purposes Only — NOT Financial Advice**
>
> Options trading involves significant risk of loss. This tool is designed to help beginners explore and understand options markets using real data. Always consult a licensed financial advisor before placing any real trades.

---

## Overview

This project provides a Python-based options scanner that pulls live market data from Yahoo Finance (free, no API key required) and ranks option contracts based on liquidity, risk, and probability metrics. It is designed to be beginner-friendly and runnable in both a terminal (VS Code) and a Jupyter Notebook environment.

The scanner covers five beginner-friendly strategies:
- **Long Call** — bet the stock goes up
- **Long Put** — bet the stock goes down
- **Covered Call** — generate income on shares you already own
- **Cash-Secured Put** — get paid to agree to buy shares at a lower price
- **Debit Spread** — defined-risk directional trades (framework included)

---

## Files

| File | Description |
|---|---|
| `options_scanner.py` | Full command-line scanner — run in terminal or VS Code |
| `options_scanner_notebook.ipynb` | Same scanner in Jupyter Notebook form — run cell by cell |
| `README.md` | This file |

---

## Requirements

**Python 3.8 or higher** is required.

Install all dependencies with one command:

```bash
pip install yfinance pandas numpy scipy tabulate colorama
```

If you get HTTP/JSON errors from yfinance (Yahoo Finance connectivity issues), also run:

```bash
pip install --upgrade yfinance curl_cffi requests
```

---

## How to Run

### Option A — Command Line / VS Code

```bash
python options_scanner.py
```

You will be prompted to:
1. Enter your own list of ticker symbols, OR
2. Scan a predefined list of stocks filtered by price

### Option B — Jupyter Notebook

```bash
jupyter notebook options_scanner_notebook.ipynb
```

Run each cell in order. Edit the configuration cell (Cell 3) to set your tickers and filters before running the scan cell.

---

## Configuration

All key settings are at the top of `options_scanner.py` (or in Cell 3 of the notebook). Edit these to control what the scanner looks for:

```python
# Tickers to scan when using the predefined list
PREDEFINED_TICKERS = ["F", "BAC", "SOFI", "PLTR", "NIO", ...]

# Only include stocks under this price in predefined scans
MAX_STOCK_PRICE = 50.0

# Options contract filters
MIN_OPEN_INTEREST  = 100    # Skip illiquid contracts
MAX_BID_ASK_SPREAD = 0.50   # Max acceptable spread ($)
MIN_DAYS_TO_EXPIRY = 7      # Ignore very near-term expirations
MAX_DAYS_TO_EXPIRY = 90     # Ignore very far-out expirations
MAX_OTM_PCT        = 0.20   # Skip contracts more than 20% out of the money

# Scoring weights — must sum to 100
WEIGHTS = {
    "open_interest":  20,
    "bid_ask_spread": 15,
    "volume":         10,
    "delta":          20,
    "theta":          10,
    "iv_vs_hv":       15,
    "dte":            10,
}
```

---

## How It Works

### 1. Data Collection
- Pulls 6 months of daily price history using `yfinance`
- Fetches live options chains for up to 3 expiration dates per ticker

### 2. Volatility Calculation
Historical volatility (HV) is calculated at three windows using annualised log returns:

| Metric | Window |
|---|---|
| HV20 | 20 trading days (~1 month) |
| HV30 | 30 trading days (~6 weeks) |
| HV60 | 60 trading days (~3 months) |

### 3. Implied Volatility Estimation
Implied volatility (IV) is estimated using binary search on the Black-Scholes pricing model, then compared against historical volatility to identify whether options are cheap (IV < HV) or expensive (IV > HV) relative to recent realized moves.

### 4. Greeks Calculation
Delta and Theta are calculated using the Black-Scholes model:
- **Delta** — how much the option price moves per $1 move in the stock
- **Theta** — how much value the option loses per day (time decay)

### 5. Scoring (0–100)
Each contract is scored across seven dimensions:

| Dimension | Max Points | What earns a high score |
|---|---|---|
| Open Interest | 20 | ≥ 5,000 (high liquidity) |
| Bid-Ask Spread | 15 | ≤ $0.05 (tight spread) |
| Volume | 10 | ≥ 1,000 contracts traded |
| Delta | 20 | 0.35–0.60 (near at-the-money) |
| Theta | 10 | Slow decay for buyers; fast for sellers |
| IV vs HV | 15 | IV < HV suggests cheap options |
| Days to Expiry | 10 | 21–60 days (balanced time/decay) |

### 6. Output
- Ranked table of top 20 contracts with all key metrics
- Plain-English summary of the top 5 trades
- "What to monitor" checklist specific to each strategy
- Optional CSV export of full results

---

## Sample Output

```
╭──────────┬──────────┬────────────────┬─────────┬────────────┬───────┬──────────╮
│  Score   │  Ticker  │  Strategy      │  Price  │  Strike    │  DTE  │  Option$ │
├──────────┼──────────┼────────────────┼─────────┼────────────┼───────┼──────────┤
│  93      │  SOFI    │  Covered Call  │  $16.07 │  $16.00    │  7    │  $0.50   │
│  81      │  BAC     │  Long Call     │  $49.85 │  $50.00    │  35   │  $1.20   │
│  76      │  F       │  Cash-Sec Put  │  $11.30 │  $11.00    │  28   │  $0.25   │
╰──────────┴──────────┴────────────────┴─────────┴────────────┴───────┴──────────╯
```

---

## Diagnostics

When the scanner runs, it prints a per-ticker filter breakdown so you can see exactly why contracts are being dropped and know which setting to adjust:

```
Contracts scanned: 142  |  No price: 12  |  Low OI (<100): 38  |
Wide spread: 19  |  DTE out of range: 8  |  Deep OTM (>20%): 41  |  Passed: 24
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Expecting value: line 1 column 1` | Run `pip install --upgrade yfinance curl_cffi` — Yahoo Finance changed their API handshake |
| `cannot convert float NaN to integer` | Already handled in latest version via `safe_int()` helper |
| `unsupported operand type(s) for \|` | You are on Python < 3.10 — latest code uses `typing.Optional` for compatibility |
| `No qualifying option contracts found` | Loosen filters: raise `MAX_BID_ASK_SPREAD`, lower `MIN_OPEN_INTEREST`, or increase `MAX_OTM_PCT` |
| All tickers fail in price filter | Run `pip install --upgrade yfinance` — the `fast_info` API was replaced with `history()` |

---

## Key Concepts for Beginners

| Term | Plain-English Meaning |
|---|---|
| **Call option** | The right to *buy* 100 shares at the strike price |
| **Put option** | The right to *sell* 100 shares at the strike price |
| **Strike price** | The agreed price at which you can buy/sell shares |
| **Expiration date** | The date the option contract expires |
| **Premium** | The price you pay (or receive) for the option contract |
| **Delta** | How much the option moves for every $1 move in the stock |
| **Theta** | How much value the option loses each day (time decay) |
| **IV** | Implied volatility — the market's expectation of future price swings |
| **HV** | Historical volatility — how much the stock actually moved recently |
| **Break-even** | The stock price at which the trade neither profits nor loses |
| **OTM** | Out of the money — the strike is away from the current stock price |
| **ATM** | At the money — the strike is near the current stock price |
| **DTE** | Days to expiration |
| **Open Interest** | Number of active contracts outstanding (higher = more liquid) |

---

## Data Source

All market data is sourced from **Yahoo Finance** via the [`yfinance`](https://github.com/ranaroussi/yfinance) library. This is free and requires no API key. Data may be delayed 15–20 minutes during market hours.

---

## Disclaimer

This software is provided for **educational and research purposes only**. It does not constitute financial advice, investment recommendations, or solicitation to buy or sell any security. Options trading involves substantial risk of loss and is not suitable for all investors. Past performance of any scoring methodology does not guarantee future results.

The authors of this project are not licensed financial advisors. Always do your own research and consult a qualified financial professional before making any investment decisions.

---

## License

MIT License — free to use, modify, and distribute with attribution.

"""
=============================================================================
OPTIONS TRADING OPPORTUNITY SCANNER
For Learning and Research Purposes Only — NOT Financial Advice
=============================================================================

This program helps you identify potentially interesting options trades by:
  - Pulling real stock/options data from Yahoo Finance (free, no API key needed)
  - Calculating historical volatility at 20, 30, and 60-day windows
  - Comparing implied vs. historical volatility
  - Scoring each option contract on liquidity, risk, and probability factors
  - Outputting a ranked table of beginner-friendly trade ideas

HOW TO RUN:
  1. Install dependencies:   pip install yfinance pandas numpy scipy tabulate colorama
  2. Run the script:         python options_scanner.py
  3. Or paste into Jupyter:  works cell-by-cell as well

DISCLAIMER:
  This tool is for EDUCATIONAL purposes only. Options trading involves
  significant risk of loss. Always consult a licensed financial advisor
  before making any real trades.
=============================================================================
"""

# ── Standard library imports ──────────────────────────────────────────────────
import warnings
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

# ── Third-party imports ───────────────────────────────────────────────────────
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from scipy.stats import norm
    from tabulate import tabulate
    from colorama import init, Fore, Style
    init(autoreset=True)  # Enables colored terminal output on Windows too
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Please run:  pip install yfinance pandas numpy scipy tabulate colorama")
    sys.exit(1)

warnings.filterwarnings("ignore")  # Suppress noisy yfinance warnings


def safe_int(value, default: int = 0) -> int:
    """Convert a value to int safely, treating NaN/None/'' as default."""
    try:
        v = float(value)
        return default if (v != v) else int(v)  # v != v is True only for NaN
    except (TypeError, ValueError):
        return default


# =============================================================================
# SECTION 1 — CONFIGURATION
# These defaults control what the scanner looks for. Edit freely.
# =============================================================================

# --- Default stock universe when user picks "scan predefined list" ---
PREDEFINED_TICKERS = [
    "F", "BAC", "SOFI", "PLTR", "NIO", "AAL", "CCL", "SNAP", "RIVN",
    "FFIE", "AGNC", "T", "INTC", "KVUE", "WBA", "PFE", "KGC", "HL",
    "SIRI", "NOK", "AMC", "CHPT", "PSFE", "CLOV", "MVIS", "LCID",
    "UWMC", "DKNG", "OPEN", "WKHS"
]

# --- Maximum stock price to include in predefined scan ---
MAX_STOCK_PRICE = 50.0

# --- Options filters ---
MIN_OPEN_INTEREST   = 100      # Skip illiquid contracts
MAX_BID_ASK_SPREAD  = 0.50     # Max acceptable spread ($)
MIN_DAYS_TO_EXPIRY  = 7        # Ignore very near-term (dangerous for beginners)
MAX_DAYS_TO_EXPIRY  = 90       # Keep focus on near-to-medium term
MAX_OTM_PCT         = 0.20     # Skip contracts more than 20% OTM

# --- Scoring weights (must sum to 100) ---
WEIGHTS = {
    "open_interest":   20,
    "bid_ask_spread":  15,
    "volume":          10,
    "delta":           20,
    "theta":           10,
    "iv_vs_hv":        15,
    "dte":             10,
}

# --- Strategies to evaluate ---
STRATEGIES = ["long_call", "long_put", "covered_call", "cash_secured_put", "debit_spread"]


# =============================================================================
# SECTION 2 — DATA FETCHING
# =============================================================================

def fetch_stock_data(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Pull 6 months of daily price history and the current price for a ticker.
    Returns (price_dataframe, current_price) or (None, None) on error.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")       # 6 months of OHLCV data
        if hist.empty:
            return None, None
        current_price = hist["Close"].iloc[-1]
        return hist, round(current_price, 2)
    except Exception as e:
        print(f"  [WARN] Could not fetch price data for {ticker}: {e}")
        return None, None


def fetch_options_chain(ticker: str) -> Tuple[list, Optional[yf.Ticker]]:
    """
    Fetch all available expiration dates and the Ticker object.
    Returns (list_of_expiry_strings, ticker_object) or ([], None).
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options          # list of "YYYY-MM-DD" strings
        if not expirations:
            return [], None
        return expirations, tk
    except Exception as e:
        print(f"  [WARN] Could not fetch options for {ticker}: {e}")
        return [], None


# =============================================================================
# SECTION 3 — VOLATILITY CALCULATIONS
# =============================================================================

def calc_historical_volatility(price_series: pd.Series, window: int) -> Optional[float]:
    """
    Annualised historical volatility using log returns over `window` trading days.

    Formula:
      1. Compute daily log returns: ln(price_t / price_{t-1})
      2. Take the standard deviation of those returns over the window
      3. Annualise: multiply by sqrt(252)  [252 trading days/year]
    """
    if len(price_series) < window + 1:
        return None
    returns = np.log(price_series / price_series.shift(1)).dropna()
    hv = returns.tail(window).std() * np.sqrt(252)
    return round(float(hv), 4)


def black_scholes_price(S, K, T, r, sigma, option_type="call") -> float:
    """
    Classic Black-Scholes option pricing formula.

    Parameters
    ----------
    S     : current stock price
    K     : strike price
    T     : time to expiration in YEARS
    r     : risk-free rate (e.g. 0.05 for 5%)
    sigma : implied volatility as a decimal (e.g. 0.30 for 30%)
    option_type : "call" or "put"

    Returns theoretical option price.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_implied_volatility(
    market_price: float, S: float, K: float, T: float,
    r: float = 0.05, option_type: str = "call"
) -> Optional[float]:
    """
    Estimate IV using binary search (bisection method) on the Black-Scholes model.
    We search for the volatility value that makes the BS price match the market price.

    Returns IV as a decimal, or None if it can't converge.
    """
    if market_price <= 0 or T <= 0:
        return None
    low, high = 0.001, 5.0          # Search between 0.1% and 500% IV
    for _ in range(100):            # Max 100 iterations — always converges
        mid = (low + high) / 2
        price = black_scholes_price(S, K, T, r, sigma=mid, option_type=option_type)
        if abs(price - market_price) < 0.001:
            return round(mid, 4)
        if price < market_price:
            low = mid
        else:
            high = mid
    return None


def calc_greeks(S, K, T, r, sigma, option_type="call") -> dict:
    """
    Calculate Delta and Theta for an option using Black-Scholes.

    Delta: sensitivity of option price to a $1 move in the stock
    Theta: daily time decay (how much value the option loses per day)
    """
    if T <= 0 or sigma <= 0:
        return {"delta": None, "theta": None}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

    # Theta (annualised then divided by 365 → daily decay)
    theta_annual = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))
    )
    theta_daily = theta_annual / 365

    return {
        "delta": round(delta, 4),
        "theta": round(theta_daily, 4),
    }


# =============================================================================
# SECTION 4 — SCORING ENGINE
# =============================================================================

def score_option(row: dict, hv20: Optional[float]) -> Tuple[int, List[str]]:
    """
    Score a single option contract on a 0–100 scale.
    Returns (score, list_of_reason_strings).

    Higher score = more favourable across all evaluated dimensions.
    """
    score = 0
    reasons = []

    # ── 1. Open Interest (liquidity) ──────────────────────────────────────────
    oi = row.get("openInterest", 0) or 0
    if oi >= 5000:
        pts = WEIGHTS["open_interest"]
        reasons.append(f"High liquidity (OI={oi:,})")
    elif oi >= 1000:
        pts = int(WEIGHTS["open_interest"] * 0.75)
        reasons.append(f"Good liquidity (OI={oi:,})")
    elif oi >= MIN_OPEN_INTEREST:
        pts = int(WEIGHTS["open_interest"] * 0.40)
        reasons.append(f"Moderate liquidity (OI={oi:,})")
    else:
        pts = 0
    score += pts

    # ── 2. Bid-Ask Spread (transaction cost proxy) ────────────────────────────
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    spread = ask - bid
    if spread <= 0.05:
        pts = WEIGHTS["bid_ask_spread"]
        reasons.append(f"Tight spread (${spread:.2f})")
    elif spread <= 0.15:
        pts = int(WEIGHTS["bid_ask_spread"] * 0.80)
        reasons.append(f"Narrow spread (${spread:.2f})")
    elif spread <= MAX_BID_ASK_SPREAD:
        pts = int(WEIGHTS["bid_ask_spread"] * 0.50)
    else:
        pts = 0
        reasons.append(f"⚠ Wide spread (${spread:.2f}) — slippage risk")
    score += pts

    # ── 3. Volume ─────────────────────────────────────────────────────────────
    vol = row.get("volume", 0) or 0
    if vol >= 1000:
        pts = WEIGHTS["volume"]
        reasons.append(f"Strong volume ({vol:,})")
    elif vol >= 100:
        pts = int(WEIGHTS["volume"] * 0.60)
    else:
        pts = int(WEIGHTS["volume"] * 0.20)
    score += pts

    # ── 4. Delta (moneyness check) ────────────────────────────────────────────
    delta = row.get("delta_calc")
    if delta is not None:
        abs_delta = abs(delta)
        # Sweet spot for beginners: 0.30–0.60 (not deep OTM, not deep ITM)
        if 0.35 <= abs_delta <= 0.60:
            pts = WEIGHTS["delta"]
            reasons.append(f"Healthy delta ({delta:.2f}) — near ATM")
        elif 0.25 <= abs_delta < 0.35:
            pts = int(WEIGHTS["delta"] * 0.70)
            reasons.append(f"Moderate delta ({delta:.2f})")
        elif 0.60 < abs_delta <= 0.80:
            pts = int(WEIGHTS["delta"] * 0.80)
            reasons.append(f"High delta ({delta:.2f}) — deeper ITM")
        else:
            pts = int(WEIGHTS["delta"] * 0.30)
            reasons.append(f"Low delta ({delta:.2f}) — far OTM, speculative")
    else:
        pts = 0
    score += pts

    # ── 5. Theta (time decay) ─────────────────────────────────────────────────
    theta = row.get("theta_calc")
    dte   = row.get("dte", 30)
    if theta is not None:
        # For buyers: we want slow decay (theta not too negative)
        # For sellers (covered call / CSP): fast decay is good
        if row.get("option_type") in ("covered_call", "cash_secured_put"):
            # Seller perspective: more theta = more premium collected per day
            if theta <= -0.03:
                pts = WEIGHTS["theta"]
                reasons.append(f"Fast decay (θ={theta:.3f}) — good for sellers")
            elif theta <= -0.01:
                pts = int(WEIGHTS["theta"] * 0.70)
            else:
                pts = int(WEIGHTS["theta"] * 0.40)
        else:
            # Buyer perspective: we want slower decay
            if dte >= 30 and theta > -0.03:
                pts = WEIGHTS["theta"]
                reasons.append(f"Manageable decay (θ={theta:.3f})")
            elif dte >= 14:
                pts = int(WEIGHTS["theta"] * 0.60)
            else:
                pts = int(WEIGHTS["theta"] * 0.20)
                reasons.append("⚠ Near expiry — theta decay accelerating")
    else:
        pts = 0
    score += pts

    # ── 6. IV vs HV (volatility edge) ────────────────────────────────────────
    iv = row.get("iv_calc")
    if iv and hv20:
        iv_hv_ratio = iv / hv20
        if iv_hv_ratio < 0.90:
            pts = WEIGHTS["iv_vs_hv"]
            reasons.append(f"IV ({iv:.0%}) < HV ({hv20:.0%}) — options may be cheap (buy-side edge)")
        elif iv_hv_ratio <= 1.10:
            pts = int(WEIGHTS["iv_vs_hv"] * 0.75)
            reasons.append(f"IV ≈ HV — fair pricing")
        elif iv_hv_ratio <= 1.30:
            pts = int(WEIGHTS["iv_vs_hv"] * 0.50)
            reasons.append(f"IV ({iv:.0%}) > HV ({hv20:.0%}) — slightly elevated premium")
        else:
            pts = int(WEIGHTS["iv_vs_hv"] * 0.20)
            reasons.append(f"⚠ IV ({iv:.0%}) >> HV ({hv20:.0%}) — expensive premium (seller edge)")
    else:
        pts = int(WEIGHTS["iv_vs_hv"] * 0.50)   # Neutral if data unavailable
    score += pts

    # ── 7. Days to Expiration ─────────────────────────────────────────────────
    if 21 <= dte <= 60:
        pts = WEIGHTS["dte"]
        reasons.append(f"Good DTE ({dte}d) — balanced time/decay")
    elif 14 <= dte < 21:
        pts = int(WEIGHTS["dte"] * 0.60)
    elif 7 <= dte < 14:
        pts = int(WEIGHTS["dte"] * 0.30)
        reasons.append(f"⚠ Short DTE ({dte}d) — high time-decay risk")
    elif dte > 60:
        pts = int(WEIGHTS["dte"] * 0.80)
        reasons.append(f"Longer dated ({dte}d) — more time, but higher cost")
    else:
        pts = 0
    score += pts

    return min(score, 100), reasons


# =============================================================================
# SECTION 5 — OPTION ANALYSIS CORE
# =============================================================================

def analyse_option_row(
    row: pd.Series,
    current_price: float,
    expiry_str: str,
    contract_type: str,       # "call" or "put"
    strategy: str,
    hv20: Optional[float],
    hv30: Optional[float],
    hv60: Optional[float],
    r: float = 0.05,
) -> Optional[dict]:
    """
    Analyse a single option contract row from the options chain DataFrame.
    Returns a result dictionary, or None if the contract should be skipped.
    """
    try:
        strike    = float(row.get("strike", 0))
        bid       = float(row.get("bid", 0) or 0)
        ask       = float(row.get("ask", 0) or 0)
        last      = float(row.get("lastPrice", 0) or 0)
        iv_raw    = row.get("impliedVolatility")
        oi        = safe_int(row.get("openInterest", 0))
        volume    = safe_int(row.get("volume", 0))

        # ── Resolve mid-price ─────────────────────────────────────────────────
        # yfinance often returns bid=0 / ask=0 for thinly traded contracts.
        # Fall back to lastPrice so we don't discard good contracts.
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif ask > 0:
            mid = ask
        elif last > 0:
            mid = last
            bid = last * 0.95   # synthetic bid/ask so spread calc still works
            ask = last * 1.05
        else:
            return None         # Truly no usable price — skip

        # ── Open interest filter ──────────────────────────────────────────────
        if oi < MIN_OPEN_INTEREST:
            return None

        # ── Bid-ask spread: dual gate (absolute $ AND % of mid) ───────────────
        # Using only an absolute cap ($0.50) kills cheap contracts unfairly.
        # Using only % kills expensive contracts unfairly.
        # We skip only when BOTH limits are breached simultaneously.
        spread     = ask - bid
        spread_pct = spread / mid if mid > 0 else 999
        if spread > MAX_BID_ASK_SPREAD and spread_pct > 0.50:
            return None

        if mid <= 0:
            return None

        # ── Days to expiration ────────────────────────────────────────────────
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
        dte = (expiry_dt - datetime.now()).days
        if dte < MIN_DAYS_TO_EXPIRY:
            return None                          # Too close to expiry
        if dte > MAX_DAYS_TO_EXPIRY:
            return None

        # ── Moneyness filter — skip deep OTM ─────────────────────────────────
        otm_pct = abs(strike - current_price) / current_price
        if otm_pct > MAX_OTM_PCT:
            return None

        T = dte / 365.0                          # Time in years for BS formula

        # ── Implied volatility ────────────────────────────────────────────────
        iv_calc = estimate_implied_volatility(mid, current_price, strike, T, r, contract_type)
        if iv_calc is None and iv_raw:
            iv_calc = float(iv_raw)              # Fall back to Yahoo's IV

        # ── Greeks ────────────────────────────────────────────────────────────
        sigma = iv_calc if iv_calc else (hv20 or 0.30)
        greeks = calc_greeks(current_price, strike, T, r, sigma, contract_type)

        # ── Break-even price ──────────────────────────────────────────────────
        if contract_type == "call":
            breakeven = strike + mid             # Pay premium; need stock > strike+premium
        else:
            breakeven = strike - mid             # Pay premium; need stock < strike-premium

        # ── Max risk ──────────────────────────────────────────────────────────
        # For long options (buyer), max loss = premium paid (× 100 shares per contract)
        max_risk = mid * 100

        # ── Strategy-specific risk label ──────────────────────────────────────
        if strategy == "covered_call":
            risk_warning = "Risk: stock drops below break-even; premium offsets some loss."
        elif strategy == "cash_secured_put":
            risk_warning = f"Risk: assigned shares at ${strike:.2f} if stock falls below strike."
        elif strategy in ("long_call", "long_put"):
            risk_warning = f"Risk: lose 100% of premium (${max_risk:.0f}) if expires worthless."
        elif strategy == "debit_spread":
            risk_warning = "Risk: limited to net debit paid; max gain capped at spread width minus debit."
        else:
            risk_warning = "Review strategy risk before trading."

        # ── Assemble result dict ──────────────────────────────────────────────
        result = {
            "strike":        strike,
            "expiry":        expiry_str,
            "dte":           dte,
            "mid":           round(mid, 2),
            "bid":           bid,
            "ask":           ask,
            "openInterest":  oi,
            "volume":        volume,
            "iv_calc":       iv_calc,
            "iv_raw":        float(iv_raw) if iv_raw else None,
            "delta_calc":    greeks["delta"],
            "theta_calc":    greeks["theta"],
            "breakeven":     round(breakeven, 2),
            "max_risk":      round(max_risk, 2),
            "option_type":   strategy,
            "contract_type": contract_type,
            "risk_warning":  risk_warning,
            "hv20":          hv20,
            "hv30":          hv30,
            "hv60":          hv60,
        }
        return result

    except Exception as e:
        return None   # Silently skip malformed rows


# =============================================================================
# SECTION 6 — PER-TICKER SCANNER
# =============================================================================

def scan_ticker(ticker: str) -> List[dict]:
    """
    Full scan of one ticker:
      1. Fetch price history → calculate HV
      2. Fetch options chain for nearest suitable expirations
      3. Analyse each contract; score it
      4. Return a list of result dicts (one per qualifying contract)
    """
    results = []
    print(f"\n{'─'*60}")
    print(f"  Scanning {Fore.CYAN}{ticker}{Style.RESET_ALL} …")

    # ── Price data & historical volatility ────────────────────────────────────
    hist, current_price = fetch_stock_data(ticker)
    if hist is None or current_price is None:
        print(f"  {Fore.YELLOW}Skipping — no price data.{Style.RESET_ALL}")
        return results

    closes = hist["Close"]
    hv20   = calc_historical_volatility(closes, 20)
    hv30   = calc_historical_volatility(closes, 30)
    hv60   = calc_historical_volatility(closes, 60)

    print(f"  Price: ${current_price:.2f}  |  HV20={hv20:.1%}  HV30={hv30:.1%}  HV60={hv60:.1%}"
          if hv20 and hv30 and hv60 else f"  Price: ${current_price:.2f}  |  Volatility data limited")

    # ── Options chain ─────────────────────────────────────────────────────────
    expirations, tk = fetch_options_chain(ticker)
    if not expirations:
        print(f"  {Fore.YELLOW}Skipping — no options chain available.{Style.RESET_ALL}")
        return results

    # Filter expirations to our DTE window
    today = datetime.now()
    valid_exps = []
    for exp in expirations:
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            dte    = (exp_dt - today).days
            if MIN_DAYS_TO_EXPIRY <= dte <= MAX_DAYS_TO_EXPIRY:
                valid_exps.append(exp)
        except Exception:
            continue

    if not valid_exps:
        print(f"  {Fore.YELLOW}No expirations in {MIN_DAYS_TO_EXPIRY}–{MAX_DAYS_TO_EXPIRY} day window.{Style.RESET_ALL}")
        print(f"  Available expirations: {expirations[:6]}")
        return results

    # ── Diagnostic counters — shown per ticker so you can tune filters ────────
    diag = {
        "total_rows":   0,
        "no_price":     0,
        "low_oi":       0,
        "wide_spread":  0,
        "dte_range":    0,
        "deep_otm":     0,
        "passed":       0,
    }

    # We look at up to 3 expirations per ticker to keep runtime reasonable
    for exp in valid_exps[:3]:
        try:
            chain = tk.option_chain(exp)
        except Exception as e:
            print(f"  [WARN] Could not fetch chain for {exp}: {e}")
            continue

        for side_df, contract_type, strategies in [
            (chain.calls, "call", ["long_call", "covered_call"]),
            (chain.puts,  "put",  ["long_put",  "cash_secured_put"]),
        ]:
            for _, row in side_df.iterrows():
                diag["total_rows"] += 1

                # ── Run the same pre-checks as analyse_option_row ─────────────
                # so we can count what's being dropped and why.
                bid   = float(row.get("bid", 0) or 0)
                ask   = float(row.get("ask", 0) or 0)
                last  = float(row.get("lastPrice", 0) or 0)
                oi    = safe_int(row.get("openInterest", 0))
                strike = float(row.get("strike", 0))

                # Price availability
                if bid <= 0 and ask <= 0 and last <= 0:
                    diag["no_price"] += 1
                    continue

                # OI
                if oi < MIN_OPEN_INTEREST:
                    diag["low_oi"] += 1
                    continue

                # Spread (same dual-gate logic as analyse_option_row)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask if ask > 0 else last)
                spread = ask - bid if (bid > 0 and ask > 0) else 0
                spread_pct = spread / mid if mid > 0 else 0
                if spread > MAX_BID_ASK_SPREAD and spread_pct > 0.50:
                    diag["wide_spread"] += 1
                    continue

                # DTE
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dte = (exp_dt - today).days
                if not (MIN_DAYS_TO_EXPIRY <= dte <= MAX_DAYS_TO_EXPIRY):
                    diag["dte_range"] += 1
                    continue

                # OTM
                otm_pct = abs(strike - current_price) / current_price if current_price > 0 else 1
                if otm_pct > MAX_OTM_PCT:
                    diag["deep_otm"] += 1
                    continue

                diag["passed"] += 1

                # Now do the full analysis and scoring
                for strategy in strategies:
                    res = analyse_option_row(
                        row, current_price, exp, contract_type, strategy, hv20, hv30, hv60
                    )
                    if res:
                        score, reasons = score_option(res, hv20)
                        res.update({
                            "ticker":        ticker,
                            "current_price": current_price,
                            "strategy":      strategy,
                            "score":         score,
                            "reasons":       reasons,
                        })
                        results.append(res)

    # ── Print diagnostics ─────────────────────────────────────────────────────
    total = diag["total_rows"]
    if total > 0:
        print(f"  Contracts scanned: {total}  |  "
              f"No price: {diag['no_price']}  |  "
              f"Low OI (<{MIN_OPEN_INTEREST}): {diag['low_oi']}  |  "
              f"Wide spread: {diag['wide_spread']}  |  "
              f"DTE out of range: {diag['dte_range']}  |  "
              f"Deep OTM (>{MAX_OTM_PCT:.0%}): {diag['deep_otm']}  |  "
              f"{Fore.GREEN}Passed: {diag['passed']}{Style.RESET_ALL}")
    print(f"  → Qualifying contracts added to results: {Fore.GREEN}{len(results)}{Style.RESET_ALL}")
    return results


# =============================================================================
# SECTION 7 — OUTPUT FORMATTING
# =============================================================================

STRATEGY_LABELS = {
    "long_call":        "Long Call",
    "long_put":         "Long Put",
    "covered_call":     "Covered Call",
    "cash_secured_put": "Cash-Secured Put",
    "debit_spread":     "Debit Spread",
}


def format_pct(v) -> str:
    return f"{v:.1%}" if v is not None else "N/A"

def format_dollar(v) -> str:
    return f"${v:.2f}" if v is not None else "N/A"

def format_num(v) -> str:
    return f"{v:,}" if v is not None else "N/A"


def build_output_table(results: List[dict]) -> pd.DataFrame:
    """
    Convert raw result dicts into a clean display DataFrame.
    """
    rows = []
    for r in results:
        rows.append({
            "Score":           r["score"],
            "Ticker":          r["ticker"],
            "Strategy":        STRATEGY_LABELS.get(r["strategy"], r["strategy"]),
            "Price":           format_dollar(r["current_price"]),
            "Strike":          format_dollar(r["strike"]),
            "Expiry":          r["expiry"],
            "DTE":             r["dte"],
            "Option $":        format_dollar(r["mid"]),
            "Delta":           f'{r["delta_calc"]:.2f}' if r["delta_calc"] else "N/A",
            "Theta/day":       f'{r["theta_calc"]:.4f}' if r["theta_calc"] else "N/A",
            "IV (calc)":       format_pct(r["iv_calc"]),
            "HV20":            format_pct(r["hv20"]),
            "Open Int":        format_num(r["openInterest"]),
            "Volume":          format_num(r["volume"]),
            "Break-even":      format_dollar(r["breakeven"]),
            "Max Risk":        format_dollar(r["max_risk"]),
            "Top Reasons":     " | ".join(r["reasons"][:3]),
            "Risk Warning":    r["risk_warning"],
        })
    return pd.DataFrame(rows).sort_values("Score", ascending=False)


def print_table(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Print the top_n results as a formatted table with colour-coded scores.
    """
    display = df.head(top_n).copy()

    # Colour-code the score column
    def score_colour(s):
        try:
            s = int(s)
            if s >= 75:   return Fore.GREEN  + str(s) + Style.RESET_ALL
            elif s >= 50: return Fore.YELLOW + str(s) + Style.RESET_ALL
            else:         return Fore.RED    + str(s) + Style.RESET_ALL
        except:
            return str(s)

    display["Score"] = display["Score"].apply(score_colour)

    # Truncate long text columns for readability
    display["Top Reasons"]  = display["Top Reasons"].str[:60]
    display["Risk Warning"] = display["Risk Warning"].str[:50]

    print(tabulate(display, headers="keys", tablefmt="rounded_outline",
                   showindex=False, numalign="right"))


def print_summary(results: List[dict], top_n: int = 5) -> None:
    """
    Print a plain-English summary of the top-ranked trades.
    """
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = sorted_results[:top_n]

    print("\n" + "=" * 70)
    print(f"  📊  TOP {top_n} TRADE IDEAS — SUMMARY & WHAT TO MONITOR")
    print("=" * 70)

    for i, r in enumerate(top, 1):
        strategy_label = STRATEGY_LABELS.get(r["strategy"], r["strategy"])
        iv_str  = format_pct(r["iv_calc"])
        hv_str  = format_pct(r["hv20"])
        reasons = " ".join(r["reasons"])

        print(f"\n  #{i}  {Fore.CYAN}{r['ticker']}{Style.RESET_ALL}  "
              f"{strategy_label}  Strike ${r['strike']:.2f}  "
              f"Exp {r['expiry']}  Score: {Fore.GREEN}{r['score']}/100{Style.RESET_ALL}")
        print(f"  Why ranked highly: {reasons}")
        print(f"  IV={iv_str}  HV20={hv_str}  "
              f"Delta={r['delta_calc']}  Theta={r['theta_calc']}")
        print(f"  Break-even: ${r['breakeven']:.2f}  |  Max Risk: ${r['max_risk']:.0f}")
        print(f"  ⚠ {r['risk_warning']}")
        print()
        print("  WHAT TO MONITOR AFTER ENTRY:")
        if r["strategy"] in ("long_call", "long_put"):
            print("   • Watch the stock price daily — are you moving toward break-even?")
            print("   • Track delta: if it falls below 0.20, the trade is becoming speculative.")
            print("   • Monitor IV changes: a drop in IV hurts option buyers (IV crush).")
            print("   • Consider exiting at 50% profit or 50% loss to manage risk.")
        elif r["strategy"] == "covered_call":
            print("   • Track whether the stock approaches the strike price.")
            print("   • If stock rallies above strike, you may be 'called away' — shares sold.")
            print("   • Can roll the call to a later date if you want to keep shares.")
        elif r["strategy"] == "cash_secured_put":
            print("   • Keep the cash to buy 100 shares reserved in your account.")
            print("   • If stock drops below strike, you'll be assigned shares — is that OK?")
            print("   • Watch for big negative news that could cause a sharp drop.")
        print(f"  {'─'*60}")

    print("\n  ⚠  REMINDER: This is for EDUCATIONAL purposes only.")
    print("     Always do your own research and consider consulting a financial advisor.")
    print("=" * 70 + "\n")


# =============================================================================
# SECTION 8 — USER INTERFACE (CLI)
# =============================================================================

def get_tickers_from_user() -> List[str]:
    """
    Prompt the user to either enter custom tickers or use the predefined list.
    """
    print("\n" + "=" * 60)
    print("  OPTIONS SCANNER — Setup")
    print("=" * 60)
    print("\n  Choose ticker source:")
    print("  [1] Enter my own list of tickers")
    print(f"  [2] Scan predefined list (stocks under ${MAX_STOCK_PRICE})")
    print("  [3] Use a mix (predefined list + my additions)")

    while True:
        choice = input("\n  Enter choice [1/2/3]: ").strip()
        if choice == "1":
            raw = input("  Enter tickers separated by commas (e.g. AAPL, MSFT, TSLA): ")
            tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
            return tickers

        elif choice == "2":
            print(f"\n  Using predefined list — will filter to stocks under ${MAX_STOCK_PRICE}.")
            return list(PREDEFINED_TICKERS)

        elif choice == "3":
            raw = input("  Enter extra tickers to add (comma-separated): ")
            extras = [t.strip().upper() for t in raw.split(",") if t.strip()]
            return list(PREDEFINED_TICKERS) + extras

        else:
            print("  Please enter 1, 2, or 3.")


def filter_by_price(tickers: List[str], max_price: float) -> List[str]:
    """
    Filter tickers to those whose current price is at or below max_price.

    Uses tk.history() to get the last closing price — this is the most
    reliable method across all yfinance versions. fast_info broke in newer
    yfinance releases and is no longer used here.
    """
    filtered = []
    skipped  = []
    print(f"\n  Filtering to stocks under ${max_price:.2f} …")

    for ticker in tickers:
        price = None
        try:
            tk   = yf.Ticker(ticker)
            hist = tk.history(period="5d")          # Just grab last 5 trading days
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            pass

        if price is None:
            # If we genuinely can't get a price, include the ticker anyway
            # so the main scanner can try (it has its own error handling).
            filtered.append(ticker)
        elif price <= max_price:
            filtered.append(ticker)
        else:
            skipped.append(f"{ticker}(${price:.0f})")

    if skipped:
        print(f"  Excluded (over ${max_price:.0f}): {', '.join(skipped)}")
    print(f"  {len(filtered)} of {len(tickers)} tickers will be scanned.")
    return filtered


# =============================================================================
# SECTION 9 — MAIN ENTRY POINT
# =============================================================================

def main():
    print(Fore.CYAN + r"""
  ╔═══════════════════════════════════════════════════════╗
  ║        OPTIONS TRADING OPPORTUNITY SCANNER            ║
  ║          For Learning & Research Only                 ║
  ║   NOT financial advice — trade at your own risk       ║
  ╚═══════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

    # ── Get ticker list from user ──────────────────────────────────────────────
    tickers = get_tickers_from_user()

    # If they chose predefined, apply price filter
    if any(t in PREDEFINED_TICKERS for t in tickers):
        user_added = [t for t in tickers if t not in PREDEFINED_TICKERS]
        predefined  = [t for t in tickers if t in PREDEFINED_TICKERS]
        predefined  = filter_by_price(predefined, MAX_STOCK_PRICE)
        tickers     = predefined + user_added

    if not tickers:
        print("  No tickers to scan. Exiting.")
        return

    print(f"\n  Scanning {len(tickers)} ticker(s): {', '.join(tickers)}")
    print("  This may take a minute depending on the number of tickers …\n")

    # ── Run the scan ───────────────────────────────────────────────────────────
    all_results = []
    for ticker in tickers:
        results = scan_ticker(ticker)
        all_results.extend(results)

    if not all_results:
        print("\n  No qualifying option contracts found. Try adjusting filters at the top of the script.")
        return

    # ── Build and display output table ─────────────────────────────────────────
    df = build_output_table(all_results)

    print(f"\n\n{'='*70}")
    print(f"  TOP 20 RANKED OPTIONS (out of {len(all_results)} qualifying contracts)")
    print(f"{'='*70}\n")
    print_table(df, top_n=20)

    # ── Print plain-English summary ────────────────────────────────────────────
    print_summary(all_results, top_n=5)

    # ── Optional: save to CSV ─────────────────────────────────────────────────
    save = input("\n  Save full results to CSV? [y/N]: ").strip().lower()
    if save == "y":
        filename = f"options_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # Remove colour codes from Score for clean CSV
        df_clean = build_output_table(all_results)   # rebuild without colour
        df_clean.to_csv(filename, index=False)
        print(f"  Saved → {filename}")

    print("\n  Done. Happy (paper) trading! 📈\n")


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

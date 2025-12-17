"""
Binomial Tree Implementation for Cox-Ross-Rubinstein (CRR) Method

Recombinant trees with equal jump sizes.

Key Points
 - For calls on non-dividend-paying stocks, early exercise is never optimal, so American = European price.
 - For puts, early exercise can be optimal if the stock falls enough, so the American put is more valuable than its European counterpart.
"""

import numpy as np
from time import time
from functools import wraps


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def CRR_method_fast_european(K, T, S0, r, N, sigma, opttype="C"):
    """
    Fast European option pricing using the Cox-Ross-Rubinstein method.
    """
    # precompute constants
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # terminal stock prices S_N(j) = S0 * u^j * d^(N-j), j=0..N
    j = np.arange(N + 1, dtype=float)
    S = S0 * (u**j) * (d ** (N - j))

    # terminal payoffs
    if opttype.upper() == "C":
        C = np.maximum(S - K, 0.0)
    else:
        C = np.maximum(K - S, 0.0)

    # backward induction using vectorized slicing
    for i in range(N, 0, -1):
        # risk-neutral expectation on the first i nodes
        C = disc * (q * C[1 : i + 1] + (1.0 - q) * C[0:i])

    return C[0]


@timing
def CRR_method_fast_american(K, T, S0, r, N, sigma, opttype="C"):
    """
    Fast American option pricing using the Cox-Ross-Rubinstein method.
    """
    # precompute constants
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # terminal stock prices
    j = np.arange(N + 1, dtype=float)
    S = S0 * (u**j) * (d ** (N - j))

    # terminal payoffs
    if opttype.upper() == "C":
        C = np.maximum(S - K, 0.0)
    else:
        C = np.maximum(K - S, 0.0)

    # backward induction with early exercise
    for i in range(N, 0, -1):
        S = S0 * (u ** np.arange(i)) * (d ** (i - np.arange(i)))
        continuation = disc * (q * C[1 : i + 1] + (1.0 - q) * C[0:i])
        if opttype.upper() == "C":
            exercise = np.maximum(S - K, 0.0)
        else:
            exercise = np.maximum(K - S, 0.0)
        C = np.maximum(continuation, exercise)

    return C[0]


@timing
def CRR_method_fast_american_dividend(
    K, T, S0, r, N, sigma, div_times, div_yields, opttype="C"
):
    """
    Fast American option pricing with discrete proportional dividends.

    Parameters:
    -----------
    K : Strike price
    T : Time to maturity (years)
    S0 : Initial stock price
    r : Risk-free rate
    N : Number of time steps
    sigma : Volatility
    div_times : Times when dividends are paid (in years from now)
    div_yields : Proportional dividend yields (delta).
    Stock price becomes (1-delta)*S after dividend.
    opttype : 'C' for call, 'P' for put
    """
    # precompute constants
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # convert dividend times to time step indices
    div_steps = [int(np.round(t / dt)) for t in div_times]
    div_dict = {step: delta for step, delta in zip(div_steps, div_yields)}

    # terminal stock prices (accounting for all dividends up to maturity)
    j = np.arange(N + 1, dtype=float)
    S = S0 * (u**j) * (d ** (N - j))

    # apply all dividends that occur by time T
    for step in div_steps:
        if step <= N:
            S *= 1.0 - div_dict[step]

    # terminal payoffs
    if opttype.upper() == "C":
        C = np.maximum(S - K, 0.0)
    else:
        C = np.maximum(K - S, 0.0)

    # backward induction with early exercise
    for i in range(N, 0, -1):
        # compute stock prices at time step i-1
        j_arr = np.arange(i)
        S = S0 * (u**j_arr) * (d ** (i - 1 - j_arr))

        # apply dividends that have occurred up to step i-1
        for step in div_steps:
            if step < i:
                S *= 1.0 - div_dict[step]

        # continuation value
        continuation = disc * (q * C[1 : i + 1] + (1.0 - q) * C[0:i])

        # exercise value
        if opttype.upper() == "C":
            exercise = np.maximum(S - K, 0.0)
        else:
            exercise = np.maximum(K - S, 0.0)

        C = np.maximum(continuation, exercise)

    return C[0]


@timing
def CRR_method_fast_american_cash_dividend(
    K, T, S0, r, N, sigma, div_times, div_amounts, opttype="C"
):
    """
    Fast American option pricing with discrete cash dividends.

    Parameters:
    -----------
    K : Strike price
    T : Time to maturity (years)
    S0 : Initial stock price
    r : Risk-free rate
    N : Number of time steps
    sigma : Volatility
    div_times : Times when dividends are paid (in years from now)
    div_amounts : Cash dividend amounts (D). Stock price becomes S - D after dividend.
    opttype : 'C' for call, 'P' for put

    Returns:
    --------
    float : Option price

    Note:
    -----
    Stock price evolution: S * exp(xu) * exp(xd) - sum(D_i) for dividends paid before time t
    """
    # precompute constants
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # convert dividend times to time step indices
    div_steps = [int(np.round(t / dt)) for t in div_times]
    div_dict = {step: amount for step, amount in zip(div_steps, div_amounts)}

    # terminal stock prices (accounting for all cash dividends up to maturity)
    j = np.arange(N + 1, dtype=float)
    S = S0 * (u**j) * (d ** (N - j))

    # subtract all cash dividends that occur by time T
    total_div = sum(div_dict[step] for step in div_steps if step <= N)
    S = np.maximum(S - total_div, 0.0)  # stock price can't go negative

    # terminal payoffs
    if opttype.upper() == "C":
        C = np.maximum(S - K, 0.0)
    else:
        C = np.maximum(K - S, 0.0)

    # backward induction with early exercise
    for i in range(N, 0, -1):
        # compute stock prices at time step i-1 before any dividend adjustment
        j_arr = np.arange(i)
        S = S0 * (u**j_arr) * (d ** (i - 1 - j_arr))

        # subtract cash dividends that have occurred up to step i-1
        cum_div = sum(div_dict[step] for step in div_steps if step < i)
        S = np.maximum(S - cum_div, 0.0)

        # continuation value
        continuation = disc * (q * C[1 : i + 1] + (1.0 - q) * C[0:i])

        # exercise value
        if opttype.upper() == "C":
            exercise = np.maximum(S - K, 0.0)
        else:
            exercise = np.maximum(K - S, 0.0)

        C = np.maximum(continuation, exercise)

    return C[0]


if __name__ == "__main__":
    S0 = 100
    K = 110
    T = 0.5
    r = 0.06
    N = 100
    sigma = 0.3

    print("European CRR (C):", CRR_method_fast_european(K, T, S0, r, N, sigma, "C"))
    print("American CRR (C):", CRR_method_fast_american(K, T, S0, r, N, sigma, "C"))
    print("European CRR (P):", CRR_method_fast_european(K, T, S0, r, N, sigma, "P"))
    print("American CRR (P):", CRR_method_fast_american(K, T, S0, r, N, sigma, "P"))

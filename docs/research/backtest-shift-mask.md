<<User>>

Let's now discuss about backtesting. 

I have signal matrix 2d. Columns are security ids and index are dates. Values are signals. 
I converted these signals into tradable portfolio weights. 

Now I have to multiply with return matrix to do the vectorized backtesting. 

Important caveat is that I have to shift signal matrix by 1 day so that we trade based on yesterday's signal. 

However, a question arises: 

We have universe mask that defines the tradable assets. Universe mask is set on initialization and daily universe dynamically changes as liquidity rank differs. 

We always mask the signal and always mask the return before but do we have to mask again after we shift the weight matrix? 

<<AI>>

This is a critical question and a common pitfall in vectorized backtesting.

**Yes, you absolutely must mask again *after* you shift the weight matrix.**

Here’s the simple explanation, followed by the correct step-by-step process.

### The "Why": The Time Mismatch

The core of the problem is a time mismatch between your *information* and your *ability to trade*.

1.  **Signal at $t-1$ (Yesterday):** You generate your signal (`signal[t-1]`) and portfolio weights (`weights[t-1]`) using the universe from *yesterday* (`mask[t-1]`).
2.  **Trade at $t$ (Today):** You shift your weights (`weights.shift(1)`) to trade on them *today*. However, the tradable universe *today* (`mask[t]`) might be different\!

**The Golden Rule of Backtesting:** You can only earn returns on assets that are in your tradable universe **on the day you are holding them.**

Your `shifted_weights[t]` represents your *desired* portfolio based on yesterday's information. You **must** apply `mask[t]` (today's mask) to these weights. This simulates the real-world scenario of liquidating any stocks that fell out of your tradable universe overnight.

-----

### What Happens If You Don't Mask Again?

  * **Stock Exits Universe:** A stock was in your universe at `t-1` but **not** at `t`.

      * `shifted_weights[t]` has a *non-NaN* weight (e.g., 0.5) for this stock.
      * `masked_returns[t]` has a *NaN* value for this stock.
      * **Your PnL calculation:** `0.5 * NaN = NaN`. This `NaN` will infect your entire portfolio sum for day `t`, polluting your backtest.

  * **Stock Enters Universe:** A stock was **not** in your universe at `t-1` but **is** at `t`.

      * `shifted_weights[t]` has a *NaN* weight (correct, you couldn't trade it).
      * `masked_returns[t]` has a *non-NaN* return.
      * **Your PnL calculation:** `NaN * return = NaN`. This is also correct but highlights that the first scenario is the real problem.

-----

### The Correct Vectorized Pipeline

Here is the correct, safe procedure.

  * `signal` (T, N): Your raw alpha signal
  * `returns` (T, N): Your raw forward returns
  * `mask` (T, N): Your *dynamic* universe mask (True/False or 1/0)

**Step 1: Create Weights (Based on $t-1$ Info)**

```python
# 1. Mask the raw signal.
# (You can't form an opinion on non-tradable stocks)
masked_signal = signal.where(mask) 

# 2. Convert masked signals to portfolio weights.
# (This runs your scaling/neutralization on the masked data)
weight_matrix = scale_to_weights(masked_signal) 
```

*`weight_matrix` now represents your target portfolio at the *end* of each day.*

**Step 2: Create Tradable Portfolio for PnL (Based on $t$ Info)**

```python
# 3. Shift weights to represent the portfolio you hold *today*.
# (weights[t] is now based on info from [t-1])
weights_t = weight_matrix.shift(time=1)

# 4. Mask returns with *today's* universe.
returns_t = returns.where(mask)

# 5. [THE CRITICAL STEP]
# Mask your *shifted weights* with *today's* universe.
# This liquidates positions that fell out of the universe.
final_weights_t = weights_t.where(mask) 
```

**Step 3: Calculate PnL**

```python
# 6. Calculate PnL.
# Both arrays are now aligned to today's (t) universe.
# Stocks that fell out will be (NaN * NaN = NaN) or (0 * NaN = NaN) 
# if you .fillna(0) on the weights.
daily_pnl = (final_weights_t * returns_t).sum(axis=1)

# 7. (Recommended) Fill NaNs on weights *after* masking
# This makes your PnL calculation cleaner: 0 * return = 0
# It correctly gives zero PnL for stocks that fell out of the universe.
final_weights_t_zeroed = weights_t.where(mask).fillna(0)
daily_pnl = (final_weights_t_zeroed * returns_t).sum(axis=1)
```
# Task 2: Monte Carlo integration with comparison to analytic

import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent
import math
import os

# 1) Define function and interval
def f(x):
    return x**2

a, b = 0.0, 2.0  # integrate x^2 on [0,2]
true_value = (b**3 - a**3) / 3.0  # analytic integral of x^2 is x^3/3

# 2) Monte Carlo estimator
def monte_carlo_integral(f, a, b, samples=100_000, rng=None):
    """
    Simple Monte Carlo integration: E[f(U)]*(b-a), U~Uniform(a,b).
    Returns estimate and standard error approximation.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = rng.uniform(a, b, size=samples)
    fx = f(x)
    est = (b - a) * fx.mean()
    # Standard error estimate: sqrt(Var[f(U)]) * (b-a) / sqrt(N)
    se = (b - a) * fx.std(ddof=1) / math.sqrt(samples) if samples > 1 else float("nan")
    return est, se

# 3) Run experiments for different sample sizes
rng = np.random.default_rng(42)
Ns = [10**2, 10**3, 10**4, 10**5]
rows = []
for n in Ns:
    est, se = monte_carlo_integral(f, a, b, samples=n, rng=rng)
    abs_err = abs(est - true_value)
    rows.append((n, est, se, true_value, abs_err))

import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user
df = pd.DataFrame(rows, columns=["N", "MC estimate", "Std. error (est.)", "Analytic", "Abs. error"])
display_dataframe_to_user("Monte Carlo vs Analytic — f(x)=x^2 on [0,2]", df)

# 4) Try SciPy quad if available
quad_result = None
try:
    from scipy.integrate import quad
    q_val, q_err = quad(lambda x: x**2, a, b)
    quad_result = (q_val, q_err)
except Exception as e:
    quad_result = None

# 5) Plot function and shaded integral region (no explicit colors per plotting constraints)
x = np.linspace(-0.5, 2.5, 400)
y = f(x)
ix = np.linspace(a, b, 200)
iy = f(ix)

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2)              # function curve
ax.fill_between(ix, iy, alpha=0.3)      # area under curve on [a,b]
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([0, max(y) + 0.1])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.axvline(x=a, linestyle='--')
ax.axvline(x=b, linestyle='--')
ax.set_title(f'Інтегрування f(x) = x^2 від {a} до {b}')
ax.grid(True)
plt.show()

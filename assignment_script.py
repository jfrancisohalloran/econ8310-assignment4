import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

URL = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv"
OUTCOMES = ["retention_1", "retention_7"]
ROPE = 0.005
N_SAMPLES = 100_000

rng = np.random.default_rng(42)

def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def count_successes(df: pd.DataFrame, outcome: str) -> dict:
    grp = df.groupby("version")[outcome].agg(count="count", sum="sum")
    return {
        version: (int(row["count"]), int(row["sum"]))
        for version, row in grp.iterrows()
    }

def draw_posterior(successes: int, trials: int) -> np.ndarray:
    return beta.rvs(successes + 1, trials - successes + 1, size=N_SAMPLES, random_state=rng)

def summarize_distribution(samples: np.ndarray) -> tuple:
    mean = samples.mean()
    low, high = np.percentile(samples, [2.5, 97.5])
    return mean, low, high

df = load_data(URL)

balance_stats = df.groupby("version")["sum_gamerounds"].describe().reset_index()
print("\nBalance check on sum_gamerounds:")
print(balance_stats.to_string(index=False))
results = []
diff_samples = {}

for outcome in OUTCOMES:
    (n0, s0), (n1, s1) = count_successes(df, outcome)["gate_30"], count_successes(df, outcome)["gate_40"]
    p0 = draw_posterior(s0, n0)
    p1 = draw_posterior(s1, n1)
    diff = p1 - p0
    diff_samples[outcome] = diff
    
    mu0, low0, high0 = summarize_distribution(p0)
    mu1, low1, high1 = summarize_distribution(p1)
    mu_d, low_d, high_d = summarize_distribution(diff)
    p_gt0 = (diff > 0).mean()
    p_rope = ((diff > -ROPE) & (diff < ROPE)).mean()
    
    results.append({
        "outcome": outcome,
        "gate_30_mean": mu0,
        "gate_30_2.5%": low0,
        "gate_30_97.5%": high0,
        "gate_40_mean": mu1,
        "gate_40_2.5%": low1,
        "gate_40_97.5%": high1,
        "diff_mean": mu_d,
        "diff_2.5%": low_d,
        "diff_97.5%": high_d,
        "P(diff>0)": p_gt0,
        "P(|diff|<0.005)": p_rope
    })

summary_df = pd.DataFrame(results)
print("\nA/B Test Summary:")
print(summary_df.to_string(index=False))

for outcome, diff in diff_samples.items():
    plt.figure()
    plt.hist(diff, bins=50)
    plt.title(f"Posterior Δ for {outcome}")
    plt.xlabel("Δ retention rate")
    plt.ylabel("Frequency")
    plt.show()

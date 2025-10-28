import pandas as pd
import numpy as np

# ---------- Config ----------
CSV_PATH = "./Software.csv" 
K = 100                              # latent factors
CLAMP_MIN, CLAMP_MAX = 1.0, 5.0

# ---------- Load ----------
dtypes = {"item": str, "user": str, "rating": float, "timestamp": int}
df = pd.read_csv(CSV_PATH, dtype=dtypes, usecols=["item","user","rating","timestamp"]).dropna()

# (Optional) light filtering so SVD is stable; comment out if not needed
uc = df["user"].value_counts()
ic = df["item"].value_counts()
df = df[df["user"].isin(uc[uc >= 5].index)]
df = df[df["item"].isin(ic[ic >= 5].index)]

# ---------- Chronological split: last=test, prev=val, rest=train ----------
df = df.sort_values(["user","timestamp"])
def split_group(g):
    n = len(g)
    if n == 1:
        g = g.copy(); g["split"] = "test"; return g
    elif n == 2:
        g = g.copy(); g["split"] = "train"; g.iloc[-2, g.columns.get_loc("split")] = "val"; g.iloc[-1, g.columns.get_loc("split")] = "test"; return g
    g = g.copy(); g["split"] = "train"; g.iloc[-2, g.columns.get_loc("split")] = "val"; g.iloc[-1, g.columns.get_loc("split")] = "test"; return g

df = df.groupby("user", group_keys=False).apply(split_group)
train = df[df["split"]=="train"].copy()

# ---------- Encode TRAIN users/items ----------
users = sorted(train["user"].unique())
items = sorted(train["item"].unique())
u2i = {u: idx for idx, u in enumerate(users)}
i2j = {it: jdx for jdx, it in enumerate(items)}
m, n = len(users), len(items)

# ---------- Build R_train (NaNs for missing) ----------
R_train = np.full((m, n), np.nan)
for _, r in train.iterrows():
    R_train[u2i[r["user"]], i2j[r["item"]]] = float(r["rating"])

# ---------- Fill -> SVD -> R_hat ----------
global_mean = np.nanmean(R_train)
user_means = np.nanmean(R_train, axis=1)
user_means = np.where(np.isnan(user_means), global_mean, user_means)
R_filled = np.where(np.isnan(R_train), user_means[:, None], R_train)

U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)
R_hat = U[:, :K] @ np.diag(S[:K]) @ Vt[:K, :]

def clamp(x): return float(max(CLAMP_MIN, min(CLAMP_MAX, x)))

# ---------- Sample ONE user & ONE item from TRAIN space ----------
rng = np.random.default_rng()

existing = df[(df["user"].isin(users)) & (df["item"].isin(items))]

# pick a truly random row that exists in actual
idx = rng.integers(0, len(existing))
row = existing.iloc[idx]
u, it, actual = row["user"], row["item"], float(row["rating"])

# ---------- Predict from R_hat ----------
pred = clamp(R_hat[u2i[u], i2j[it]])

# ---------- (Optional) get actual if present in original data ----------
actual_row = df[(df["user"] == u) & (df["item"] == it)].head(1)
actual = float(actual_row["rating"].iloc[0]) if not actual_row.empty else np.nan
deviation = (pred - actual) if not np.isnan(actual) else np.nan

print("Sample prediction (from R_hat):")
print(f"  user      : {u}")
print(f"  item      : {it}")
print(f"  predicted : {pred:.3f}")
print(f"  actual    : {actual:.3f}" if not np.isnan(actual) else "  actual    : NaN (no rating found)")
print(f"  deviation : {deviation:.3f}" if not np.isnan(deviation) else "  deviation : NaN")
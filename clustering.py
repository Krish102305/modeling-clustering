"""Modeling - clustering: gym member segments (who gets the monthly $10 credit).
Run:  python clustering.py   (expects ../data/gym_members.csv)
Every number on the slides comes from this script.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(HERE, "gym_members.csv"))
print(f"members={len(df)}, cols={df.shape[1]}, mean extras spend=${df.monthly_extras_spend.mean():.2f}")

# Features: how a member uses the gym + who they are.
# Left out: member_id (just a label) and monthly_extras_spend (it is what the credit decision is judged on,
# so clustering on it would be circular; we use it afterwards to value the segments).
FEATURES = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month", "months_member"]

train, test = train_test_split(df, test_size=0.30, random_state=SEED)
print(f"train={len(train)}, test={len(test)}")
scaler = StandardScaler().fit(train[FEATURES])          # scaling fit on training members only
Xtr = scaler.transform(train[FEATURES])

print("\nk, within-group sum of squares (training):")
wss = {}
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xtr)
    wss[k] = km.inertia_
    drop = "" if k == 1 else f"  (drop from k-1: {(wss[k-1]-wss[k])/wss[k-1]:.1%})"
    print(f"  k={k}: {wss[k]:8.1f}{drop}")

# Test spread: do NOT refit. Use the centers fitted on the training members, assign each test member
# to its nearest center, and sum the squared distances (test members scaled with the TRAINING scaler).
Xte = scaler.transform(test[FEATURES])
wss_test = {}
for k in range(1, 9):
    centers_k = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xtr).cluster_centers_
    d2 = ((Xte[:, None, :] - centers_k[None, :, :]) ** 2).sum(axis=2)   # (n_test, k)
    wss_test[k] = float(d2.min(axis=1).sum())
print("\nk, test spread (sum of squared distances to nearest TRAINING center), drop from k-1:")
for k in range(1, 9):
    drop = "" if k == 1 else f"  (drop {(wss_test[k-1]-wss_test[k])/wss_test[k-1]:.1%})"
    print(f"  k={k}: {wss_test[k]:8.1f}{drop}")
print(f"Test spread k=1: {wss_test[1]:.1f}   k=4: {wss_test[4]:.1f}   ({(wss_test[1]-wss_test[4])/wss_test[1]:.1%} lower)")
print(f"Train spread k=1: {wss[1]:.1f}   k=4: {wss[4]:.1f}   ({(wss[1]-wss[4])/wss[1]:.1%} lower)")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(list(wss), list(wss.values()), "o-", color="#1f4e9c")
axs[0].set_title("Training members (n=%d)" % len(train))
axs[1].plot(list(wss_test), list(wss_test.values()), "o-", color="#c0392b")
axs[1].set_title("Test members (n=%d), training centers" % len(test))
for a in axs:
    a.set_xlabel("Number of groups (k)"); a.set_ylabel("Within-group spread"); a.axvline(4, ls="--", color="gray", lw=1)
fig.suptitle("Elbow plot: training vs test")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "elbow.png"), dpi=200)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # chosen from the elbow (see printed drops)
v1 = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Xtr)
print(f"\nchosen k = {K}")

# Assign every member (train + test) to a segment
df["segment"] = v1.predict(scaler.transform(df[FEATURES]))
df["is_test"] = df.index.isin(test.index)

centers = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES)
centers["n_members"] = df.groupby("segment").size().values
centers["avg_extras_spend"] = df.groupby("segment")["monthly_extras_spend"].mean().values
centers["share_spend_under_20"] = df.groupby("segment")["monthly_extras_spend"].apply(lambda s: (s < 20).mean()).values
pd.set_option("display.width", 200)
print("\nSegment centroids (original units):")
print(centers.round(1).to_string())

# Finance assumptions, applied member by member
def credit_net(spend):           # net to the gym of giving this member the $10 credit
    return np.where(spend < 20, 5, -10)
df["credit_net"] = credit_net(df["monthly_extras_spend"].values)

base_total = df["credit_net"].sum()
print(f"\nBASELINE (credit to everyone): net ${base_total:,.0f}/month over {len(df)} members (${base_total/len(df):.2f} per member)")
print("Net of giving the credit, by segment (sum over members in the segment):")
seg_net = df.groupby("segment")["credit_net"].agg(["sum", "mean", "size"])
print(seg_net.round(2).to_string())

# Rule: give the credit to segments where the credit nets positive on the TRAINING members of that segment
tr = df[~df.is_test]
chosen = [s for s, g in tr.groupby("segment") if g["credit_net"].sum() > 0]
print(f"\nsegments that get the credit (positive net on training members): {chosen}")
v1_total = df.loc[df.segment.isin(chosen), "credit_net"].sum()
print(f"v1 (credit to chosen segments only): net ${v1_total:,.0f}/month  (${v1_total/len(df):.2f} per member)")
print(f"improvement over baseline: ${v1_total-base_total:,.0f}/month = ${(v1_total-base_total)*12:,.0f}/yr")
for lab, mask in [("train", ~df.is_test), ("test", df.is_test)]:
    b = df.loc[mask, "credit_net"].sum()
    v = df.loc[mask & df.segment.isin(chosen), "credit_net"].sum()
    print(f"  {lab:5s} ({mask.sum()} members): baseline ${b:,.0f}, v1 ${v:,.0f}")

# Decision function
def assign(member):
    """member: dict with age, visits_per_month, avg_visit_minutes, classes_per_month, months_member."""
    x = scaler.transform(pd.DataFrame([member])[FEATURES])
    seg = int(v1.predict(x)[0])
    return seg, ("credit" if seg in chosen else "no credit")

new = dict(age=30, visits_per_month=4, avg_visit_minutes=42, classes_per_month=1, months_member=5)
print(f"\nNew member {new} (extras spend $10, not used for segmentation): ->", assign(new))

# Save segments for reference
df.to_csv(os.path.join(OUT, "gym_with_segments.csv"), index=False)

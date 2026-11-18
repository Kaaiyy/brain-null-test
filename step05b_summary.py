import numpy as np, pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv("results/brain_many.csv")
d = d[d.subject != 4]  # subject 04 was used in step 04
print("subjects:", d.subject.nunique(), " slices:", d.groupby(["subject", "z"]).ngroups)

KEYS = ["subject", "z", "bias", "noise"]
def wide(t):
    x = d[d.tissue == t]
    out = pd.DataFrame({"dice": x.groupby(KEYS)[f"dice_{t.lower()}"].first()})
    for st in ("area", "share", "bnd"):
        s = x[x.stat == st].set_index(KEYS)
        out[f"{st}_real"] = s.real; out[f"{st}_r"] = s.ratio
        out[f"{st}_pHi"] = s.p_hi; out[f"{st}_pLo"] = s.p_lo
    return out.reset_index()

pd.set_option("display.width", 200)
print("\nTABLE A: null test per condition (detect = p<=0.05 vs 19 scrambles)")
for t in ("WM", "GM"):
    w = wide(t)
    g = w.groupby(["bias", "noise"]).agg(
        n=("dice", "size"), dice=("dice", "mean"),
        share_r=("share_r", "median"),
        share_det=("share_pHi", lambda s: (s <= 0.05).mean()),
        bnd_r=("bnd_r", "median"),
        bnd_smooth=("bnd_pLo", lambda s: (s <= 0.05).mean()),
        bnd_rough=("bnd_pHi", lambda s: (s <= 0.05).mean()),
        area_r=("area_r", "median"))
    print(t); print(g.round(2).to_string())

print("\nTABLE B: Spearman with true Dice (pooled; within-condition mean; subject-bootstrap 95% CI of pooled)")
rng = np.random.default_rng(0)
for t in ("WM", "GM"):
    w = wide(t)
    subs = w.subject.unique()
    parts = {s: w[w.subject == s] for s in subs}
    rows = []
    for c in ("share_real", "share_r", "area_real", "bnd_real", "bnd_r"):
        pooled = spearmanr(w[c], w.dice)[0]
        within = np.nanmean([spearmanr(g[c], g.dice)[0] for _, g in w.groupby(["bias", "noise"])])
        bs = []
        for _ in range(300):
            b = pd.concat([parts[s] for s in rng.choice(subs, len(subs))])
            bs.append(spearmanr(b[c], b.dice)[0])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(measure=c, pooled=pooled, within_cond=within, ci_lo=lo, ci_hi=hi))
    print(t); print(pd.DataFrame(rows).round(2).to_string(index=False))

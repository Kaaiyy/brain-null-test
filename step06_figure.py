import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"      # WM, GM
INK, SUB, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": SUB, "axes.labelcolor": INK,
                     "xtick.color": SUB, "ytick.color": SUB, "text.color": INK})

d = pd.read_csv("results/brain_many.csv")
d = d[d.subject != 4]
KEYS = ["subject", "z", "bias", "noise"]
def wide(t):
    x = d[d.tissue == t]
    out = pd.DataFrame({"dice": x.groupby(KEYS)[f"dice_{t.lower()}"].first()})
    for st in ("share", "bnd"):
        s = x[x.stat == st].set_index(KEYS)
        out[f"{st}_real"] = s.real; out[f"{st}_pHi"] = s.p_hi
    return out.reset_index()
W, G = wide("WM"), wide("GM")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
conds = [(b, n) for b in (0.0, 0.4) for n in (0, 3, 9)]
xs = np.arange(len(conds)); w = 0.38
for off, df, col, nm in ((-w/2, W, BLUE, "white matter"), (w/2, G, ORANGE, "grey matter")):
    rate = [(df[(df.bias == b) & (df.noise == n)].share_pHi <= 0.05).mean() for b, n in conds]
    ax[0].bar(xs + off, rate, w * 0.92, color=col, label=nm)
ax[0].set_xticks(xs)
ax[0].set_xticklabels([f"b{int(b*100)}\nn{n}" for b, n in conds])
ax[0].set_ylim(0, 1); ax[0].set_ylabel("share of slices where the null test fires")
ax[0].set_xlabel("bias % / noise %")
ax[0].set_title("A. Connected-share vs scrambled copies", loc="left", fontsize=10)
ax[0].legend(frameon=False, loc="upper left")

for a, df, col, nm, lab in ((ax[1], W, BLUE, "white matter", "B"), (ax[2], G, ORANGE, "grey matter", "C")):
    a.scatter(df.bnd_real, df.dice, s=14, color=col, alpha=0.55, linewidths=0)
    rho = pd.Series(df.bnd_real).corr(df.dice, method="spearman")
    a.set_xlabel("edge length per area (free baseline, no null)")
    a.set_ylabel("true Dice")
    a.set_title(f"{lab}. {nm}: baseline tracks Dice (rho {rho:.2f})", loc="left", fontsize=10)
for a in ax:
    a.grid(True, color=GRID, linewidth=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("figures/06_brain_summary.png", dpi=130)
print("saved figures/06_brain_summary.png")

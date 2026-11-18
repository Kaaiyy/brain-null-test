import os, sys, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sim import simulate, tissue3
from nulltools import iaaft, stats

def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan

def seg4(img, seed=0):
    x = img.reshape(-1, 1)
    rng = np.random.default_rng(seed)
    sub = x[rng.choice(len(x), size=min(20000, len(x)), replace=False)]
    km = KMeans(n_clusters=4, n_init=3, random_state=seed).fit(sub)
    order = np.argsort(km.cluster_centers_.ravel())  # bg, CSF, GM, WM
    rank = np.empty(4, int); rank[order] = [0, 1, 2, 3]
    return rank[km.predict(x)].reshape(img.shape)

def run(lab2d, n_surr=19, out="results/brain_null.csv"):
    truth = tissue3(lab2d)
    ys, xs = np.where(truth > 0)
    p = 10
    sl = (slice(max(ys.min() - p, 0), ys.max() + p), slice(max(xs.min() - p, 0), xs.max() + p))
    lab2d = lab2d[sl]; truth = truth[sl]
    rows = []
    for b in [0.0, 0.2, 0.4]:
        for n in [0, 3, 9]:
            img = simulate(lab2d, b, n, seed=0)
            seg = seg4(img)
            dg = dice(seg == 2, truth == 2); dw = dice(seg == 3, truth == 3)
            sur = []
            for i in range(n_surr):
                sur.append({k: stats(seg4(iaaft(img, seed=i + 1), seed=i), k) for k in (2, 3)})
            for k, nm in ((2, "GM"), (3, "WM")):
                real = stats(seg, k)
                for st in ("area", "share", "bnd"):
                    sv = np.array([s[k][st] for s in sur], float)
                    r = real[st]
                    p_hi = (1 + np.nansum(sv >= r)) / (n_surr + 1)
                    p_lo = (1 + np.nansum(sv <= r)) / (n_surr + 1)
                    rows.append(dict(bias=b, noise=n, tissue=nm, stat=st, real=r,
                                     surr_mean=np.nanmean(sv), ratio=r / np.nanmean(sv),
                                     p_hi=p_hi, p_lo=p_lo, dice_gm=dg, dice_wm=dw))
            print("done bias", b, "noise", n, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df

if __name__ == "__main__":
    import brainweb
    f = brainweb.get_file("subject_04.bin.gz", brainweb.LINKS["subject_04.bin.gz"],
                          cache_dir=os.path.abspath("data"))
    lab = brainweb.load_file(f)
    df = run(lab[lab.shape[0] // 2])
    pd.set_option("display.width", 200)
    print(df[df.tissue == "WM"][["bias", "noise", "stat", "real", "surr_mean", "ratio", "p_hi", "p_lo"]]
          .round(3).to_string(index=False))

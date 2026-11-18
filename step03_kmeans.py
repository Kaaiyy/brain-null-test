import os, numpy as np, pandas as pd, brainweb
from sklearn.cluster import KMeans
from sim import simulate, tissue3

def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan

def kmeans_seg(img, mask, seed=0):
    x = img[mask].reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=5, random_state=seed).fit(x)
    order = np.argsort(km.cluster_centers_.ravel())  # dark -> bright = CSF, GM, WM
    rank = np.empty(3, int); rank[order] = [1, 2, 3]
    seg = np.zeros(img.shape, int)
    seg[mask] = rank[km.labels_]
    return seg

f = brainweb.get_file("subject_04.bin.gz", brainweb.LINKS["subject_04.bin.gz"],
                      cache_dir=os.path.abspath("data"))
lab = brainweb.load_file(f)
lab2d = lab[lab.shape[0] // 2]
truth = tissue3(lab2d)
mask = truth > 0

rows = []
for b in [0.0, 0.2, 0.4]:
    for n in [0, 3, 9]:
        for s in range(5):
            seg = kmeans_seg(simulate(lab2d, b, n, seed=s), mask)
            d = {k: dice(seg == k, truth == k) for k in (1, 2, 3)}
            rows.append(dict(bias=b, noise=n, seed=s, dice_csf=d[1], dice_gm=d[2],
                             dice_wm=d[3], dice_mean=np.mean(list(d.values()))))
df = pd.DataFrame(rows)
df.to_csv("results/brain_kmeans.csv", index=False)
print(df.groupby(["bias", "noise"])[["dice_csf", "dice_gm", "dice_wm", "dice_mean"]]
        .mean().round(3).to_string())

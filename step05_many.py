import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sim import simulate, tissue3
from nulltools import iaaft, stats
from step04_null import seg4, dice

OUT = "results/brain_many.csv"
CONDS = [(b, n) for b in (0.0, 0.4) for n in (0, 3, 9)]

def work(subj, z, lab2d, n_surr=19):
    truth = tissue3(lab2d)
    ys, xs = np.where(truth > 0)
    p = 10
    sl = (slice(max(ys.min() - p, 0), ys.max() + p), slice(max(xs.min() - p, 0), xs.max() + p))
    lab2d = lab2d[sl]; truth = truth[sl]
    rows = []
    for b, n in CONDS:
        img = simulate(lab2d, b, n, seed=int(subj) * 1000 + int(z))
        seg = seg4(img)
        dg = dice(seg == 2, truth == 2); dw = dice(seg == 3, truth == 3)
        sur = [{k: stats(seg4(iaaft(img, seed=i + 1), seed=i), k) for k in (2, 3)}
               for i in range(n_surr)]
        for k, nm in ((2, "GM"), (3, "WM")):
            real = stats(seg, k)
            for st in ("area", "share", "bnd"):
                sv = np.array([s[k][st] for s in sur], float)
                r = real[st]
                rows.append(dict(subject=subj, z=z, bias=b, noise=n, tissue=nm, stat=st,
                                 real=r, surr_mean=np.nanmean(sv), ratio=r / np.nanmean(sv),
                                 p_hi=(1 + np.nansum(sv >= r)) / (n_surr + 1),
                                 p_lo=(1 + np.nansum(sv <= r)) / (n_surr + 1),
                                 dice_gm=dg, dice_wm=dw))
    return rows

def append(rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)

if __name__ == "__main__":
    import brainweb
    os.makedirs("results", exist_ok=True)
    files = brainweb.get_files(cache_dir=os.path.abspath("data"))
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT)
        done = set(zip(d.subject, d.z))
    jobs = []
    for f in files:
        subj = int(os.path.basename(f).split("_")[1].split(".")[0])
        vol = brainweb.load_file(f)
        t = (tissue3(vol[:, ::4, ::4]) > 0).any(axis=(1, 2))
        zs = np.where(t)[0]
        for frac in (0.4, 0.6):
            z = int(zs.min() + frac * (zs.max() - zs.min()))
            if (subj, z) not in done:
                jobs.append((subj, z, np.array(vol[z])))
        del vol
    print(len(jobs), "slices to run", flush=True)
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    k = 0
    for rows in Parallel(n_jobs=nj, return_as="generator_unordered")(
            delayed(work)(s, z, a) for s, z, a in jobs):
        append(rows); k += 1
        print("finished", k, "of", len(jobs), flush=True)

import os, numpy as np, brainweb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim import simulate

f = brainweb.get_file("subject_04.bin.gz", brainweb.LINKS["subject_04.bin.gz"],
                      cache_dir=os.path.abspath("data"))
lab = brainweb.load_file(f)
lab2d = lab[lab.shape[0] // 2]

fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for i, b in enumerate([0.0, 0.2, 0.4]):
    for j, n in enumerate([0, 3, 9]):
        ax[i, j].imshow(simulate(lab2d, b, n), cmap="gray", vmin=0, vmax=1)
        ax[i, j].set_title(f"bias {int(b*100)}%  noise {n}%")
        ax[i, j].axis("off")
plt.tight_layout()
plt.savefig("figures/02_simulated.png", dpi=110)
print("saved figures/02_simulated.png")

import os, numpy as np, brainweb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = os.path.abspath("data")
os.makedirs(cache, exist_ok=True)
name = "subject_04.bin.gz"
f = brainweb.get_file(name, brainweb.LINKS[name], cache_dir=cache)
lab = brainweb.load_file(f)
print("shape", lab.shape, "dtype", lab.dtype)
vals, n = np.unique(lab, return_counts=True)
for v, c in zip(vals, n):
    print("label", v, "voxels", c)

z = lab.shape[0] // 2
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(lab[z], cmap="nipy_spectral"); ax[0].set_title("axis 0 slice")
ax[1].imshow(lab[:, lab.shape[1] // 2], cmap="nipy_spectral"); ax[1].set_title("axis 1 slice")
ax[2].imshow(lab[:, :, lab.shape[2] // 2], cmap="nipy_spectral"); ax[2].set_title("axis 2 slice")
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("figures/01_labels.png", dpi=120)
print("saved figures/01_labels.png")

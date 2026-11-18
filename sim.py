import numpy as np
from scipy.ndimage import gaussian_filter

MEAN = {0: 0.0, 1: 0.15, 2: 0.55, 3: 0.80}  # 1=CSF 2=GM 3=WM

def tissue3(lab):
    c = np.rint(np.asarray(lab, float) / 16).astype(int)
    return np.where(np.isin(c, [1, 2, 3]), c, 0)

def clean_image(lab, blur=1.0):
    t = tissue3(lab)
    img = np.zeros(t.shape, float)
    for k, v in MEAN.items():
        img[t == k] = v
    return gaussian_filter(img, blur)

def bias_field(shape, level, seed=0):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=shape), sigma=min(shape) / 4)
    f = f / np.abs(f).max()
    return 1 + (level / 2) * f

def add_noise(img, pct, seed=0):
    rng = np.random.default_rng(seed)
    return img + rng.normal(0, pct / 100 * MEAN[3], img.shape)

def simulate(lab, bias=0.0, noise=0.0, seed=0):
    img = clean_image(lab) * bias_field(lab.shape, bias, seed)
    return add_noise(img, noise, seed + 1)

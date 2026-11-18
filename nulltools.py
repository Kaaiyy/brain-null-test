import numpy as np
from scipy import ndimage as ndi

def iaaft(x, n_iter=20, seed=0):
    rng = np.random.default_rng(seed)
    srt = np.sort(x.ravel())
    amp = np.abs(np.fft.fft2(x))
    y = rng.permutation(x.ravel()).reshape(x.shape)
    for _ in range(n_iter):
        Y = np.fft.fft2(y)
        y = np.real(np.fft.ifft2(amp * np.exp(1j * np.angle(Y))))
        ranks = np.argsort(np.argsort(y.ravel()))
        y = srt[ranks].reshape(x.shape)
    return y

def stats(seg, k):
    m = seg == k
    area = m.sum()
    if area == 0:
        return dict(area=0.0, share=np.nan, bnd=np.nan)
    lab, n = ndi.label(m)
    big = np.bincount(lab.ravel())[1:].max()
    bnd = (m & ~ndi.binary_erosion(m)).sum()
    return dict(area=area / m.size, share=big / area, bnd=bnd / area)

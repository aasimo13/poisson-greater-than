"""Regenerate the h tables from core.h and compare with the shipped files.

h(mu) = smallest nu with two point minimax error 1% between Poisson(mu) and Poisson(nu).
Grids (mu):  hgrid.npy      linspace(0,2,401) + linspace(2.01,40,3800) + linspace(40.05,220,3600)
             hgrid_ext.npy  the same plus linspace(220.05,420,4000)
Each file is a 2 x n array: row 0 = mu grid, row 1 = h(mu).

Usage: python3 src/make_hgrid.py [--write]    full regeneration; compares with data/ (--write overwrites data/)
       python3 src/make_hgrid.py --sample N    recompute only N evenly spaced entries of hgrid_ext.npy and compare
Runtime: see README (full run is several minutes, the sample is seconds).
"""
import sys, os, time
import numpy as np
from core import h

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

base = [np.linspace(0, 2, 401), np.linspace(2.01, 40, 3800), np.linspace(40.05, 220, 3600)]
ext = base + [np.linspace(220.05, 420, 4000)]


def table(parts, cache):
    g = np.concatenate(parts)
    hv = np.empty_like(g)
    for i, m in enumerate(g):
        key = round(float(m), 9)
        if key not in cache:
            cache[key] = h(float(m))
        hv[i] = cache[key]
    return np.vstack([g, hv])


def compare(name, new):
    old = np.load(os.path.join(DATA, name))
    if old.shape != new.shape:
        print(f"{name}: SHAPE MISMATCH shipped {old.shape} regenerated {new.shape}")
        return False
    dg = np.abs(old[0] - new[0]).max(); dh = np.abs(old[1] - new[1]).max()
    ok = dg < 1e-12 and dh < 1e-8
    print(f"{name}: shape {new.shape}, max |grid diff| {dg:.2e}, max |h diff| {dh:.2e} -> {'MATCH' if ok else 'MISMATCH'}")
    return ok


def sample_check(n):
    old = np.load(os.path.join(DATA, "hgrid_ext.npy"))
    idx = np.unique(np.linspace(0, old.shape[1] - 1, n).astype(int))
    dh = max(abs(h(float(old[0, i])) - old[1, i]) for i in idx)
    ok = dh < 1e-8
    print(f"sample of {len(idx)} entries of hgrid_ext.npy: max |h diff| {dh:.2e} -> {'MATCH' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    t0 = time.time()
    if "--sample" in sys.argv:
        sys.exit(0 if sample_check(int(sys.argv[sys.argv.index("--sample") + 1])) else 1)
    cache = {}
    t_base = table(base, cache)
    t_ext = table(ext, cache)
    if "--write" in sys.argv:
        np.save(os.path.join(DATA, "hgrid.npy"), t_base)
        np.save(os.path.join(DATA, "hgrid_ext.npy"), t_ext)
        print("wrote data/hgrid.npy and data/hgrid_ext.npy")
    ok = compare("hgrid.npy", t_base) & compare("hgrid_ext.npy", t_ext)
    print(f"{'ALL MATCH' if ok else 'MISMATCH'} ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)

"""Recompute the budget of each stored best design for the molecular and feedback results.

Prints one line per design: label, recomputed budget. Each design is the best found by mc.py
(results/mc_*.jsonl). mc.py reads ISI and FEEDBACK from the environment at import, so each
configuration reloads it.
Usage: python3 src/mc_check.py [quick|all]
"""
import os, sys, importlib
import numpy as np

CASES = [  # label, ISI, FEEDBACK, receiver, x, y, quick?
    ("one type, banded, no spillover", 0, "none", "shared", [98.2976, 4.6013, 0, 28.7852], [28.7852, 0, 4.6013, 98.2976], True),
    ("two types cancelling, no spillover", 0, "none", "cancel", [0.028, 6.947, 34.331, 83.502], [0.037, 16.453, 54.808, 118.281], True),
    ("two types counted separately, no spillover", 0, "none", "two", [0.001, 4.701, 22.414, 52.805], [0.001, 5.005, 23.512, 53.561], True),
    ("one type, threshold, 10% spillover", 0.1, "none", "shared", [360.4, 240.307, 120.214, 0.12], [0.12, 120.214, 240.307, 360.4], True),
    ("two types cancelling, 10% spillover", 0.1, "none", "cancel", [0.126, 119.909, 248.223, 399.231], [0.143, 116.723, 256.513, 418.64], True),
    ("one type, banded, 1% spillover, previous answer known", 0.01, "answer", "shared", [159.665, 11.954, 0.003, 52.963], [52.963, 0.003, 11.984, 159.671], True),
    ("one type, banded, 1% spillover, previous pair known", 0.01, "genie", "shared", [143.708, 9.938, 0.003, 46.612], [46.484, 0.003, 9.926, 143.646], True),
    ("two types counted separately, 10% spillover", 0.1, "none", "two", [0.002, 32.976, 81.325, 144.041], [0.002, 34.397, 83.031, 141.848], False),
]

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mc = None
    for label, isi, fb, kind, x, y, q in CASES:
        if mode == "quick" and not q:
            continue
        os.environ["ISI"] = str(isi); os.environ["FEEDBACK"] = fb
        mc = importlib.import_module("mc") if mc is None else importlib.reload(mc)
        P = mc.budget(np.array(x, float), np.array(y, float), kind, rel=1e-5)
        print(f"{label}: {P:.2f}", flush=True)

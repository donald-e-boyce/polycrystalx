"""Generate random Voronoi microstructures and save them to jobs/data/.

Usage
-----
    python gen_voronoi.py <num_grains> <num_micros>

Each microstructure is saved as jobs/data/micro-<num_grains>g-<i>.npz,
where i runs from 1 to num_micros.  An existing file is skipped so that
previously generated microstructures are not overwritten.
"""
import argparse
import os

import numpy as np
from polycrystal.microstructure.voronoi import Voronoi

UNIT_CUBE = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

DATA_DIR = os.path.join(os.path.dirname(__file__), "jobs", "data")


def filename(num_grains, index):
    return os.path.join(DATA_DIR, f"micro-{num_grains}g-{index}.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random Voronoi microstructures on the unit cube."
    )
    parser.add_argument("num_grains", type=int, help="number of grains")
    parser.add_argument("num_micros", type=int, help="number of microstructures to generate")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    for i in range(1, args.num_micros + 1):
        fname = filename(args.num_grains, i)
        if os.path.exists(fname):
            print(f"skipping {os.path.basename(fname)} (already exists)")
            continue
        Voronoi.random_voronoi(args.num_grains, UNIT_CUBE, fname=fname)
        print(f"wrote {os.path.basename(fname)}")


if __name__ == "__main__":
    main()

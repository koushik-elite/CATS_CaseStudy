#!/usr/bin/env python3
"""
Describe numpy files in a directory.

Usage:
  python scripts/describe_numpy.py /path/to/folder

This script scans for .npy and .npz files and prints filename, dtype, shape,
deep shape (for nested arrays/objects), number of elements and estimated memory.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from typing import Any, Tuple


if __name__ == '__main__':
    arr = np.load("/home/koushik/CATS_CaseStudy/results/nift_96_CATSF_custom_sl96_pl96_dm256_nh32_dl3_df512_qiFalse_0/true.npy", allow_pickle=True)
    print(type(arr))
    print(arr.shape)
    print(arr.dtype)
    print(arr)

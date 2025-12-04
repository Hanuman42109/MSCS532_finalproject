#!/usr/bin/env python3
"""
matmul_demo.py
Simple harness comparing three approaches:
- pure Python triple loop (small sizes)
- numpy.dot (vectorized, BLAS-backed)
- blocked multiplication using numpy slices (cache-friendly)
"""
import numpy as np
import time
import argparse

def python_matmul(A, B):
    n = A.shape[0]
    C = [[0.0]*n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Ai[k] * B[k,j]
            C[i][j] = s
    return np.array(C)

def blocked_matmul(A, B, block=64):
    n = A.shape[0]
    C = np.zeros((n,n))
    for ii in range(0, n, block):
        for jj in range(0, n, block):
            for kk in range(0, n, block):
                A_block = A[ii:ii+block, kk:kk+block]
                B_block = B[kk:kk+block, jj:jj+block]
                C[ii:ii+block, jj:jj+block] += A_block.dot(B_block)
    return C

def time_func(f, *args, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = f(*args)
        times.append(time.perf_counter() - t0)
    return min(times), times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256, help="matrix dimension")
    parser.add_argument("--block", type=int, default=64, help="block size for blocked matmul")
    args = parser.parse_args()

    n = args.n
    np.random.seed(0)
    A = np.random.rand(n,n).astype(np.float64)
    B = np.random.rand(n,n).astype(np.float64)

    print(f"Running n={n}, block={args.block}")
    # Warmup
    _ = np.dot(A,B)

    # NumPy dot
    tmin, trials = time_func(np.dot, A, B, repeats=3)
    print("numpy.dot: min {:.6f}s, trials {}".format(tmin, trials))

    # Blocked
    tmin_block, trials_block = time_func(blocked_matmul, A, B, args.block, repeats=3)
    print("blocked_matmul: min {:.6f}s, trials {}".format(tmin_block, trials_block))

    # Python loops (only for small n)
    if n <= 128:
        An = A[:128,:128]; Bn = B[:128,:128]
        tmin_py, trials_py = time_func(python_matmul, An, Bn, repeats=1)
        print("python_loop (128x128): {:.6f}s".format(tmin_py))

    # Sanity-check correctness (for a small sub-block)
    if n >= 128:
        C_ref = np.dot(A,B)
        C_block = blocked_matmul(A,B,args.block)
        print("blocked correctness:", np.allclose(C_block[:128,:128], C_ref[:128,:128], atol=1e-8))

if __name__ == "__main__":
    main()
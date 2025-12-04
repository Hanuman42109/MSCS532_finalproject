# HPC Matrix Multiplication Optimization Demo

This repository demonstrates three approaches to matrix multiplication and compares their performance in a High-Performance Computing (HPC) context:

1.  **Pure Python Triple Loop**
2.  **Vectorized NumPy (`numpy.dot`)**
3.  **Cache‑Friendly Blocked Matrix Multiplication**

The goal is to highlight how algorithmic choices, data locality, and vectorization significantly impact runtime on modern hardware.

------------------------------------------------------------------------

## 📌 Features

-   Pure Python implementation for baseline comparison
-   BLAS‑backed NumPy vectorized multiplication
-   Blocked (tiled) multiplication to improve cache locality
-   Performance benchmarking harness
-   Correctness checks for blocked implementation

------------------------------------------------------------------------

## 🚀 Running the Script

### **Prerequisites**

- Python 3.8+
- NumPy

### **Install requirements**

``` bash
pip install numpy
```

### **Run the demo**

``` bash
python matmul_demo.py --n 256 --block 64
```

### **Arguments**

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n`     | 256     | Matrix dimension (creates n×n matrices) |
| `--block` | 64      | Tile size for blocked multiplication |

------------------------------------------------------------------------

## 📊 Example Output

### For **n = 128, block = 64**

    numpy.dot:        ~0.00010s
    blocked_matmul:   ~0.00013s
    python_loop:      ~0.433s
    correctness:      True

### For **n = 256, block = 64**

    numpy.dot:        ~0.00051s
    blocked_matmul:   ~0.00138s
    correctness:      True

------------------------------------------------------------------------

## 📈 Performance Summary

| Matrix Size | Method       | Time (s)  |
|-------------|--------------|-----------|
| 128×128     | NumPy        | 0.00010   |
| 128×128     | Blocked      | 0.00013   |
| 128×128     | Python Loop  | 0.433     |
| 256×256     | NumPy        | 0.00051   |
| 256×256     | Blocked      | 0.00138   |

------------------------------------------------------------------------

## 🧠 Key Insights

-   NumPy (`numpy.dot`) is extremely fast due to optimized BLAS backends.
-   Blocked multiplication improves cache locality but still incurs python slicing overhead.
-   Pure Python loops are thousands of times slower and unsuitable for HPC workloads.
-   Results support findings from empirical HPC performance studies: **data structure, caching, and vectorization dominate performance.**
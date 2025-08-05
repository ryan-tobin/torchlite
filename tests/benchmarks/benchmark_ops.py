"""
Benchmarks for tensor operations.
Run with: python -m pytest tests/benchmarks/ -v
"""
import time
import numpy as np
import pytest
from torchlite import Tensor

class BenchmarkTensorOps:
    """Benchmark tensor operations."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.sizes = [10, 100, 1000]
    
    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_addition_speed(self, size):
        """Benchmark tensor addition."""
        x = Tensor(np.random.randn(size, size))
        y = Tensor(np.random.randn(size, size))
        
        start = time.time()
        for _ in range(100):
            z = x + y
        elapsed = time.time() - start
        
        ops_per_second = 100 / elapsed
        print(f"\nAddition ({size}x{size}): {ops_per_second:.2f} ops/sec")
        assert ops_per_second > 10  # Minimum performance requirement
    
    @pytest.mark.parametrize("size", [10, 100, 500])
    def test_matmul_speed(self, size):
        """Benchmark matrix multiplication."""
        x = Tensor(np.random.randn(size, size))
        y = Tensor(np.random.randn(size, size))
        
        start = time.time()
        for _ in range(10):
            z = x @ y
        elapsed = time.time() - start
        
        ops_per_second = 10 / elapsed
        gflops = (2 * size**3 * ops_per_second) / 1e9
        print(f"\nMatMul ({size}x{size}): {gflops:.2f} GFLOPS")
        assert gflops > 0.1  # Minimum performance requirement
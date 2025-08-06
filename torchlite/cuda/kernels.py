"""
CUDA kernels for GPU acceleration.
Requires CuPy installation.

!! NOT AVAILABLE ON macOS !!
"""

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

if CUDA_AVAILABLE:
    cuda_add = cp.ElementwiseKernel(
        "float32 x, float32 y",
        "float32 z",
        "z = x + y",
        "cuda_add")

    cuda_multiply = cp.ElementwiseKernel(
        "float32 x, float32 y", "float32 z", "z = x * y", "cuda_multiply"
    )

    # Activation functions
    cuda_relu = cp.ElementwiseKernel(
        "float32 x",
        "float32 y",
        "y = fmaxf(0.0f, x)",
        "cuda_relu")

    cuda_sigmoid = cp.ElementwiseKernel(
        "float32 x",
        "float32 y",
        "y = 1.0f / (1.0f + expf(-x))",
        "cuda_sigmoid")

    cuda_tanh = cp.ElementwiseKernel(
        "float32 x",
        "float32 y",
        "y = tanhf(x)",
        "cuda_tanh")

    def cuda_matmul(a, b):
        """GPU-accelerated matrix multiplcation."""
        return cp.matmul(a, b)

    # Custom CUDA kernel for more complex operations
    conv2d_kernel = cp.RawKernel(
        r"""
    extern "C" __global__
    void conv2d_forward(const float* input, const float* weight, float* output,
                       int batch_size, int in_channels, int out_channels,
                       int in_height, int in_width, int kernel_size,
                       int out_height, int out_width, int stride, int pad) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_threads = batch_size * out_channels * out_height * out_width;

        if (idx >= total_threads) return;

        int w_out = idx % out_width;
        int h_out = (idx / out_width) % out_height;
        int c_out = (idx / (out_width * out_height)) % out_channels;
        int batch = idx / (out_width * out_height * out_channels);

        float sum = 0.0f;

        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h_out * stride - pad + kh;
                    int w_in = w_out * stride - pad + kw;

                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((batch * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        output[idx] = sum;
    }
    """,
        "conv2d_forward",
    )

else:

    def _not_available(*args, **kwargs):
        raise RuntimeError(
            "CUDA not available. Please install CuPY. Please note: CuPY is not available on MacOS"
        )

    cuda_add = _not_available
    cuda_multiply = _not_available
    cuda_matmul = _not_available
    cuda_relu = _not_available
    cuda_sigmoid = _not_available
    cuda_tanh = _not_available

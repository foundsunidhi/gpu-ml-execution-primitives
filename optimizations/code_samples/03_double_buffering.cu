// Illustrates ping-pong buffering conceptually

__global__ void double_buffer(float* A) {
    __shared__ float buf0[256];
    __shared__ float buf1[256];

    int tid = threadIdx.x;

    buf0[tid] = A[tid];       // load tile 0
    __syncthreads();

    // Compute on buf0 while buf1 is being prepared
    float val = buf0[tid] * 2.0f;

    buf1[tid] = A[tid + 256]; // load tile 1
    __syncthreads();

    val += buf1[tid] * 3.0f;
    A[tid] = val;
}

// Demonstrates vectorized global memory loads

__global__ void vector_load(const float4* A, float4* B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = A[idx];   // loads 16 bytes at once
    B[idx] = v;
}

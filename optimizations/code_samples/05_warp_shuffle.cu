// Warp-level reduction without shared memory

__global__ void warp_reduce(float* data) {
    float val = data[threadIdx.x];

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x == 0)
        data[blockIdx.x] = val;
}

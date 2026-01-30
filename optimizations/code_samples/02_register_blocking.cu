// Each thread computes multiple output elements (register tile)

__global__ void register_blocked(float* C) {
    float r0 = 0.0f;
    float r1 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 128; i++) {
        float a = 1.0f;
        float b0 = 2.0f;
        float b1 = 3.0f;
        r0 += a * b0;
        r1 += a * b1;
    }

    C[threadIdx.x * 2]     = r0;
    C[threadIdx.x * 2 + 1] = r1;
}

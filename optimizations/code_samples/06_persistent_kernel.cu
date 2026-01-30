// Simplified persistent kernel pattern

__global__ void persistent(float* work, int* counter) {
    while (true) {
        int idx = atomicAdd(counter, 1);
        if (idx >= 1024) return;

        work[idx] *= 2.0f;
    }
}

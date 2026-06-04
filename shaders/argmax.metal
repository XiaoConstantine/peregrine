#include <metal_stdlib>
using namespace metal;

constant uint ARGMAX_THREADS = 256;

kernel void argmax_pairs_f32_u32(
    device const float *input_values [[buffer(0)]],
    device const uint *input_indices [[buffer(1)]],
    device uint *output_index [[buffer(2)]],
    constant uint &pair_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float best_values[ARGMAX_THREADS];
    threadgroup uint best_indices[ARGMAX_THREADS];

    float best_value = -INFINITY;
    uint best_index = 0;
    for (uint pair_index = tid; pair_index < pair_count; pair_index += ARGMAX_THREADS) {
        const float value = input_values[pair_index];
        if (value > best_value) {
            best_value = value;
            best_index = input_indices[pair_index];
        }
    }

    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ARGMAX_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            const float other_value = best_values[tid + stride];
            const uint other_index = best_indices[tid + stride];
            if (other_value > best_values[tid]) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) output_index[0] = best_indices[0];
}

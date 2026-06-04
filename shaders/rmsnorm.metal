#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

// RMSNorm over the last dim, in f32 (verified exact against mx.fast.rms_norm):
//   y = x * rsqrt(mean(x^2) + eps) * weight
// weight: bf16 [dim] (shared across rows); x/y: f32 [rows, dim].
// One thread per row (correctness-first; a threadgroup reduction can come later).
kernel void rmsnorm_f32(
    device const float *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const device float *xr = x + (ulong)row * dim;
    device float *yr = y + (ulong)row * dim;

    float ss = 0.0f;
    for (uint i = 0; i < dim; i++) ss += xr[i] * xr[i];
    // rsqrt is ≤2 ULP (no -ffast-math); the CPU ref uses 1/sqrt, so GPU vs CPU
    // can differ ~5e-7 relative — far under the test's 1e-4 bar.
    const float scale = rsqrt(ss / float(dim) + eps);
    for (uint i = 0; i < dim; i++) yr[i] = xr[i] * scale * bf16_to_float(weight[i]);
}

constant uint ADD_RMSNORM_MAX_THREADS = 512;

// residual_out = lhs + rhs; norm_out = rmsnorm(residual_out, weight).
// Ported from Kestrel's Qwen3.5 prefill add-rmsnorm contract. residual_out may
// alias lhs and norm_out may alias rhs when those source buffers are dead.
kernel void add_rmsnorm_f32(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device const ushort *weight [[buffer(2)]],
    device float *residual_out [[buffer(3)]],
    device float *norm_out [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant uint &threads_per_threadgroup [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (dim == 0 ||
        rows == 0 ||
        row >= rows ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > ADD_RMSNORM_MAX_THREADS) return;

    const uint row_offset = row * dim;
    threadgroup float partial[ADD_RMSNORM_MAX_THREADS];

    float ss = 0.0f;
    for (uint i = tid; i < dim; i += threads_per_threadgroup) {
        const float residual = lhs[row_offset + i] + rhs[row_offset + i];
        residual_out[row_offset + i] = residual;
        ss += residual * residual;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += threads_per_threadgroup) {
        const float residual = residual_out[row_offset + i];
        norm_out[row_offset + i] = residual * scale * bf16_to_float(weight[i]);
    }
}

#if defined(__HAVE_BFLOAT__)
template<typename T>
METAL_FUNC void rmsnorm_threadgroup_impl(
    device const T *input,
    device const T *weight,
    device T *output,
    constant uint &hidden_size,
    constant uint &row_count,
    constant float &epsilon,
    constant uint &threads_per_threadgroup,
    threadgroup float *partial_sums,
    uint tid,
    uint row)
{
    if (hidden_size == 0 ||
        row_count == 0 ||
        row >= row_count ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > ADD_RMSNORM_MAX_THREADS) return;

    const uint row_offset = row * hidden_size;
    float sum = 0.0f;
    for (uint col = tid; col < hidden_size; col += threads_per_threadgroup) {
        const float value = float(input[row_offset + col]);
        sum += value * value;
    }
    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sums[tid] += partial_sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial_sums[0] / float(hidden_size) + epsilon);
    for (uint col = tid; col < hidden_size; col += threads_per_threadgroup) {
        output[row_offset + col] = T(float(input[row_offset + col]) * scale * float(weight[col]));
    }
}

template<typename T>
METAL_FUNC void rmsnorm_pair_threadgroup_impl(
    device const T *input_a,
    device const T *weight_a,
    device T *output_a,
    constant uint &row_count_a,
    device const T *input_b,
    device const T *weight_b,
    device T *output_b,
    constant uint &row_count_b,
    constant uint &hidden_size,
    constant float &epsilon,
    constant uint &threads_per_threadgroup,
    threadgroup float *partial_sums,
    uint tid,
    uint row)
{
    const uint total_rows = row_count_a + row_count_b;
    if (hidden_size == 0 ||
        total_rows == 0 ||
        row >= total_rows ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > ADD_RMSNORM_MAX_THREADS) return;

    const bool use_a = row < row_count_a;
    const uint row_index = use_a ? row : row - row_count_a;
    const uint row_offset = row_index * hidden_size;

    float sum = 0.0f;
    for (uint col = tid; col < hidden_size; col += threads_per_threadgroup) {
        const float value = use_a ?
            float(input_a[row_offset + col]) :
            float(input_b[row_offset + col]);
        sum += value * value;
    }
    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sums[tid] += partial_sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial_sums[0] / float(hidden_size) + epsilon);
    for (uint col = tid; col < hidden_size; col += threads_per_threadgroup) {
        const float input_value = use_a ?
            float(input_a[row_offset + col]) :
            float(input_b[row_offset + col]);
        const float weight_value = use_a ?
            float(weight_a[col]) :
            float(weight_b[col]);
        if (use_a) {
            output_a[row_offset + col] = T(input_value * scale * weight_value);
        } else {
            output_b[row_offset + col] = T(input_value * scale * weight_value);
        }
    }
}

template<typename T>
METAL_FUNC void add_rmsnorm_threadgroup_impl(
    device const T *lhs,
    device const T *rhs,
    device const T *weight,
    device T *residual_out,
    device T *norm_out,
    constant uint &hidden_size,
    constant uint &row_count,
    constant float &epsilon,
    constant uint &threads_per_threadgroup,
    threadgroup float *partial_sums,
    uint tid,
    uint row)
{
    if (hidden_size == 0 ||
        row_count == 0 ||
        row >= row_count ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > ADD_RMSNORM_MAX_THREADS) return;

    const uint row_offset = row * hidden_size;
    float sum = 0.0f;
    for (uint col = tid; col < hidden_size; col += threads_per_threadgroup) {
        // Match Kestrel tensor_add -> rmsnorm: normalize the dtype-rounded
        // residual, not the full f32 sum.
        const T residual = T(float(lhs[row_offset + col]) + float(rhs[row_offset + col]));
        residual_out[row_offset + col] = residual;
        const float normalized_input = float(residual);
        sum += normalized_input * normalized_input;
    }
    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial_sums[tid] += partial_sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial_sums[0] / float(hidden_size) + epsilon);
    for (uint col = tid; col < hidden_size; col += threads_per_threadgroup) {
        norm_out[row_offset + col] = T(float(residual_out[row_offset + col]) * scale * float(weight[col]));
    }
}

kernel void rmsnorm_bf16(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &hidden_size [[buffer(3)]],
    constant uint &row_count [[buffer(4)]],
    constant float &epsilon [[buffer(5)]],
    constant uint &threads_per_threadgroup [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    threadgroup float partial_sums[ADD_RMSNORM_MAX_THREADS];
    rmsnorm_threadgroup_impl<bfloat>(
        input,
        weight,
        output,
        hidden_size,
        row_count,
        epsilon,
        threads_per_threadgroup,
        partial_sums,
        tid,
        row);
}

kernel void rmsnorm_pair_bf16(
    device const bfloat *input_a [[buffer(0)]],
    device const bfloat *weight_a [[buffer(1)]],
    device bfloat *output_a [[buffer(2)]],
    constant uint &row_count_a [[buffer(3)]],
    device const bfloat *input_b [[buffer(4)]],
    device const bfloat *weight_b [[buffer(5)]],
    device bfloat *output_b [[buffer(6)]],
    constant uint &row_count_b [[buffer(7)]],
    constant uint &hidden_size [[buffer(8)]],
    constant float &epsilon [[buffer(9)]],
    constant uint &threads_per_threadgroup [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    threadgroup float partial_sums[ADD_RMSNORM_MAX_THREADS];
    rmsnorm_pair_threadgroup_impl<bfloat>(
        input_a,
        weight_a,
        output_a,
        row_count_a,
        input_b,
        weight_b,
        output_b,
        row_count_b,
        hidden_size,
        epsilon,
        threads_per_threadgroup,
        partial_sums,
        tid,
        row);
}

kernel void add_rmsnorm_bf16(
    device const bfloat *lhs [[buffer(0)]],
    device const bfloat *rhs [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device bfloat *residual_out [[buffer(3)]],
    device bfloat *norm_out [[buffer(4)]],
    constant uint &hidden_size [[buffer(5)]],
    constant uint &row_count [[buffer(6)]],
    constant float &epsilon [[buffer(7)]],
    constant uint &threads_per_threadgroup [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    threadgroup float partial_sums[ADD_RMSNORM_MAX_THREADS];
    add_rmsnorm_threadgroup_impl<bfloat>(
        lhs,
        rhs,
        weight,
        residual_out,
        norm_out,
        hidden_size,
        row_count,
        epsilon,
        threads_per_threadgroup,
        partial_sums,
        tid,
        row);
}
#endif

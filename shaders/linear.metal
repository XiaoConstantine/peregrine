#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

constant uint LINEAR_VEC_CACHED_INPUT_SIMDGROUP_WIDTH = 32;
constant uint Q4_AFFINE_FAST_THREADS_PER_THREADGROUP = 64;
constant uint Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP = 4;
constant uint Q4_AFFINE_FAST_SIMDGROUPS_PER_THREADGROUP = 2;
constant uint Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP =
    Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP * Q4_AFFINE_FAST_SIMDGROUPS_PER_THREADGROUP;

// q4 group-64 affine quantized matrix-vector:
//   out[r] = sum_c (scale[r, c/64] * q[r,c] + bias[r, c/64]) * x[c]
//
// Layout (verified against mlx.dequantize):
//   weight: uint32 [out_dim, in_dim/8] — 8 nibbles per word, low nibble = lowest column
//   scales/biases: bf16 [out_dim, in_dim/64]
//   x: f32 [in_dim]; out: f32 [out_dim]
//
// ONE SIMD GROUP (32 lanes) PER OUTPUT ROW. Decode is bandwidth-bound (it streams
// all ~5GB of weights per token), so coalesced loads are everything: lane L reads
// packed words L, L+32, L+64, ... so the 32 lanes of a step read 32 CONTIGUOUS
// words (one cache line's worth), then simd_sum reduces the partial dot products.
// Dispatch grid = out_dim * 32 (32 = the Apple-Silicon SIMD width); the bridge
// uses threadExecutionWidth (32) threads per threadgroup, so each threadgroup is
// exactly one SIMD group, and all 32 lanes of a group share the same row.
kernel void q4_affine_group64_qmv_f32(
    device const uint *weight [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row = gid / 32;
    if (row >= out_dim) return;

    const uint words = in_dim / 8; // packed uint32 columns per row
    const device uint *wrow = weight + (ulong)row * words;
    const device ushort *srow = scales + (ulong)row * (in_dim / 64);
    const device ushort *brow = biases + (ulong)row * (in_dim / 64);

    float acc = 0.0f;
    // Each word packs 8 columns; 8 words make one 64-wide group, so word w is in
    // group w/8. Lanes stride by 32 for fully coalesced weight loads.
    for (uint w = lane; w < words; w += 32) {
        const uint packed = wrow[w];
        const uint g = w / 8;
        const float s = bf16_to_float(srow[g]);
        const float b = bf16_to_float(brow[g]);
        const uint c0 = w * 8;
        for (uint j = 0; j < 8; j++) {
            const uint q = (packed >> (j * 4)) & 0xF;
            acc += (s * float(q) + b) * x[c0 + j];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

#if defined(__HAVE_BFLOAT__)
METAL_FUNC void q4_affine_group64_qmv_fast_compute4_bf16(
    device const bfloat *input,
    device const uint *weight,
    device const ushort *scales,
    device const ushort *biases,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id,
    thread float *result
) {
    constexpr uint values_per_thread = 16;
    const uint bytes_per_row = in_dim / 2;
    const uint group_count = in_dim / 64;
    const uint lane = uint(simd_lane_id);
    const device uchar *weight_bytes = reinterpret_cast<const device uchar *>(weight);

    for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
        result[row] = 0.0f;
    }

    for (uint k = 0; k < in_dim; k += values_per_thread * LINEAR_VEC_CACHED_INPUT_SIMDGROUP_WIDTH) {
        const uint input_base = k + lane * values_per_thread;
        float x_thread[values_per_thread];
        float input_sum = 0.0f;
        for (uint index = 0; index < values_per_thread; index += 4) {
            const float x0 = float(input[input_base + index + 0]);
            const float x1 = float(input[input_base + index + 1]);
            const float x2 = float(input[input_base + index + 2]);
            const float x3 = float(input[input_base + index + 3]);
            input_sum += x0 + x1 + x2 + x3;
            x_thread[index + 0] = x0;
            x_thread[index + 1] = x1 / 16.0f;
            x_thread[index + 2] = x2 / 256.0f;
            x_thread[index + 3] = x3 / 4096.0f;
        }

        const uint k_byte_offset = k / 2 + lane * 8;
        const uint group_index = k / 64 + lane / 4;
        for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint out_index = out_base + row;
            const uint weight_base = out_index * bytes_per_row + k_byte_offset;
            const device ushort *packed16 = reinterpret_cast<const device ushort *>(weight_bytes + weight_base);
            float quant_acc = 0.0f;
            for (uint pack_index = 0; pack_index < 4; pack_index += 1) {
                const ushort packed = packed16[pack_index];
                const uint x_base = pack_index * 4;
                quant_acc += float(packed & 0x000f) * x_thread[x_base + 0];
                quant_acc += float(packed & 0x00f0) * x_thread[x_base + 1];
                quant_acc += float(packed & 0x0f00) * x_thread[x_base + 2];
                quant_acc += float(packed & 0xf000) * x_thread[x_base + 3];
            }

            const uint metadata_index = out_index * group_count + group_index;
            result[row] += bf16_to_float(scales[metadata_index]) * quant_acc +
                bf16_to_float(biases[metadata_index]) * input_sum;
        }
    }

    for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
        result[row] = simd_sum(result[row]);
    }
}

METAL_FUNC void q4_affine_group64_qmv_fast_write4_bf16(
    device const bfloat *input,
    device const uint *weight,
    device const ushort *scales,
    device const ushort *biases,
    device bfloat *output,
    uint output_offset,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id
) {
    float result[Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP];
    q4_affine_group64_qmv_fast_compute4_bf16(
        input,
        weight,
        scales,
        biases,
        out_base,
        in_dim,
        simd_lane_id,
        result
    );

    if (simd_lane_id == 0) {
        for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
            output[output_offset + out_base + row] = bfloat(result[row]);
        }
    }
}

kernel void linear_q4_affine_group64_qmv_fast_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &in_dim [[buffer(7)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_AFFINE_FAST_THREADS_PER_THREADGROUP ||
        token_count == 0 || in_dim % 512 != 0 ||
        out_dim % Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP != 0) return;

    const uint groups_per_token = out_dim / Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP;
    const uint token_index = threadgroup_index / groups_per_token;
    if (token_index >= token_count) return;
    const uint local_group_index = threadgroup_index - token_index * groups_per_token;
    const uint out_base = local_group_index * Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP;
    q4_affine_group64_qmv_fast_write4_bf16(
        input + (ulong)token_index * in_dim,
        weight,
        scales,
        biases,
        output,
        token_index * out_dim,
        out_base,
        in_dim,
        simd_lane_id
    );
}

kernel void linear_vec_q4_affine_group64_qmv_fast_argmax_partial_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device float *partial_values [[buffer(4)]],
    device uint *partial_indices [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &in_dim [[buffer(7)]],
    uint tid_in_threadgroup [[thread_index_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_AFFINE_FAST_THREADS_PER_THREADGROUP ||
        in_dim % 512 != 0 ||
        out_dim % Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP != 0) return;

    const uint out_base = threadgroup_index * Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP;
    float result[Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP];
    q4_affine_group64_qmv_fast_compute4_bf16(
        input,
        weight,
        scales,
        biases,
        out_base,
        in_dim,
        simd_lane_id,
        result
    );

    threadgroup float group_values[Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP];
    threadgroup uint group_indices[Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP];
    if (simd_lane_id == 0) {
        const uint slot_base = uint(simdgroup_index) * Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP;
        for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint slot = slot_base + row;
            const uint out_index = out_base + row;
            // Match the previous serving BF16 logits -> CPU argmax rounding.
            group_values[slot] = float(bfloat(result[row]));
            group_indices[slot] = out_index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_threadgroup == 0) {
        float best_value = group_values[0];
        uint best_index = group_indices[0];
        for (uint slot = 1; slot < Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP; slot += 1) {
            const float value = group_values[slot];
            if (value > best_value) {
                best_value = value;
                best_index = group_indices[slot];
            }
        }
        partial_values[threadgroup_index] = best_value;
        partial_indices[threadgroup_index] = best_index;
    }
}

METAL_FUNC void q4_affine_group64_qmv_fast_compute4x2_bf16(
    device const bfloat *input0,
    device const bfloat *input1,
    device const uint *weight,
    device const ushort *scales,
    device const ushort *biases,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id,
    thread float *result0,
    thread float *result1
) {
    constexpr uint values_per_thread = 16;
    const uint bytes_per_row = in_dim / 2;
    const uint group_count = in_dim / 64;
    const uint lane = uint(simd_lane_id);
    const device uchar *weight_bytes = reinterpret_cast<const device uchar *>(weight);

    for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
        result0[row] = 0.0f;
        result1[row] = 0.0f;
    }

    for (uint k = 0; k < in_dim; k += values_per_thread * LINEAR_VEC_CACHED_INPUT_SIMDGROUP_WIDTH) {
        const uint input_base = k + lane * values_per_thread;
        float x0_thread[values_per_thread];
        float x1_thread[values_per_thread];
        float input0_sum = 0.0f;
        float input1_sum = 0.0f;
        for (uint index = 0; index < values_per_thread; index += 4) {
            const float x00 = float(input0[input_base + index + 0]);
            const float x01 = float(input0[input_base + index + 1]);
            const float x02 = float(input0[input_base + index + 2]);
            const float x03 = float(input0[input_base + index + 3]);
            const float x10 = float(input1[input_base + index + 0]);
            const float x11 = float(input1[input_base + index + 1]);
            const float x12 = float(input1[input_base + index + 2]);
            const float x13 = float(input1[input_base + index + 3]);
            input0_sum += x00 + x01 + x02 + x03;
            input1_sum += x10 + x11 + x12 + x13;
            x0_thread[index + 0] = x00;
            x0_thread[index + 1] = x01 / 16.0f;
            x0_thread[index + 2] = x02 / 256.0f;
            x0_thread[index + 3] = x03 / 4096.0f;
            x1_thread[index + 0] = x10;
            x1_thread[index + 1] = x11 / 16.0f;
            x1_thread[index + 2] = x12 / 256.0f;
            x1_thread[index + 3] = x13 / 4096.0f;
        }

        const uint k_byte_offset = k / 2 + lane * 8;
        const uint group_index = k / 64 + lane / 4;
        for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint out_index = out_base + row;
            const uint weight_base = out_index * bytes_per_row + k_byte_offset;
            const device ushort *packed16 = reinterpret_cast<const device ushort *>(weight_bytes + weight_base);
            float quant0_acc = 0.0f;
            float quant1_acc = 0.0f;
            for (uint pack_index = 0; pack_index < 4; pack_index += 1) {
                const ushort packed = packed16[pack_index];
                const uint x_base = pack_index * 4;
                quant0_acc += float(packed & 0x000f) * x0_thread[x_base + 0];
                quant0_acc += float(packed & 0x00f0) * x0_thread[x_base + 1];
                quant0_acc += float(packed & 0x0f00) * x0_thread[x_base + 2];
                quant0_acc += float(packed & 0xf000) * x0_thread[x_base + 3];
                quant1_acc += float(packed & 0x000f) * x1_thread[x_base + 0];
                quant1_acc += float(packed & 0x00f0) * x1_thread[x_base + 1];
                quant1_acc += float(packed & 0x0f00) * x1_thread[x_base + 2];
                quant1_acc += float(packed & 0xf000) * x1_thread[x_base + 3];
            }

            const uint metadata_index = out_index * group_count + group_index;
            const float scale = bf16_to_float(scales[metadata_index]);
            const float bias = bf16_to_float(biases[metadata_index]);
            result0[row] += scale * quant0_acc + bias * input0_sum;
            result1[row] += scale * quant1_acc + bias * input1_sum;
        }
    }

    for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
    }
}

kernel void linear_vec_q4_affine_group64_qmv_fast_argmax2_partial_bf16(
    device const bfloat *input0 [[buffer(0)]],
    device const bfloat *input1 [[buffer(1)]],
    device const uint *weight [[buffer(2)]],
    device const ushort *scales [[buffer(3)]],
    device const ushort *biases [[buffer(4)]],
    device float *partial_values0 [[buffer(5)]],
    device uint *partial_indices0 [[buffer(6)]],
    device float *partial_values1 [[buffer(7)]],
    device uint *partial_indices1 [[buffer(8)]],
    constant uint &out_dim [[buffer(9)]],
    constant uint &in_dim [[buffer(10)]],
    uint tid_in_threadgroup [[thread_index_in_threadgroup]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_AFFINE_FAST_THREADS_PER_THREADGROUP ||
        in_dim % 512 != 0 ||
        out_dim % Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP != 0) return;

    const uint out_base = threadgroup_index * Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP;
    float result0[Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP];
    float result1[Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP];
    q4_affine_group64_qmv_fast_compute4x2_bf16(
        input0,
        input1,
        weight,
        scales,
        biases,
        out_base,
        in_dim,
        simd_lane_id,
        result0,
        result1
    );

    threadgroup float group_values0[Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP];
    threadgroup uint group_indices0[Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP];
    threadgroup float group_values1[Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP];
    threadgroup uint group_indices1[Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP];
    if (simd_lane_id == 0) {
        const uint slot_base = uint(simdgroup_index) * Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP;
        for (uint row = 0; row < Q4_AFFINE_FAST_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint slot = slot_base + row;
            const uint out_index = out_base + row;
            group_values0[slot] = float(bfloat(result0[row]));
            group_indices0[slot] = out_index;
            group_values1[slot] = float(bfloat(result1[row]));
            group_indices1[slot] = out_index;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_threadgroup == 0) {
        float best_value0 = group_values0[0];
        uint best_index0 = group_indices0[0];
        float best_value1 = group_values1[0];
        uint best_index1 = group_indices1[0];
        for (uint slot = 1; slot < Q4_AFFINE_FAST_RESULTS_PER_THREADGROUP; slot += 1) {
            const float value0 = group_values0[slot];
            if (value0 > best_value0) {
                best_value0 = value0;
                best_index0 = group_indices0[slot];
            }
            const float value1 = group_values1[slot];
            if (value1 > best_value1) {
                best_value1 = value1;
                best_index1 = group_indices1[slot];
            }
        }
        partial_values0[threadgroup_index] = best_value0;
        partial_indices0[threadgroup_index] = best_index0;
        partial_values1[threadgroup_index] = best_value1;
        partial_indices1[threadgroup_index] = best_index1;
    }
}

kernel void q4_affine_group64_dequantize_bf16(
    device const uint *weight [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint &out_dim [[buffer(4)]],
    constant uint &in_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong total = ulong(out_dim) * ulong(in_dim);
    if (ulong(gid) >= total || out_dim == 0 || in_dim == 0 || in_dim % 64 != 0) return;

    const uint out_index = gid / in_dim;
    const uint k = gid - out_index * in_dim;
    const uint packed_in_dim = in_dim / 8;
    const uint group_count = in_dim / 64;
    const uint packed = weight[out_index * packed_in_dim + k / 8];
    const uint q = (packed >> ((k & 7u) * 4u)) & 0x0fu;
    const uint metadata_index = out_index * group_count + k / 64;
    const float scale = bf16_to_float(scales[metadata_index]);
    const float bias = bf16_to_float(biases[metadata_index]);
    output[gid] = bfloat(scale * float(q) + bias);
}
#endif

constant uint Q4_AFFINE_QMM_TILE_M = 32;
constant uint Q4_AFFINE_QMM_TILE_N = 64;
constant uint Q4_AFFINE_QMM_TILE_K = 32;
constant uint Q4_AFFINE_QMM_WM = 2;
constant uint Q4_AFFINE_QMM_WN = 4;
constant uint Q4_AFFINE_QMM_PAD_A = 8;
constant uint Q4_AFFINE_QMM_PAD_B = 8;
constant uint Q4_AFFINE_QMM_LHS_LD = Q4_AFFINE_QMM_TILE_K + Q4_AFFINE_QMM_PAD_A;
constant uint Q4_AFFINE_QMM_NT_RHS_LD = Q4_AFFINE_QMM_TILE_K + Q4_AFFINE_QMM_PAD_B;
constant uint Q4_AFFINE_QMM_TM_STRIDE = 8 * Q4_AFFINE_QMM_WM;
constant uint Q4_AFFINE_QMM_TN_STRIDE = 8 * Q4_AFFINE_QMM_WN;
constant uint Q4_AFFINE_QMM_TM = Q4_AFFINE_QMM_TILE_M / Q4_AFFINE_QMM_TM_STRIDE;
constant uint Q4_AFFINE_QMM_TN = Q4_AFFINE_QMM_TILE_N / Q4_AFFINE_QMM_TN_STRIDE;
constant uint Q4_AFFINE_QMM_THREADS_PER_THREADGROUP =
    Q4_AFFINE_QMM_WM * Q4_AFFINE_QMM_WN * 32;
constant uint Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD =
    (Q4_AFFINE_QMM_TILE_M * Q4_AFFINE_QMM_TILE_K) / Q4_AFFINE_QMM_THREADS_PER_THREADGROUP;
constant uint Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD =
    (Q4_AFFINE_QMM_TILE_N * (Q4_AFFINE_QMM_TILE_K / 2)) / Q4_AFFINE_QMM_THREADS_PER_THREADGROUP;
static_assert((Q4_AFFINE_QMM_TILE_M * Q4_AFFINE_QMM_TILE_K) % Q4_AFFINE_QMM_THREADS_PER_THREADGROUP == 0,
    "QMM lhs tile cells must divide evenly over the threadgroup");
static_assert((Q4_AFFINE_QMM_TILE_N * (Q4_AFFINE_QMM_TILE_K / 2)) % Q4_AFFINE_QMM_THREADS_PER_THREADGROUP == 0,
    "QMM rhs packed bytes must divide evenly over the threadgroup");
static_assert(Q4_AFFINE_QMM_TILE_K % 2 == 0,
    "QMM K tile must use whole q4 bytes");
static_assert(Q4_AFFINE_QMM_TILE_K <= 64 && 64 % Q4_AFFINE_QMM_TILE_K == 0,
    "QMM K tile must divide the q4 group size");

// Q4 group-64 affine matrix-matrix projection for prompt prefill:
//   output[token, out] = dequant(weight[out, :]) dot input[token, :]
// input/output are f32 because Peregrine's current decode stack stores hidden
// states in f32. Scales and biases are bf16 stored as ushort, matching qmv.
kernel void q4_affine_group64_qmm_f32(
    device const float *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &in_dim [[buffer(7)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    if (threads_per_threadgroup != Q4_AFFINE_QMM_THREADS_PER_THREADGROUP ||
        token_count == 0 || out_dim == 0 || in_dim == 0 || in_dim % 64 != 0) return;

    const uint tiles_n = (out_dim + Q4_AFFINE_QMM_TILE_N - 1) / Q4_AFFINE_QMM_TILE_N;
    const uint tile_row = threadgroup_index / tiles_n;
    const uint tile_col = threadgroup_index - tile_row * tiles_n;
    const uint row_base = tile_row * Q4_AFFINE_QMM_TILE_M;
    const uint col_base = tile_col * Q4_AFFINE_QMM_TILE_N;
    if (row_base >= token_count) return;

    threadgroup float lhs_tile[Q4_AFFINE_QMM_TILE_M * Q4_AFFINE_QMM_LHS_LD];
    threadgroup float rhs_tile[Q4_AFFINE_QMM_TILE_N * Q4_AFFINE_QMM_NT_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[Q4_AFFINE_QMM_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[Q4_AFFINE_QMM_TN];
    simdgroup_matrix<float, 8, 8> results[Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN];

    for (uint i = 0; i < Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN; i += 1) {
        results[i] = simdgroup_matrix<float, 8, 8>(0);
    }

    const ushort tm = 8 * (simdgroup_index / Q4_AFFINE_QMM_WN);
    const ushort tn = 8 * (simdgroup_index % Q4_AFFINE_QMM_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * Q4_AFFINE_QMM_LHS_LD;
    const ushort rhs_offset = (tn + sn) * Q4_AFFINE_QMM_NT_RHS_LD + sm;

    const uint bytes_per_row = in_dim / 2;
    const uint group_count = in_dim / 64;
    const device uchar *weight_bytes = reinterpret_cast<const device uchar *>(weight);

    for (uint k_base = 0; k_base < in_dim; k_base += Q4_AFFINE_QMM_TILE_K) {
        for (uint load = 0; load < Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD; load += 1) {
            const uint lhs_index = local_tid * Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD + load;
            const uint lhs_row = lhs_index / Q4_AFFINE_QMM_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * Q4_AFFINE_QMM_TILE_K;
            const uint global_row = row_base + lhs_row;
            const uint global_col = k_base + lhs_col;
            lhs_tile[lhs_row * Q4_AFFINE_QMM_LHS_LD + lhs_col] =
                (global_row < token_count) ? input[(ulong)global_row * in_dim + global_col] : 0.0f;
        }

        for (uint load = 0; load < Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD; load += 1) {
            const uint rhs_pack_index = local_tid * Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD + load;
            const uint local_out = rhs_pack_index / (Q4_AFFINE_QMM_TILE_K / 2);
            const uint local_k_byte = rhs_pack_index - local_out * (Q4_AFFINE_QMM_TILE_K / 2);
            const uint global_out = col_base + local_out;
            const uint global_k = k_base + local_k_byte * 2;
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (global_out < out_dim) {
                const uchar packed = weight_bytes[(ulong)global_out * bytes_per_row + global_k / 2];
                const uint metadata_index = global_out * group_count + global_k / 64;
                const float scale = bf16_to_float(scales[metadata_index]);
                const float bias = bf16_to_float(biases[metadata_index]);
                value0 = scale * float(packed & 0x0fu) + bias;
                value1 = scale * float((packed >> 4) & 0x0fu) + bias;
            }
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 0] = value0;
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 1] = value1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < Q4_AFFINE_QMM_TILE_K; kk += 8) {
            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                lhs_simd[i].thread_elements()[0] =
                    lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 1];
            }

            for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                rhs_simd[j].thread_elements()[0] =
                    rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + 0];
                rhs_simd[j].thread_elements()[1] =
                    rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + Q4_AFFINE_QMM_NT_RHS_LD];
            }

            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                    const uint j_serp = (i % 2) ? (Q4_AFFINE_QMM_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * Q4_AFFINE_QMM_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * Q4_AFFINE_QMM_TN + j_serp]
                    );
                }
            }

            lhs_block += 8;
            rhs_block += 8;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= token_count || col_offset >= out_dim) return;

    device float *dst = output + (ulong)row_offset * out_dim + col_offset;
    const uint2 dst_tile_dims = uint2(
        min(Q4_AFFINE_QMM_TILE_N, out_dim - col_base) - (tn + sn),
        min(Q4_AFFINE_QMM_TILE_M, token_count - row_base) - (tm + sm)
    );
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
        if (i * Q4_AFFINE_QMM_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
            thread const auto &accum = results[i * Q4_AFFINE_QMM_TN + j].thread_elements();
            const uint offset = (i * Q4_AFFINE_QMM_TM_STRIDE) * out_dim +
                j * Q4_AFFINE_QMM_TN_STRIDE;
            if (j * Q4_AFFINE_QMM_TN_STRIDE < dst_tile_dims.x) {
                dst[offset] = accum[0];
            }
            if (j * Q4_AFFINE_QMM_TN_STRIDE + 1 < dst_tile_dims.x) {
                dst[offset + 1] = accum[1];
            }
        }
    }
}

// Same projection as q4_affine_group64_qmm_f32, fused with residual add:
//   output[token, out] = qmm(input, weight)[token, out] + residual[token, out]
kernel void q4_affine_group64_qmm_residual_add_f32(
    device const float *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device const float *residual [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &out_dim [[buffer(7)]],
    constant uint &in_dim [[buffer(8)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    if (threads_per_threadgroup != Q4_AFFINE_QMM_THREADS_PER_THREADGROUP ||
        token_count == 0 || out_dim == 0 || in_dim == 0 || in_dim % 64 != 0) return;

    const uint tiles_n = (out_dim + Q4_AFFINE_QMM_TILE_N - 1) / Q4_AFFINE_QMM_TILE_N;
    const uint tile_row = threadgroup_index / tiles_n;
    const uint tile_col = threadgroup_index - tile_row * tiles_n;
    const uint row_base = tile_row * Q4_AFFINE_QMM_TILE_M;
    const uint col_base = tile_col * Q4_AFFINE_QMM_TILE_N;
    if (row_base >= token_count) return;

    threadgroup float lhs_tile[Q4_AFFINE_QMM_TILE_M * Q4_AFFINE_QMM_LHS_LD];
    threadgroup float rhs_tile[Q4_AFFINE_QMM_TILE_N * Q4_AFFINE_QMM_NT_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[Q4_AFFINE_QMM_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[Q4_AFFINE_QMM_TN];
    simdgroup_matrix<float, 8, 8> results[Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN];

    for (uint i = 0; i < Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN; i += 1) {
        results[i] = simdgroup_matrix<float, 8, 8>(0);
    }

    const ushort tm = 8 * (simdgroup_index / Q4_AFFINE_QMM_WN);
    const ushort tn = 8 * (simdgroup_index % Q4_AFFINE_QMM_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * Q4_AFFINE_QMM_LHS_LD;
    const ushort rhs_offset = (tn + sn) * Q4_AFFINE_QMM_NT_RHS_LD + sm;

    const uint bytes_per_row = in_dim / 2;
    const uint group_count = in_dim / 64;
    const device uchar *weight_bytes = reinterpret_cast<const device uchar *>(weight);

    for (uint k_base = 0; k_base < in_dim; k_base += Q4_AFFINE_QMM_TILE_K) {
        for (uint load = 0; load < Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD; load += 1) {
            const uint lhs_index = local_tid * Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD + load;
            const uint lhs_row = lhs_index / Q4_AFFINE_QMM_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * Q4_AFFINE_QMM_TILE_K;
            const uint global_row = row_base + lhs_row;
            const uint global_col = k_base + lhs_col;
            lhs_tile[lhs_row * Q4_AFFINE_QMM_LHS_LD + lhs_col] =
                (global_row < token_count) ? input[(ulong)global_row * in_dim + global_col] : 0.0f;
        }

        for (uint load = 0; load < Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD; load += 1) {
            const uint rhs_pack_index = local_tid * Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD + load;
            const uint local_out = rhs_pack_index / (Q4_AFFINE_QMM_TILE_K / 2);
            const uint local_k_byte = rhs_pack_index - local_out * (Q4_AFFINE_QMM_TILE_K / 2);
            const uint global_out = col_base + local_out;
            const uint global_k = k_base + local_k_byte * 2;
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (global_out < out_dim) {
                const uchar packed = weight_bytes[(ulong)global_out * bytes_per_row + global_k / 2];
                const uint metadata_index = global_out * group_count + global_k / 64;
                const float scale = bf16_to_float(scales[metadata_index]);
                const float bias = bf16_to_float(biases[metadata_index]);
                value0 = scale * float(packed & 0x0fu) + bias;
                value1 = scale * float((packed >> 4) & 0x0fu) + bias;
            }
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 0] = value0;
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 1] = value1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < Q4_AFFINE_QMM_TILE_K; kk += 8) {
            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                lhs_simd[i].thread_elements()[0] =
                    lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 1];
            }

            for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                rhs_simd[j].thread_elements()[0] =
                    rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + 0];
                rhs_simd[j].thread_elements()[1] =
                    rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + Q4_AFFINE_QMM_NT_RHS_LD];
            }

            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                    const uint j_serp = (i % 2) ? (Q4_AFFINE_QMM_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * Q4_AFFINE_QMM_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * Q4_AFFINE_QMM_TN + j_serp]
                    );
                }
            }

            lhs_block += 8;
            rhs_block += 8;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= token_count || col_offset >= out_dim) return;

    device float *dst = output + (ulong)row_offset * out_dim + col_offset;
    const device float *res = residual + (ulong)row_offset * out_dim + col_offset;
    const uint2 dst_tile_dims = uint2(
        min(Q4_AFFINE_QMM_TILE_N, out_dim - col_base) - (tn + sn),
        min(Q4_AFFINE_QMM_TILE_M, token_count - row_base) - (tm + sm)
    );
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
        if (i * Q4_AFFINE_QMM_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
            thread const auto &accum = results[i * Q4_AFFINE_QMM_TN + j].thread_elements();
            const uint offset = (i * Q4_AFFINE_QMM_TM_STRIDE) * out_dim +
                j * Q4_AFFINE_QMM_TN_STRIDE;
            if (j * Q4_AFFINE_QMM_TN_STRIDE < dst_tile_dims.x) {
                dst[offset] = accum[0] + res[offset];
            }
            if (j * Q4_AFFINE_QMM_TN_STRIDE + 1 < dst_tile_dims.x) {
                dst[offset + 1] = accum[1] + res[offset + 1];
            }
        }
    }
}

#if defined(__HAVE_BFLOAT__)
template <
    uint TILE_M,
    uint TILE_N,
    uint TILE_K,
    uint WM,
    uint WN,
    uint PAD_A,
    uint PAD_B,
    bool ADD_RESIDUAL
>
METAL_FUNC void q4_affine_group64_qmm_nt_bf16_tile_impl(
    device const bfloat *input,
    device const uint *weight,
    device const bfloat *scales,
    device const bfloat *biases,
    device const bfloat *residual,
    device bfloat *output,
    threadgroup bfloat *lhs_tile,
    threadgroup bfloat *rhs_tile,
    constant uint &token_count,
    constant uint &out_dim,
    constant uint &in_dim,
    ushort simd_lane_id,
    ushort simdgroup_index,
    uint threads_per_threadgroup,
    uint local_tid,
    uint threadgroup_index
) {
    constexpr uint LHS_LD = TILE_K + PAD_A;
    constexpr uint RHS_LD = TILE_K + PAD_B;
    constexpr uint TM_STRIDE = 8 * WM;
    constexpr uint TN_STRIDE = 8 * WN;
    constexpr uint TM = TILE_M / TM_STRIDE;
    constexpr uint TN = TILE_N / TN_STRIDE;
    constexpr uint THREADS = WM * WN * 32;
    constexpr uint LHS_LOADS_PER_THREAD = (TILE_M * TILE_K) / THREADS;
    constexpr uint RHS_PACK_LOADS_PER_THREAD = (TILE_N * (TILE_K / 2)) / THREADS;

    static_assert(TILE_K <= 64 && 64 % TILE_K == 0, "QMM K tile must divide the q4 group size");
    static_assert(TILE_K % 2 == 0, "QMM K tile must use whole q4 bytes");
    static_assert(TILE_M % TM_STRIDE == 0, "QMM TILE_M must divide simdgroup tile stride");
    static_assert(TILE_N % TN_STRIDE == 0, "QMM TILE_N must divide simdgroup tile stride");
    static_assert((TILE_M * TILE_K) % THREADS == 0, "QMM lhs tile cells must divide evenly");
    static_assert((TILE_N * (TILE_K / 2)) % THREADS == 0, "QMM rhs packed bytes must divide evenly");

    if (threads_per_threadgroup != THREADS ||
        token_count == 0 || out_dim == 0 || in_dim == 0 || in_dim % 64 != 0) return;

    const uint tiles_n = (out_dim + TILE_N - 1) / TILE_N;
    const uint tile_row = threadgroup_index / tiles_n;
    const uint tile_col = threadgroup_index - tile_row * tiles_n;
    const uint row_base = tile_row * TILE_M;
    const uint col_base = tile_col * TILE_N;
    if (row_base >= token_count) return;

    simdgroup_matrix<float, 8, 8> lhs_simd[TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[TN];
    simdgroup_matrix<float, 8, 8> results[TM * TN];

    for (uint i = 0; i < TM * TN; i += 1) {
        results[i] = simdgroup_matrix<float, 8, 8>(0);
    }

    const ushort tm = 8 * (simdgroup_index / WN);
    const ushort tn = 8 * (simdgroup_index % WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * LHS_LD;
    const ushort rhs_offset = (tn + sn) * RHS_LD + sm;

    const uint bytes_per_row = in_dim / 2;
    const uint group_count = in_dim / 64;
    const device uchar *weight_bytes = reinterpret_cast<const device uchar *>(weight);

    for (uint k_base = 0; k_base < in_dim; k_base += TILE_K) {
        for (uint load = 0; load < LHS_LOADS_PER_THREAD; load += 1) {
            const uint lhs_index = local_tid * LHS_LOADS_PER_THREAD + load;
            const uint lhs_row = lhs_index / TILE_K;
            const uint lhs_col = lhs_index - lhs_row * TILE_K;
            const uint global_row = row_base + lhs_row;
            const uint global_col = k_base + lhs_col;
            lhs_tile[lhs_row * LHS_LD + lhs_col] =
                (global_row < token_count) ? input[(ulong)global_row * in_dim + global_col] : bfloat(0.0f);
        }

        for (uint load = 0; load < RHS_PACK_LOADS_PER_THREAD; load += 1) {
            const uint rhs_pack_index = local_tid * RHS_PACK_LOADS_PER_THREAD + load;
            const uint local_out = rhs_pack_index / (TILE_K / 2);
            const uint local_k_byte = rhs_pack_index - local_out * (TILE_K / 2);
            const uint global_out = col_base + local_out;
            const uint global_k = k_base + local_k_byte * 2;
            bfloat value0 = bfloat(0.0f);
            bfloat value1 = bfloat(0.0f);
            if (global_out < out_dim) {
                const uchar packed = weight_bytes[(ulong)global_out * bytes_per_row + global_k / 2];
                const uint metadata_index = global_out * group_count + global_k / 64;
                const float scale = float(scales[metadata_index]);
                const float bias = float(biases[metadata_index]);
                value0 = bfloat(scale * float(packed & 0x0fu) + bias);
                value1 = bfloat(scale * float((packed >> 4) & 0x0fu) + bias);
            }
            rhs_tile[local_out * RHS_LD + local_k_byte * 2 + 0] = value0;
            rhs_tile[local_out * RHS_LD + local_k_byte * 2 + 1] = value1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup bfloat *lhs_block = lhs_tile + lhs_offset;
        const threadgroup bfloat *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < TILE_K; kk += 8) {
            for (uint i = 0; i < TM; i += 1) {
                lhs_simd[i].thread_elements()[0] =
                    float(lhs_block[i * TM_STRIDE * LHS_LD + 0]);
                lhs_simd[i].thread_elements()[1] =
                    float(lhs_block[i * TM_STRIDE * LHS_LD + 1]);
            }

            for (uint j = 0; j < TN; j += 1) {
                rhs_simd[j].thread_elements()[0] =
                    float(rhs_block[j * TN_STRIDE * RHS_LD + 0]);
                rhs_simd[j].thread_elements()[1] =
                    float(rhs_block[j * TN_STRIDE * RHS_LD + RHS_LD]);
            }

            for (uint i = 0; i < TM; i += 1) {
                for (uint j = 0; j < TN; j += 1) {
                    const uint j_serp = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * TN + j_serp]
                    );
                }
            }

            lhs_block += 8;
            rhs_block += 8;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= token_count || col_offset >= out_dim) return;

    device bfloat *dst = output + (ulong)row_offset * out_dim + col_offset;
    const device bfloat *res = residual + (ulong)row_offset * out_dim + col_offset;
    const uint2 dst_tile_dims = uint2(
        min(TILE_N, out_dim - col_base) - (tn + sn),
        min(TILE_M, token_count - row_base) - (tm + sm)
    );
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < TM; i += 1) {
        if (i * TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < TN; j += 1) {
            thread const auto &accum = results[i * TN + j].thread_elements();
            const uint offset = (i * TM_STRIDE) * out_dim + j * TN_STRIDE;
            if (j * TN_STRIDE < dst_tile_dims.x) {
                const bfloat rounded_dot = bfloat(accum[0]);
                dst[offset] = ADD_RESIDUAL ? bfloat(float(rounded_dot) + float(res[offset])) : rounded_dot;
            }
            if (j * TN_STRIDE + 1 < dst_tile_dims.x) {
                const bfloat rounded_dot = bfloat(accum[1]);
                dst[offset + 1] = ADD_RESIDUAL ? bfloat(float(rounded_dot) + float(res[offset + 1])) : rounded_dot;
            }
        }
    }
}

kernel void linear_q4_affine_group64_prefill_qmm_m32n64_nt_bf16_tiles_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]],
    device const bfloat *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    constant uint &out_dim [[buffer(6)]],
    constant uint &in_dim [[buffer(7)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    threadgroup bfloat lhs_tile[32 * (32 + 8)];
    threadgroup bfloat rhs_tile[64 * (32 + 8)];
    q4_affine_group64_qmm_nt_bf16_tile_impl<32, 64, 32, 2, 4, 8, 8, false>(
        input, weight, scales, biases, output, output, lhs_tile, rhs_tile, token_count, out_dim, in_dim,
        simd_lane_id, simdgroup_index, threads_per_threadgroup, local_tid, threadgroup_index
    );
}

kernel void linear_q4_affine_group64_prefill_qmm_m32n64_nt_bf16_tiles_residual_add_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]],
    device const bfloat *biases [[buffer(3)]],
    device const bfloat *residual [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &out_dim [[buffer(7)]],
    constant uint &in_dim [[buffer(8)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    threadgroup bfloat lhs_tile[32 * (32 + 8)];
    threadgroup bfloat rhs_tile[64 * (32 + 8)];
    q4_affine_group64_qmm_nt_bf16_tile_impl<32, 64, 32, 2, 4, 8, 8, true>(
        input, weight, scales, biases, residual, output, lhs_tile, rhs_tile, token_count, out_dim, in_dim,
        simd_lane_id, simdgroup_index, threads_per_threadgroup, local_tid, threadgroup_index
    );
}
#endif

// Fused MLP gate/up prefill:
//   output[token, out] = silu(gate_proj(input))[token,out] * up_proj(input)[token,out]
// This mirrors the two QMMs plus silu_mul path, but shares the input tile and
// activation write. It is only used for Qwen3.5-9B MLP prefill.
kernel void q4_affine_group64_qmm_silu_gate_f32(
    device const float *input [[buffer(0)]],
    device const uint *gate_weight [[buffer(1)]],
    device const ushort *gate_scales [[buffer(2)]],
    device const ushort *gate_biases [[buffer(3)]],
    device const uint *up_weight [[buffer(4)]],
    device const ushort *up_scales [[buffer(5)]],
    device const ushort *up_biases [[buffer(6)]],
    device float *output [[buffer(7)]],
    constant uint &token_count [[buffer(8)]],
    constant uint &out_dim [[buffer(9)]],
    constant uint &in_dim [[buffer(10)]],
    constant uint &input_token_count [[buffer(11)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    if (threads_per_threadgroup != Q4_AFFINE_QMM_THREADS_PER_THREADGROUP ||
        token_count == 0 || input_token_count == 0 || input_token_count > token_count ||
        out_dim == 0 || in_dim == 0 || in_dim % 64 != 0) return;

    const uint tiles_n = (out_dim + Q4_AFFINE_QMM_TILE_N - 1) / Q4_AFFINE_QMM_TILE_N;
    const uint tile_row = threadgroup_index / tiles_n;
    const uint tile_col = threadgroup_index - tile_row * tiles_n;
    const uint row_base = tile_row * Q4_AFFINE_QMM_TILE_M;
    const uint col_base = tile_col * Q4_AFFINE_QMM_TILE_N;
    if (row_base >= token_count) return;

    threadgroup float lhs_tile[Q4_AFFINE_QMM_TILE_M * Q4_AFFINE_QMM_LHS_LD];
    threadgroup float rhs_tile[Q4_AFFINE_QMM_TILE_N * Q4_AFFINE_QMM_NT_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[Q4_AFFINE_QMM_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[Q4_AFFINE_QMM_TN];
    simdgroup_matrix<float, 8, 8> gate_results[Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN];
    simdgroup_matrix<float, 8, 8> up_results[Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN];

    for (uint i = 0; i < Q4_AFFINE_QMM_TM * Q4_AFFINE_QMM_TN; i += 1) {
        gate_results[i] = simdgroup_matrix<float, 8, 8>(0);
        up_results[i] = simdgroup_matrix<float, 8, 8>(0);
    }

    const ushort tm = 8 * (simdgroup_index / Q4_AFFINE_QMM_WN);
    const ushort tn = 8 * (simdgroup_index % Q4_AFFINE_QMM_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * Q4_AFFINE_QMM_LHS_LD;
    const ushort rhs_offset = (tn + sn) * Q4_AFFINE_QMM_NT_RHS_LD + sm;

    const uint bytes_per_row = in_dim / 2;
    const uint group_count = in_dim / 64;
    const device uchar *gate_weight_bytes = reinterpret_cast<const device uchar *>(gate_weight);
    const device uchar *up_weight_bytes = reinterpret_cast<const device uchar *>(up_weight);

    for (uint k_base = 0; k_base < in_dim; k_base += Q4_AFFINE_QMM_TILE_K) {
        for (uint load = 0; load < Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD; load += 1) {
            const uint lhs_index = local_tid * Q4_AFFINE_QMM_LHS_LOADS_PER_THREAD + load;
            const uint lhs_row = lhs_index / Q4_AFFINE_QMM_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * Q4_AFFINE_QMM_TILE_K;
            const uint global_row = row_base + lhs_row;
            const uint global_col = k_base + lhs_col;
            lhs_tile[lhs_row * Q4_AFFINE_QMM_LHS_LD + lhs_col] =
                (global_row < input_token_count) ? input[(ulong)global_row * in_dim + global_col] : 0.0f;
        }

        for (uint load = 0; load < Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD; load += 1) {
            const uint rhs_pack_index = local_tid * Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD + load;
            const uint local_out = rhs_pack_index / (Q4_AFFINE_QMM_TILE_K / 2);
            const uint local_k_byte = rhs_pack_index - local_out * (Q4_AFFINE_QMM_TILE_K / 2);
            const uint global_out = col_base + local_out;
            const uint global_k = k_base + local_k_byte * 2;
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (global_out < out_dim) {
                const uchar packed = gate_weight_bytes[(ulong)global_out * bytes_per_row + global_k / 2];
                const uint metadata_index = global_out * group_count + global_k / 64;
                const float scale = bf16_to_float(gate_scales[metadata_index]);
                const float bias = bf16_to_float(gate_biases[metadata_index]);
                value0 = scale * float(packed & 0x0fu) + bias;
                value1 = scale * float((packed >> 4) & 0x0fu) + bias;
            }
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 0] = value0;
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 1] = value1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *gate_lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *gate_rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < Q4_AFFINE_QMM_TILE_K; kk += 8) {
            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                lhs_simd[i].thread_elements()[0] =
                    gate_lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    gate_lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 1];
            }

            for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                rhs_simd[j].thread_elements()[0] =
                    gate_rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + 0];
                rhs_simd[j].thread_elements()[1] =
                    gate_rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + Q4_AFFINE_QMM_NT_RHS_LD];
            }

            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                    const uint j_serp = (i % 2) ? (Q4_AFFINE_QMM_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        gate_results[i * Q4_AFFINE_QMM_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        gate_results[i * Q4_AFFINE_QMM_TN + j_serp]
                    );
                }
            }

            gate_lhs_block += 8;
            gate_rhs_block += 8;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint load = 0; load < Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD; load += 1) {
            const uint rhs_pack_index = local_tid * Q4_AFFINE_QMM_RHS_PACK_LOADS_PER_THREAD + load;
            const uint local_out = rhs_pack_index / (Q4_AFFINE_QMM_TILE_K / 2);
            const uint local_k_byte = rhs_pack_index - local_out * (Q4_AFFINE_QMM_TILE_K / 2);
            const uint global_out = col_base + local_out;
            const uint global_k = k_base + local_k_byte * 2;
            float value0 = 0.0f;
            float value1 = 0.0f;
            if (global_out < out_dim) {
                const uchar packed = up_weight_bytes[(ulong)global_out * bytes_per_row + global_k / 2];
                const uint metadata_index = global_out * group_count + global_k / 64;
                const float scale = bf16_to_float(up_scales[metadata_index]);
                const float bias = bf16_to_float(up_biases[metadata_index]);
                value0 = scale * float(packed & 0x0fu) + bias;
                value1 = scale * float((packed >> 4) & 0x0fu) + bias;
            }
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 0] = value0;
            rhs_tile[local_out * Q4_AFFINE_QMM_NT_RHS_LD + local_k_byte * 2 + 1] = value1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *up_lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *up_rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < Q4_AFFINE_QMM_TILE_K; kk += 8) {
            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                lhs_simd[i].thread_elements()[0] =
                    up_lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    up_lhs_block[i * Q4_AFFINE_QMM_TM_STRIDE * Q4_AFFINE_QMM_LHS_LD + 1];
            }

            for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                rhs_simd[j].thread_elements()[0] =
                    up_rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + 0];
                rhs_simd[j].thread_elements()[1] =
                    up_rhs_block[j * Q4_AFFINE_QMM_TN_STRIDE * Q4_AFFINE_QMM_NT_RHS_LD + Q4_AFFINE_QMM_NT_RHS_LD];
            }

            for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
                for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
                    const uint j_serp = (i % 2) ? (Q4_AFFINE_QMM_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        up_results[i * Q4_AFFINE_QMM_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        up_results[i * Q4_AFFINE_QMM_TN + j_serp]
                    );
                }
            }

            up_lhs_block += 8;
            up_rhs_block += 8;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= token_count || col_offset >= out_dim) return;

    device float *dst = output + (ulong)row_offset * out_dim + col_offset;
    const uint2 dst_tile_dims = uint2(
        min(Q4_AFFINE_QMM_TILE_N, out_dim - col_base) - (tn + sn),
        min(Q4_AFFINE_QMM_TILE_M, token_count - row_base) - (tm + sm)
    );
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < Q4_AFFINE_QMM_TM; i += 1) {
        if (i * Q4_AFFINE_QMM_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < Q4_AFFINE_QMM_TN; j += 1) {
            thread const auto &gate_accum = gate_results[i * Q4_AFFINE_QMM_TN + j].thread_elements();
            thread const auto &up_accum = up_results[i * Q4_AFFINE_QMM_TN + j].thread_elements();
            const uint offset = (i * Q4_AFFINE_QMM_TM_STRIDE) * out_dim +
                j * Q4_AFFINE_QMM_TN_STRIDE;
            if (j * Q4_AFFINE_QMM_TN_STRIDE < dst_tile_dims.x) {
                const float gate = gate_accum[0];
                const float up = up_accum[0];
                const float silu = gate / (1.0f + exp(-gate));
                dst[offset] = silu * up;
            }
            if (j * Q4_AFFINE_QMM_TN_STRIDE + 1 < dst_tile_dims.x) {
                const float gate = gate_accum[1];
                const float up = up_accum[1];
                const float silu = gate / (1.0f + exp(-gate));
                dst[offset + 1] = silu * up;
            }
        }
    }
}

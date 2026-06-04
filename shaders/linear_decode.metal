#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

constant uint Q4_DECODE_SIMDGROUP_WIDTH = 32;
constant uint Q4_DECODE_THREADS_PER_THREADGROUP = 64;
constant uint Q4_DECODE_RESULTS_PER_SIMDGROUP = 4;
constant uint Q4_DECODE_SIMDGROUPS_PER_THREADGROUP = 2;
constant uint Q4_DECODE_RESULTS_PER_THREADGROUP =
    Q4_DECODE_RESULTS_PER_SIMDGROUP * Q4_DECODE_SIMDGROUPS_PER_THREADGROUP;

#if defined(__HAVE_BFLOAT__)
METAL_FUNC void q4_decode_group64_compute4_bf16(
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

    for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
        result[row] = 0.0f;
    }

    for (uint k = 0; k < in_dim; k += values_per_thread * Q4_DECODE_SIMDGROUP_WIDTH) {
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
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
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

    for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
        result[row] = simd_sum(result[row]);
    }
}

METAL_FUNC void q4_decode_group64_write4_residual_add_bf16(
    device const bfloat *input,
    device const uint *weight,
    device const ushort *scales,
    device const ushort *biases,
    device const bfloat *residual,
    device bfloat *output,
    uint output_offset,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id
) {
    float result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    q4_decode_group64_compute4_bf16(
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
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint output_index = output_offset + out_base + row;
            const bfloat rounded_dot = bfloat(result[row]);
            output[output_index] = bfloat(float(rounded_dot) + float(residual[output_index]));
        }
    }
}

kernel void linear_q4_affine_group64_qmv_fast_residual_add_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *weight [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device const bfloat *residual [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &out_dim [[buffer(7)]],
    constant uint &in_dim [[buffer(8)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_DECODE_THREADS_PER_THREADGROUP ||
        token_count == 0 || in_dim % 512 != 0 ||
        out_dim % Q4_DECODE_RESULTS_PER_THREADGROUP != 0) return;

    const uint groups_per_token = out_dim / Q4_DECODE_RESULTS_PER_THREADGROUP;
    const uint token_index = threadgroup_index / groups_per_token;
    if (token_index >= token_count) return;
    const uint local_group_index = threadgroup_index - token_index * groups_per_token;
    const uint out_base = local_group_index * Q4_DECODE_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_DECODE_RESULTS_PER_SIMDGROUP;
    q4_decode_group64_write4_residual_add_bf16(
        input + (ulong)token_index * in_dim,
        weight,
        scales,
        biases,
        residual,
        output,
        token_index * out_dim,
        out_base,
        in_dim,
        simd_lane_id
    );
}

METAL_FUNC void q4_decode_group64_write4_silu_gate_bf16(
    device const bfloat *input,
    device const uint *gate_weight,
    device const ushort *gate_scales,
    device const ushort *gate_biases,
    device const uint *up_weight,
    device const ushort *up_scales,
    device const ushort *up_biases,
    device bfloat *output,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id
) {
    float gate_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    float up_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    q4_decode_group64_compute4_bf16(
        input,
        gate_weight,
        gate_scales,
        gate_biases,
        out_base,
        in_dim,
        simd_lane_id,
        gate_result
    );
    q4_decode_group64_compute4_bf16(
        input,
        up_weight,
        up_scales,
        up_biases,
        out_base,
        in_dim,
        simd_lane_id,
        up_result
    );

    if (simd_lane_id == 0) {
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint out_index = out_base + row;
            const float gate = float(bfloat(gate_result[row]));
            const float up = float(bfloat(up_result[row]));
            const float silu = gate / (1.0f + exp(-gate));
            output[out_index] = bfloat(silu * up);
        }
    }
}

METAL_FUNC void q4_decode_group64_write4_silu_gate_h4096_bf16(
    device const bfloat *input,
    device const uint *gate_weight,
    device const ushort *gate_scales,
    device const ushort *gate_biases,
    device const uint *up_weight,
    device const ushort *up_scales,
    device const ushort *up_biases,
    device bfloat *output,
    uint out_base,
    ushort simd_lane_id
) {
    float gate_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    float up_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    constexpr uint values_per_thread = 16;
    constexpr uint in_dim = 4096;
    constexpr uint bytes_per_row = in_dim / 2;
    constexpr uint group_count = in_dim / 64;
    const uint lane = uint(simd_lane_id);
    const device uchar *gate_weight_bytes = reinterpret_cast<const device uchar *>(gate_weight);
    const device uchar *up_weight_bytes = reinterpret_cast<const device uchar *>(up_weight);

    for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
        gate_result[row] = 0.0f;
        up_result[row] = 0.0f;
    }

    for (uint chunk = 0; chunk < 8; chunk += 1) {
        const uint k = chunk * values_per_thread * Q4_DECODE_SIMDGROUP_WIDTH;
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
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint out_index = out_base + row;
            const uint metadata_index = out_index * group_count + group_index;
            const uint weight_base = out_index * bytes_per_row + k_byte_offset;
            const device ushort *gate_packed16 = reinterpret_cast<const device ushort *>(gate_weight_bytes + weight_base);
            const device ushort *up_packed16 = reinterpret_cast<const device ushort *>(up_weight_bytes + weight_base);
            float gate_quant_acc = 0.0f;
            float up_quant_acc = 0.0f;
            for (uint pack_index = 0; pack_index < 4; pack_index += 1) {
                const ushort gate_packed = gate_packed16[pack_index];
                const ushort up_packed = up_packed16[pack_index];
                const uint x_base = pack_index * 4;
                gate_quant_acc += float(gate_packed & 0x000f) * x_thread[x_base + 0];
                gate_quant_acc += float(gate_packed & 0x00f0) * x_thread[x_base + 1];
                gate_quant_acc += float(gate_packed & 0x0f00) * x_thread[x_base + 2];
                gate_quant_acc += float(gate_packed & 0xf000) * x_thread[x_base + 3];
                up_quant_acc += float(up_packed & 0x000f) * x_thread[x_base + 0];
                up_quant_acc += float(up_packed & 0x00f0) * x_thread[x_base + 1];
                up_quant_acc += float(up_packed & 0x0f00) * x_thread[x_base + 2];
                up_quant_acc += float(up_packed & 0xf000) * x_thread[x_base + 3];
            }

            gate_result[row] += bf16_to_float(gate_scales[metadata_index]) * gate_quant_acc +
                bf16_to_float(gate_biases[metadata_index]) * input_sum;
            up_result[row] += bf16_to_float(up_scales[metadata_index]) * up_quant_acc +
                bf16_to_float(up_biases[metadata_index]) * input_sum;
        }
    }

    for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
        gate_result[row] = simd_sum(gate_result[row]);
        up_result[row] = simd_sum(up_result[row]);
    }

    if (simd_lane_id == 0) {
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint out_index = out_base + row;
            const float gate = float(bfloat(gate_result[row]));
            const float up = float(bfloat(up_result[row]));
            const float silu = gate / (1.0f + exp(-gate));
            output[out_index] = bfloat(silu * up);
        }
    }
}

kernel void linear_vec_q4_affine_group64_multi2_silu_gate_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *gate_weight [[buffer(1)]],
    device const ushort *gate_scales [[buffer(2)]],
    device const ushort *gate_biases [[buffer(3)]],
    device const uint *up_weight [[buffer(4)]],
    device const ushort *up_scales [[buffer(5)]],
    device const ushort *up_biases [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant uint &out_dim [[buffer(8)]],
    constant uint &in_dim [[buffer(9)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_DECODE_THREADS_PER_THREADGROUP ||
        in_dim % 512 != 0 || out_dim % Q4_DECODE_RESULTS_PER_THREADGROUP != 0) return;
    const uint out_base = threadgroup_index * Q4_DECODE_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_DECODE_RESULTS_PER_SIMDGROUP;
    q4_decode_group64_write4_silu_gate_bf16(
        input,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        output,
        out_base,
        in_dim,
        simd_lane_id
    );
}

kernel void linear_vec_q4_affine_group64_multi2_silu_gate_h4096_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *gate_weight [[buffer(1)]],
    device const ushort *gate_scales [[buffer(2)]],
    device const ushort *gate_biases [[buffer(3)]],
    device const uint *up_weight [[buffer(4)]],
    device const ushort *up_scales [[buffer(5)]],
    device const ushort *up_biases [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant uint &out_dim [[buffer(8)]],
    constant uint &in_dim [[buffer(9)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_DECODE_THREADS_PER_THREADGROUP ||
        in_dim != 4096 || out_dim % Q4_DECODE_RESULTS_PER_THREADGROUP != 0) return;
    const uint out_base = threadgroup_index * Q4_DECODE_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_DECODE_RESULTS_PER_SIMDGROUP;
    q4_decode_group64_write4_silu_gate_h4096_bf16(
        input,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        output,
        out_base,
        simd_lane_id
    );
}
#endif

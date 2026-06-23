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

METAL_FUNC void q4_decode_group64_compute4x2_bf16(
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

    for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
        result0[row] = 0.0f;
        result1[row] = 0.0f;
    }

    for (uint k = 0; k < in_dim; k += values_per_thread * Q4_DECODE_SIMDGROUP_WIDTH) {
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
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
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

    for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
        result0[row] = simd_sum(result0[row]);
        result1[row] = simd_sum(result1[row]);
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

METAL_FUNC void q4_decode_group64_write4x2_residual_add_bf16(
    device const bfloat *input0,
    device const bfloat *input1,
    device const uint *weight,
    device const ushort *scales,
    device const ushort *biases,
    device const bfloat *residual0,
    device const bfloat *residual1,
    device bfloat *output0,
    device bfloat *output1,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id
) {
    float result0[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    float result1[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    q4_decode_group64_compute4x2_bf16(
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

    if (simd_lane_id == 0) {
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint output_index = out_base + row;
            const bfloat rounded0 = bfloat(result0[row]);
            const bfloat rounded1 = bfloat(result1[row]);
            output0[output_index] = bfloat(float(rounded0) + float(residual0[output_index]));
            output1[output_index] = bfloat(float(rounded1) + float(residual1[output_index]));
        }
    }
}

kernel void linear_q4_affine_group64_qmv2_fast_residual_add_bf16(
    device const bfloat *input0 [[buffer(0)]],
    device const bfloat *input1 [[buffer(1)]],
    device const uint *weight [[buffer(2)]],
    device const ushort *scales [[buffer(3)]],
    device const ushort *biases [[buffer(4)]],
    device const bfloat *residual0 [[buffer(5)]],
    device const bfloat *residual1 [[buffer(6)]],
    device bfloat *output0 [[buffer(7)]],
    device bfloat *output1 [[buffer(8)]],
    constant uint &out_dim [[buffer(9)]],
    constant uint &in_dim [[buffer(10)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_DECODE_THREADS_PER_THREADGROUP ||
        in_dim % 512 != 0 || out_dim % Q4_DECODE_RESULTS_PER_THREADGROUP != 0) return;

    const uint out_base = threadgroup_index * Q4_DECODE_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_DECODE_RESULTS_PER_SIMDGROUP;
    q4_decode_group64_write4x2_residual_add_bf16(
        input0,
        input1,
        weight,
        scales,
        biases,
        residual0,
        residual1,
        output0,
        output1,
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

METAL_FUNC void q4_decode_group64_write4x2_silu_gate_bf16(
    device const bfloat *input0,
    device const bfloat *input1,
    device const uint *gate_weight,
    device const ushort *gate_scales,
    device const ushort *gate_biases,
    device const uint *up_weight,
    device const ushort *up_scales,
    device const ushort *up_biases,
    device bfloat *output0,
    device bfloat *output1,
    uint out_base,
    uint in_dim,
    ushort simd_lane_id
) {
    float gate0_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    float gate1_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    float up0_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    float up1_result[Q4_DECODE_RESULTS_PER_SIMDGROUP];
    q4_decode_group64_compute4x2_bf16(
        input0,
        input1,
        gate_weight,
        gate_scales,
        gate_biases,
        out_base,
        in_dim,
        simd_lane_id,
        gate0_result,
        gate1_result
    );
    q4_decode_group64_compute4x2_bf16(
        input0,
        input1,
        up_weight,
        up_scales,
        up_biases,
        out_base,
        in_dim,
        simd_lane_id,
        up0_result,
        up1_result
    );

    if (simd_lane_id == 0) {
        for (uint row = 0; row < Q4_DECODE_RESULTS_PER_SIMDGROUP; row += 1) {
            const uint out_index = out_base + row;
            const float gate0 = float(bfloat(gate0_result[row]));
            const float gate1 = float(bfloat(gate1_result[row]));
            const float up0 = float(bfloat(up0_result[row]));
            const float up1 = float(bfloat(up1_result[row]));
            const float silu0 = gate0 / (1.0f + exp(-gate0));
            const float silu1 = gate1 / (1.0f + exp(-gate1));
            output0[out_index] = bfloat(silu0 * up0);
            output1[out_index] = bfloat(silu1 * up1);
        }
    }
}

kernel void linear_vec_q4_affine_group64_multi2_silu_gate2_bf16(
    device const bfloat *input0 [[buffer(0)]],
    device const bfloat *input1 [[buffer(1)]],
    device const uint *gate_weight [[buffer(2)]],
    device const ushort *gate_scales [[buffer(3)]],
    device const ushort *gate_biases [[buffer(4)]],
    device const uint *up_weight [[buffer(5)]],
    device const ushort *up_scales [[buffer(6)]],
    device const ushort *up_biases [[buffer(7)]],
    device bfloat *output0 [[buffer(8)]],
    device bfloat *output1 [[buffer(9)]],
    constant uint &out_dim [[buffer(10)]],
    constant uint &in_dim [[buffer(11)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]]
) {
    if (threads_per_threadgroup != Q4_DECODE_THREADS_PER_THREADGROUP ||
        in_dim % 512 != 0 || out_dim % Q4_DECODE_RESULTS_PER_THREADGROUP != 0) return;
    const uint out_base = threadgroup_index * Q4_DECODE_RESULTS_PER_THREADGROUP +
        uint(simdgroup_index) * Q4_DECODE_RESULTS_PER_SIMDGROUP;
    q4_decode_group64_write4x2_silu_gate_bf16(
        input0,
        input1,
        gate_weight,
        gate_scales,
        gate_biases,
        up_weight,
        up_scales,
        up_biases,
        output0,
        output1,
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

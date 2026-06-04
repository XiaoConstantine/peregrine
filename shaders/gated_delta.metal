#include <metal_stdlib>
using namespace metal;

// Gated-DeltaNet recurrence (the linear-attention scan), per value head hv
// with state[hv] a [Dv, Dk] matrix. Verified against MLX's gated_delta_ops
// and bitwise-identical to the prior one-row-per-simdgroup kernel. For each
// (hv, dv):
//   state_row *= g[hv]                                    (decay)
//   kv_mem     = dot(state_row, k[hk])                    (hk = hv / (Hv/Hk), GQA)
//   delta      = (v[hv,dv] - kv_mem) * beta[hv]
//   state_row += k[hk] * delta                            (rank-1 update)
//   y[hv,dv]   = dot(state_row, q[hk])
//
// Each simdgroup owns ROWS consecutive dv rows so the q/k fragment loads are
// amortized across rows: the one-row shape re-read the same head's q/k rows
// Dv times per token and measured 14.0ms per 1600-token serving chunk on the
// M3 Pro gate machine; this shape measures 7.3ms with identical outputs.
// State lives in registers across the whole token loop. The per-row
// accumulation order (lane-local chunk loop, then one simd_sum per quantity)
// is intentionally unchanged so results stay bitwise-equal to the prior
// serving kernel. Decode reuses this kernel with tokens=1.

constant uint GATED_DELTA_H128_THREADS_PER_SIMDGROUP = 32;
constant uint GATED_DELTA_H128_KEY_CHUNKS = 4;
constant uint GATED_DELTA_H128_ROWS_PER_SIMDGROUP = 8;
constant uint GATED_DELTA_H128_SIMDGROUPS_PER_THREADGROUP = 4;
constant uint GATED_DELTA_PREFILL_VALUE_MAJOR_THREADS_PER_THREADGROUP =
    GATED_DELTA_H128_THREADS_PER_SIMDGROUP * GATED_DELTA_H128_SIMDGROUPS_PER_THREADGROUP;

#if defined(__HAVE_BFLOAT__)
// Qwen3.5 h128 prefill recurrence for prepared q/k norm and prepared gates.
// q/k: [tokens, Hk, Dk] f32, conv_out: [tokens, 2*Hk*Dk + Hv*Dv] bf16.
// Reads v directly from conv_out and updates Peregrine's [Hv, Dv, Dk] f32
// state; outputs y in bf16.
kernel void gated_delta_prefill_h128_norm_prepared_value_major_r8_bf16(
    device float *state [[buffer(0)]],
    device const float *q [[buffer(1)]],
    device const float *k [[buffer(2)]],
    device const bfloat *conv_out [[buffer(3)]],
    device const float *g [[buffer(4)]],
    device const float *beta [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    constant uint &Hk [[buffer(8)]],
    constant uint &Hv [[buffer(9)]],
    constant uint &Dk [[buffer(10)]],
    constant uint &Dv [[buffer(11)]],
    constant uint &conv_dim [[buffer(12)]],
    constant uint &has_previous_state [[buffer(13)]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    if (tokens == 0 || Hk == 0 || Hv == 0 || Dk != 128 || Dv == 0) return;
    if (Hv % Hk != 0) return;
    if (threads_per_threadgroup != GATED_DELTA_PREFILL_VALUE_MAJOR_THREADS_PER_THREADGROUP) return;
    const uint key_dim = Hk * Dk;
    const uint value_dim = Hv * Dv;
    if (conv_dim < 2 * key_dim + value_dim) return;
    const uint rows_per_threadgroup =
        GATED_DELTA_H128_ROWS_PER_SIMDGROUP * GATED_DELTA_H128_SIMDGROUPS_PER_THREADGROUP;
    if (Dv % rows_per_threadgroup != 0) return;

    const uint tiles_per_head = Dv / rows_per_threadgroup;
    if (threadgroup_index >= Hv * tiles_per_head) return;

    const uint hv = threadgroup_index / tiles_per_head;
    const uint tile_index = threadgroup_index - hv * tiles_per_head;
    const uint dv0 = tile_index * rows_per_threadgroup +
        uint(simdgroup_index) * GATED_DELTA_H128_ROWS_PER_SIMDGROUP;
    const uint hk = hv / (Hv / Hk);
    const uint lane = uint(lane_id);
    const ulong value_base = (ulong)hv * Dv;

    float state_local[GATED_DELTA_H128_ROWS_PER_SIMDGROUP][GATED_DELTA_H128_KEY_CHUNKS];
    for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
        const ulong row_base = (ulong)(hv * Dv + dv0 + r) * Dk;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            const uint dk = lane + chunk * GATED_DELTA_H128_THREADS_PER_SIMDGROUP;
            state_local[r][chunk] = has_previous_state != 0 ? state[row_base + dk] : 0.0f;
        }
    }

    for (uint t = 0; t < tokens; t++) {
        const ulong qk_base = ((ulong)t * Hk + hk) * Dk;
        const ulong gate_offset = (ulong)t * Hv + hv;
        const ulong conv_value_base = (ulong)t * conv_dim + 2 * (ulong)key_dim + value_base + dv0;
        const ulong y_base = (ulong)t * Hv * Dv + value_base + dv0;
        const float decay = g[gate_offset];
        const float beta_value = beta[gate_offset];

        float key_local[GATED_DELTA_H128_KEY_CHUNKS];
        float query_local[GATED_DELTA_H128_KEY_CHUNKS];
        float kq_local = 0.0f;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            const uint dk = lane + chunk * GATED_DELTA_H128_THREADS_PER_SIMDGROUP;
            key_local[chunk] = k[qk_base + dk];
            query_local[chunk] = q[qk_base + dk];
            kq_local += key_local[chunk] * query_local[chunk];
        }
        const float kq = simd_sum(kq_local);

        float decayed[GATED_DELTA_H128_ROWS_PER_SIMDGROUP][GATED_DELTA_H128_KEY_CHUNKS];
        float kv_mem[GATED_DELTA_H128_ROWS_PER_SIMDGROUP];
        float output_base[GATED_DELTA_H128_ROWS_PER_SIMDGROUP];
        for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
            float kv_mem_local = 0.0f;
            float output_base_local = 0.0f;
            for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
                const float decayed_value = state_local[r][chunk] * decay;
                decayed[r][chunk] = decayed_value;
                kv_mem_local += decayed_value * key_local[chunk];
                output_base_local += decayed_value * query_local[chunk];
            }
            kv_mem[r] = simd_sum(kv_mem_local);
            output_base[r] = simd_sum(output_base_local);
        }

        for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
            const float delta = (float(conv_out[conv_value_base + r]) - kv_mem[r]) * beta_value;
            for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
                state_local[r][chunk] = decayed[r][chunk] + key_local[chunk] * delta;
            }
            if (lane == 0) {
                y[y_base + r] = bfloat(output_base[r] + delta * kq);
            }
        }
    }

    for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
        const ulong row_base = (ulong)(hv * Dv + dv0 + r) * Dk;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            const uint dk = lane + chunk * GATED_DELTA_H128_THREADS_PER_SIMDGROUP;
            state[row_base + dk] = state_local[r][chunk];
        }
    }
}
#endif

#if defined(__HAVE_BFLOAT__)
// One-token decode variant with the q/k L2 norms and gate coefficients
// computed inline, removing the separate qk-norm and gating dispatches (and
// one barrier level) from the decode chain. Math follows
// qk_l2norm_prefill_bf16 and gating_many_bf16; layout matches the r8 prefill
// kernel with tokens=1.
kernel void gated_delta_decode_h128_fused_bf16(
    device float *state [[buffer(0)]],
    device const bfloat *conv_out [[buffer(1)]],
    device const bfloat *a [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device const float *A_log [[buffer(4)]],
    device const bfloat *dt_bias [[buffer(5)]],
    device bfloat *y [[buffer(6)]],
    constant uint &Hk [[buffer(7)]],
    constant uint &Hv [[buffer(8)]],
    constant uint &Dk [[buffer(9)]],
    constant uint &Dv [[buffer(10)]],
    constant uint &conv_dim [[buffer(11)]],
    constant uint &has_previous_state [[buffer(12)]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    if (Hk == 0 || Hv == 0 || Dk != 128 || Dv == 0) return;
    if (Hv % Hk != 0) return;
    if (threads_per_threadgroup != GATED_DELTA_PREFILL_VALUE_MAJOR_THREADS_PER_THREADGROUP) return;
    const uint key_dim = Hk * Dk;
    const uint value_dim = Hv * Dv;
    if (conv_dim < 2 * key_dim + value_dim) return;
    const uint rows_per_threadgroup =
        GATED_DELTA_H128_ROWS_PER_SIMDGROUP * GATED_DELTA_H128_SIMDGROUPS_PER_THREADGROUP;
    if (Dv % rows_per_threadgroup != 0) return;

    const uint tiles_per_head = Dv / rows_per_threadgroup;
    if (threadgroup_index >= Hv * tiles_per_head) return;

    const uint hv = threadgroup_index / tiles_per_head;
    const uint tile_index = threadgroup_index - hv * tiles_per_head;
    const uint dv0 = tile_index * rows_per_threadgroup +
        uint(simdgroup_index) * GATED_DELTA_H128_ROWS_PER_SIMDGROUP;
    const uint hk = hv / (Hv / Hk);
    const uint lane = uint(lane_id);

    // gate coefficients (gating_many_bf16 math, computed redundantly per lane)
    const float gate_x = float(a[hv]) + float(dt_bias[hv]);
    const float softplus = max(gate_x, 0.0f) + log(1.0f + exp(-abs(gate_x)));
    const float decay = exp(-exp(A_log[hv]) * softplus);
    const float beta_value = 1.0f / (1.0f + exp(-float(b[hv])));

    // q/k L2 norms (qk_l2norm_prefill_bf16 math) inline from conv_out
    float q_raw[GATED_DELTA_H128_KEY_CHUNKS];
    float k_raw[GATED_DELTA_H128_KEY_CHUNKS];
    float q_ss_local = 0.0f;
    float k_ss_local = 0.0f;
    for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
        const uint dk = lane + chunk * GATED_DELTA_H128_THREADS_PER_SIMDGROUP;
        q_raw[chunk] = float(conv_out[(ulong)hk * Dk + dk]);
        k_raw[chunk] = float(conv_out[(ulong)key_dim + (ulong)hk * Dk + dk]);
        q_ss_local += q_raw[chunk] * q_raw[chunk];
        k_ss_local += k_raw[chunk] * k_raw[chunk];
    }
    const float q_inv_norm = rsqrt(simd_sum(q_ss_local) + 1.0e-6f) * rsqrt(float(Dk));
    const float k_inv_norm = rsqrt(simd_sum(k_ss_local) + 1.0e-6f);

    float key_local[GATED_DELTA_H128_KEY_CHUNKS];
    float query_local[GATED_DELTA_H128_KEY_CHUNKS];
    float kq_local = 0.0f;
    for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
        query_local[chunk] = q_raw[chunk] * q_inv_norm;
        key_local[chunk] = k_raw[chunk] * k_inv_norm;
        kq_local += key_local[chunk] * query_local[chunk];
    }
    const float kq = simd_sum(kq_local);

    float state_local[GATED_DELTA_H128_ROWS_PER_SIMDGROUP][GATED_DELTA_H128_KEY_CHUNKS];
    for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
        const ulong row_base = (ulong)(hv * Dv + dv0 + r) * Dk;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            const uint dk = lane + chunk * GATED_DELTA_H128_THREADS_PER_SIMDGROUP;
            state_local[r][chunk] = has_previous_state != 0 ? state[row_base + dk] : 0.0f;
        }
    }

    const ulong conv_value_base = 2 * (ulong)key_dim + (ulong)hv * Dv + dv0;
    float decayed[GATED_DELTA_H128_ROWS_PER_SIMDGROUP][GATED_DELTA_H128_KEY_CHUNKS];
    float kv_mem[GATED_DELTA_H128_ROWS_PER_SIMDGROUP];
    float output_base[GATED_DELTA_H128_ROWS_PER_SIMDGROUP];
    for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
        float kv_mem_local = 0.0f;
        float output_base_local = 0.0f;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            const float decayed_value = state_local[r][chunk] * decay;
            decayed[r][chunk] = decayed_value;
            kv_mem_local += decayed_value * key_local[chunk];
            output_base_local += decayed_value * query_local[chunk];
        }
        kv_mem[r] = simd_sum(kv_mem_local);
        output_base[r] = simd_sum(output_base_local);
    }

    const ulong y_base = (ulong)hv * Dv + dv0;
    for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
        const float delta = (float(conv_out[conv_value_base + r]) - kv_mem[r]) * beta_value;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            state_local[r][chunk] = decayed[r][chunk] + key_local[chunk] * delta;
        }
        if (lane == 0) {
            y[y_base + r] = bfloat(output_base[r] + delta * kq);
        }
    }

    for (uint r = 0; r < GATED_DELTA_H128_ROWS_PER_SIMDGROUP; r++) {
        const ulong row_base = (ulong)(hv * Dv + dv0 + r) * Dk;
        for (uint chunk = 0; chunk < GATED_DELTA_H128_KEY_CHUNKS; chunk++) {
            const uint dk = lane + chunk * GATED_DELTA_H128_THREADS_PER_SIMDGROUP;
            state[row_base + dk] = state_local[r][chunk];
        }
    }
}
#endif

#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

// One-token decode attention (flash-decoding shape) over Peregrine's
// full-attention cache layout: K is [num_kv_heads, stride, head_dim], V is
// [num_kv_heads, head_dim, stride]. The prior one-thread-per-q-head kernel
// scanned the whole context serially and measured 1.08s at 16k context; this
// split-context pair measures 0.49ms (134 GB/s, memory-bound) on the gate
// machine and replaces both the serial kernel and the tiled prefill-path
// detour that long-context decode previously used. Accumulation stays in f32
// and only the final context output is rounded to bf16. The non-prefix decode
// path binds its local cache as the prefix buffers with prefix_len=0.
constant uint QWEN35_DECODE_HEAD_DIM = 256;
constant uint QWEN35_DECODE_GROUP = 4;          // q heads per kv head
constant uint QWEN35_DECODE_SIMDGROUPS = 8;
constant uint QWEN35_DECODE_CHUNK_TOKENS = 1024;
constant uint QWEN35_DECODE_TILE_TOKENS = 64;

#if defined(__HAVE_BFLOAT__)
// Tiled flash-decoding partials: phase A scores a 64-token tile (coalesced K
// reads), phase B then streams each V row contiguously across the tile so the
// value-dim-major layout reads whole cache lines instead of 2-byte gathers.
kernel void sdpa_decode_prefix_flash_partials_bf16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *prefix_k [[buffer(1)]],
    device const bfloat *prefix_v [[buffer(2)]],
    device const bfloat *local_k [[buffer(3)]],
    device const bfloat *local_v [[buffer(4)]],
    device float *partials [[buffer(5)]],
    constant uint &num_q_heads [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &seq_len [[buffer(9)]],
    constant uint &prefix_len [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant uint &prefix_stride [[buffer(12)]],
    constant uint &local_stride [[buffer(13)]],
    constant uint &chunk_count [[buffer(14)]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sg_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]])
{
    if (num_kv_heads == 0 || num_q_heads != num_kv_heads * QWEN35_DECODE_GROUP) return;
    if (head_dim != QWEN35_DECODE_HEAD_DIM) return;
    if (threads_per_threadgroup != QWEN35_DECODE_SIMDGROUPS * 32) return;
    if (tg_id >= num_kv_heads * chunk_count) return;

    const uint kvh = tg_id / chunk_count;
    const uint chunk = tg_id - kvh * chunk_count;
    const uint t_begin = chunk * QWEN35_DECODE_CHUNK_TOKENS;
    const uint t_end = min(t_begin + QWEN35_DECODE_CHUNK_TOKENS, seq_len);
    const uint lane = uint(lane_id);
    const uint sg = uint(sg_id);

    const device bfloat *prefix_kbase = prefix_k + (ulong)kvh * prefix_stride * head_dim;
    const device bfloat *prefix_vbase = prefix_v + (ulong)kvh * head_dim * prefix_stride;
    const device bfloat *local_kbase = local_k + (ulong)kvh * local_stride * head_dim;
    const device bfloat *local_vbase = local_v + (ulong)kvh * head_dim * local_stride;

    float4 q_local[QWEN35_DECODE_GROUP][2];
    for (uint g = 0; g < QWEN35_DECODE_GROUP; g++) {
        const device bfloat4 *qv = (const device bfloat4 *)(q + (ulong)(kvh * QWEN35_DECODE_GROUP + g) * head_dim);
        q_local[g][0] = float4(qv[lane]);
        q_local[g][1] = float4(qv[32 + lane]);
    }

    float m[QWEN35_DECODE_GROUP], denom[QWEN35_DECODE_GROUP];
    // acc4[j] is a float4 over the QWEN35_DECODE_GROUP q heads for this lane's j-th dim:
    // j < 4 -> d = lane*4 + j, else d = 128 + lane*4 + (j-4).
    float4 acc4[8];
    for (uint g = 0; g < QWEN35_DECODE_GROUP; g++) { m[g] = -INFINITY; denom[g] = 0.0f; }
    for (uint j = 0; j < 8; j++) acc4[j] = float4(0.0f);

    threadgroup float4 w_tile[QWEN35_DECODE_SIMDGROUPS][QWEN35_DECODE_TILE_TOKENS]; // scores then weights, per g lane of float4
    const uint tiles_in_chunk = QWEN35_DECODE_CHUNK_TOKENS / QWEN35_DECODE_TILE_TOKENS;

    for (uint tile = sg; tile < tiles_in_chunk; tile += QWEN35_DECODE_SIMDGROUPS) {
        const uint tile_base = t_begin + tile * QWEN35_DECODE_TILE_TOKENS;
        if (tile_base >= t_end) break;
        const uint valid = min((uint)QWEN35_DECODE_TILE_TOKENS, t_end - tile_base);

        // phase A: scores for the tile
        for (uint tt = 0; tt < QWEN35_DECODE_TILE_TOKENS; tt++) {
            const uint t = tile_base + tt;
            float4 s4 = float4(-INFINITY);
            if (tt < valid) {
                const bool use_prefix = t < prefix_len;
                const uint local_t = use_prefix ? 0 : t - prefix_len;
                const device bfloat *kt = use_prefix ?
                    prefix_kbase + (ulong)t * head_dim :
                    local_kbase + (ulong)local_t * head_dim;
                const device bfloat4 *kv4 = (const device bfloat4 *)kt;
                const float4 k0 = float4(kv4[lane]);
                const float4 k1 = float4(kv4[32 + lane]);
                float4 dots;
                for (uint g = 0; g < QWEN35_DECODE_GROUP; g++) {
                    const float4 p = q_local[g][0] * k0 + q_local[g][1] * k1;
                    dots[g] = p.x + p.y + p.z + p.w;
                }
                s4 = simd_sum(dots) * scale;
            }
            if (lane == (tt & 31u)) w_tile[sg][tt] = s4;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // phase A2: online-softmax bookkeeping for the tile
        float4 corr;
        {
            float4 s_a = w_tile[sg][lane];
            float4 s_b = w_tile[sg][32 + lane];
            float4 tile_max;
            for (uint g = 0; g < QWEN35_DECODE_GROUP; g++) {
                tile_max[g] = simd_max(max(s_a[g], s_b[g]));
            }
            float4 w_a, w_b;
            for (uint g = 0; g < QWEN35_DECODE_GROUP; g++) {
                const float m_new = max(m[g], tile_max[g]);
                corr[g] = m[g] == -INFINITY ? 0.0f : exp(m[g] - m_new);
                w_a[g] = s_a[g] == -INFINITY ? 0.0f : exp(s_a[g] - m_new);
                w_b[g] = s_b[g] == -INFINITY ? 0.0f : exp(s_b[g] - m_new);
                denom[g] = denom[g] * corr[g] + simd_sum(w_a[g] + w_b[g]);
                m[g] = m_new;
            }
            w_tile[sg][lane] = w_a;
            w_tile[sg][32 + lane] = w_b;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 8; j++) acc4[j] *= corr;

        // phase B: contiguous V rows across the tile
        const bool fully_prefix = tile_base + valid <= prefix_len;
        for (uint j = 0; j < 8; j++) {
            const uint d = (j < 4) ? (lane * 4 + j) : (128 + lane * 4 + (j - 4));
            if (fully_prefix) {
                const device bfloat4 *vrow = (const device bfloat4 *)(prefix_vbase + (ulong)d * prefix_stride + tile_base);
                float4 acc_j = float4(0.0f); // accumulate over g via transpose below
                float4 sum_g = float4(0.0f);
                for (uint tt4 = 0; tt4 < valid / 4; tt4++) {
                    const float4 vv = float4(vrow[tt4]);
                    sum_g += w_tile[sg][tt4 * 4 + 0] * vv.x;
                    sum_g += w_tile[sg][tt4 * 4 + 1] * vv.y;
                    sum_g += w_tile[sg][tt4 * 4 + 2] * vv.z;
                    sum_g += w_tile[sg][tt4 * 4 + 3] * vv.w;
                }
                for (uint tt = (valid / 4) * 4; tt < valid; tt++) {
                    const float vv = float(prefix_vbase[(ulong)d * prefix_stride + tile_base + tt]);
                    sum_g += w_tile[sg][tt] * vv;
                }
                acc4[j] += sum_g;
                (void)acc_j;
            } else {
                float4 sum_g = float4(0.0f);
                for (uint tt = 0; tt < valid; tt++) {
                    const uint t = tile_base + tt;
                    const bool use_prefix = t < prefix_len;
                    const uint local_t = use_prefix ? 0 : t - prefix_len;
                    const float vv = use_prefix ?
                        float(prefix_vbase[(ulong)d * prefix_stride + t]) :
                        float(local_vbase[(ulong)d * local_stride + local_t]);
                    sum_g += w_tile[sg][tt] * vv;
                }
                acc4[j] += sum_g;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // merge the simdgroup partials per q head and emit
    threadgroup float tg_m[QWEN35_DECODE_GROUP][QWEN35_DECODE_SIMDGROUPS];
    threadgroup float tg_denom[QWEN35_DECODE_GROUP][QWEN35_DECODE_SIMDGROUPS];
    threadgroup float tg_acc_full[QWEN35_DECODE_SIMDGROUPS][QWEN35_DECODE_HEAD_DIM];
    for (uint g = 0; g < QWEN35_DECODE_GROUP; g++) {
        if (lane == 0) {
            tg_m[g][sg] = m[g];
            tg_denom[g][sg] = denom[g];
        }
        for (uint j = 0; j < 8; j++) {
            const uint d = (j < 4) ? (lane * 4 + j) : (128 + lane * 4 + (j - 4));
            tg_acc_full[sg][d] = acc4[j][g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float m_all = -INFINITY;
            for (uint i = 0; i < QWEN35_DECODE_SIMDGROUPS; i++) m_all = max(m_all, tg_m[g][i]);
            float denom_all = 0.0f;
            float out_local[8];
            for (uint j = 0; j < 8; j++) out_local[j] = 0.0f;
            for (uint i = 0; i < QWEN35_DECODE_SIMDGROUPS; i++) {
                const float w = tg_m[g][i] == -INFINITY ? 0.0f : exp(tg_m[g][i] - m_all);
                denom_all += tg_denom[g][i] * w;
                for (uint j = 0; j < 8; j++) {
                    const uint d = (j < 4) ? (lane * 4 + j) : (128 + lane * 4 + (j - 4));
                    out_local[j] += tg_acc_full[i][d] * w;
                }
            }
            device float *dst = partials + ((ulong)(kvh * QWEN35_DECODE_GROUP + g) * chunk_count + chunk) * (QWEN35_DECODE_HEAD_DIM + 2);
            if (lane == 0) {
                dst[0] = m_all;
                dst[1] = denom_all;
            }
            for (uint j = 0; j < 8; j++) {
                const uint d = (j < 4) ? (lane * 4 + j) : (128 + lane * 4 + (j - 4));
                dst[2 + d] = out_local[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Pass 2: merge each q head's chunk partials and apply the attention output
// gate (out = merged * sigmoid(gate)) in one dispatch, removing the separate
// sigmoid-mul kernel and its barrier from the decode chain.
kernel void sdpa_decode_prefix_flash_reduce_gated_bf16(
    device const float *partials [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &num_q_heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &chunk_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (head_dim != QWEN35_DECODE_HEAD_DIM) return;
    const uint h = tid / QWEN35_DECODE_HEAD_DIM;
    const uint d = tid - h * QWEN35_DECODE_HEAD_DIM;
    if (h >= num_q_heads) return;

    const device float *base = partials + (ulong)h * chunk_count * (QWEN35_DECODE_HEAD_DIM + 2);
    float m_all = -INFINITY;
    for (uint c = 0; c < chunk_count; c++) m_all = max(m_all, base[c * (QWEN35_DECODE_HEAD_DIM + 2)]);
    float denom_all = 0.0f;
    float acc = 0.0f;
    for (uint c = 0; c < chunk_count; c++) {
        const device float *pc = base + c * (QWEN35_DECODE_HEAD_DIM + 2);
        const float w = pc[0] == -INFINITY ? 0.0f : exp(pc[0] - m_all);
        denom_all += pc[1] * w;
        acc += pc[2 + d] * w;
    }
    const ulong i = (ulong)h * QWEN35_DECODE_HEAD_DIM + d;
    const float sigmoid = 1.0f / (1.0f + exp(-float(gate[i])));
    out[i] = bfloat((acc / denom_all) * sigmoid);
}
#endif

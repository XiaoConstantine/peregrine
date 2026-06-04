#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "bf16.h"
using namespace metal;

constant uint SDPA_PREFILL_TILE_SEQ = 1024;
constant uint SDPA_PREFILL_THREADS = 256;
constant uint QWEN35_QK_PREPARE_THREADS = 256;
constant uint SDPA_PREFILL_D256_TILE_M = 8;
constant uint SDPA_PREFILL_D256_TILE_N = 32;
constant uint SDPA_PREFILL_D256_THREADS = 32;
constant uint SDPA_PREFILL_D256_HEAD_DIM = 256;
constant uint SDPA_PREFILL_D256_Q_LD = SDPA_PREFILL_D256_HEAD_DIM + 8;
constant uint SDPA_PREFILL_D256_K_LD = SDPA_PREFILL_D256_TILE_N + 8;
constant uint SDPA_PREFILL_D256_DSPLIT_HEAD_DIM = 128;
constant uint SDPA_PREFILL_D256_DSPLIT_V_LD = SDPA_PREFILL_D256_DSPLIT_HEAD_DIM + 8;
constant uint PREFILL_SCORE_TILE_M = 32;
constant uint PREFILL_SCORE_TILE_N = 32;
constant uint PREFILL_SCORE_TILE_K = 16;
constant uint PREFILL_SCORE_WM = 2;
constant uint PREFILL_SCORE_WN = 2;
constant uint PREFILL_SCORE_LHS_LD = PREFILL_SCORE_TILE_K + 8;
constant uint PREFILL_SCORE_RHS_LD = PREFILL_SCORE_TILE_N + 8;
constant uint PREFILL_SCORE_TM_STRIDE = 8 * PREFILL_SCORE_WM;
constant uint PREFILL_SCORE_TN_STRIDE = 8 * PREFILL_SCORE_WN;
constant uint PREFILL_SCORE_TM = PREFILL_SCORE_TILE_M / PREFILL_SCORE_TM_STRIDE;
constant uint PREFILL_SCORE_TN = PREFILL_SCORE_TILE_N / PREFILL_SCORE_TN_STRIDE;
constant uint PREFILL_SCORE_THREADS = PREFILL_SCORE_WM * PREFILL_SCORE_WN * 32;
constant uint PREFILL_VALUE_TILE_M = 32;
constant uint PREFILL_VALUE_TILE_N = 32;
constant uint PREFILL_VALUE_TILE_K = 16;
constant uint PREFILL_VALUE_WM = 2;
constant uint PREFILL_VALUE_WN = 2;
constant uint PREFILL_VALUE_LHS_LD = PREFILL_VALUE_TILE_K + 8;
constant uint PREFILL_VALUE_RHS_LD = PREFILL_VALUE_TILE_N + 8;
constant uint PREFILL_VALUE_TM_STRIDE = 8 * PREFILL_VALUE_WM;
constant uint PREFILL_VALUE_TN_STRIDE = 8 * PREFILL_VALUE_WN;
constant uint PREFILL_VALUE_TM = PREFILL_VALUE_TILE_M / PREFILL_VALUE_TM_STRIDE;
constant uint PREFILL_VALUE_TN = PREFILL_VALUE_TILE_N / PREFILL_VALUE_TN_STRIDE;
constant uint PREFILL_VALUE_THREADS = PREFILL_VALUE_WM * PREFILL_VALUE_WN * 32;
constant uint PREFILL_SOFTMAX_THREADS = 256;
constant uint QWEN35_NUM_Q_HEADS = 16;
constant uint QWEN35_NUM_KV_HEADS = 4;
constant uint QWEN35_HEAD_DIM = 256;
constant uint QWEN35_ROTARY_DIM = 64;
constant float QWEN35_ROPE_BASE = 1.0e7f;
constant float QWEN35_RMS_EPS = 1.0e-6f;

struct Qwen35DecodePrepareParams {
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint position_offset;
    float theta;
    uint cache_stride_tokens;
    uint dst_token_index;
    uint threads_per_threadgroup;
};

// Qwen3.5-9B full-attention prefill setup. This fuses:
//   q RMSNorm + q RoPE, k RMSNorm + k RoPE + K cache append, and V cache append.
// The projection and Q/gate split remain separate; attention consumes q in-place
// and K/V from the published cache.
kernel void qwen35_qk_norm_rope_append_many_f32(
    device float *q [[buffer(0)]],
    device const ushort *q_weight [[buffer(1)]],
    device const float *k [[buffer(2)]],
    device const ushort *k_weight [[buffer(3)]],
    device const float *v [[buffer(4)]],
    device float *cache_k [[buffer(5)]],
    device float *cache_v [[buffer(6)]],
    constant uint &start_rope_pos [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    constant uint &start_cache_pos [[buffer(9)]],
    constant uint &cache_stride [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row_group [[threadgroup_position_in_grid]])
{
    if (threads_per_threadgroup != QWEN35_QK_PREPARE_THREADS || tokens == 0) return;

    const uint q_rows = tokens * QWEN35_NUM_Q_HEADS;
    const uint kv_rows = tokens * QWEN35_NUM_KV_HEADS;
    if (row_group >= q_rows + kv_rows) return;

    threadgroup float partial[QWEN35_QK_PREPARE_THREADS];

    const bool is_q = row_group < q_rows;
    const uint logical_row = is_q ? row_group : row_group - q_rows;
    const uint token = is_q ? logical_row / QWEN35_NUM_Q_HEADS : logical_row / QWEN35_NUM_KV_HEADS;
    const uint head = is_q ? logical_row - token * QWEN35_NUM_Q_HEADS : logical_row - token * QWEN35_NUM_KV_HEADS;
    const device float *src = is_q ?
        q + (ulong)logical_row * QWEN35_HEAD_DIM :
        k + (ulong)logical_row * QWEN35_HEAD_DIM;
    const device ushort *weight = is_q ? q_weight : k_weight;

    float local_ss = 0.0f;
    for (uint d = tid; d < QWEN35_HEAD_DIM; d += threads_per_threadgroup) {
        const float x = src[d];
        local_ss += x * x;
    }
    partial[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(QWEN35_HEAD_DIM) + QWEN35_RMS_EPS);
    const uint pos = start_rope_pos + token;
    device float *dst = is_q ?
        q + (ulong)logical_row * QWEN35_HEAD_DIM :
        cache_k + ((ulong)head * cache_stride + (start_cache_pos + token)) * QWEN35_HEAD_DIM;

    const uint half_rotary = QWEN35_ROTARY_DIM / 2;
    for (uint i = tid; i < half_rotary; i += threads_per_threadgroup) {
        const float a = src[i] * scale * bf16_to_float(weight[i]);
        const float b = src[i + half_rotary] * scale * bf16_to_float(weight[i + half_rotary]);
        const float inv_freq = pow(QWEN35_ROPE_BASE, -(2.0f * float(i)) / float(QWEN35_ROTARY_DIM));
        const float th = float(pos) * inv_freq;
        const float c = cos(th);
        const float s = sin(th);
        dst[i] = a * c - b * s;
        dst[i + half_rotary] = b * c + a * s;
    }

    for (uint d = QWEN35_ROTARY_DIM + tid; d < QWEN35_HEAD_DIM; d += threads_per_threadgroup) {
        dst[d] = src[d] * scale * bf16_to_float(weight[d]);
    }

    if (!is_q) {
        const device float *vsrc = v + (ulong)logical_row * QWEN35_HEAD_DIM;
        for (uint d = tid; d < QWEN35_HEAD_DIM; d += threads_per_threadgroup) {
            cache_v[((ulong)head * QWEN35_HEAD_DIM + d) * cache_stride + (start_cache_pos + token)] = vsrc[d];
        }
    }
}

#if defined(__HAVE_BFLOAT__)
// BF16 activation-path version of qwen35_qk_norm_rope_append_many_f32.
// This matches Kestrel's Qwen3.5 q4 prefill setup contract: Q/K are normalized
// and RoPE'd in BF16, K is appended head-major, and V is appended
// value-dim-major without crossing through f32 activation buffers.
kernel void qwen35_qk_norm_rope_append_many_bf16(
    device bfloat *q [[buffer(0)]],
    device const bfloat *q_weight [[buffer(1)]],
    device const bfloat *k [[buffer(2)]],
    device const bfloat *k_weight [[buffer(3)]],
    device const bfloat *v [[buffer(4)]],
    device bfloat *cache_k [[buffer(5)]],
    device bfloat *cache_v [[buffer(6)]],
    constant uint &start_rope_pos [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    constant uint &start_cache_pos [[buffer(9)]],
    constant uint &cache_stride [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row_group [[threadgroup_position_in_grid]])
{
    if (threads_per_threadgroup != QWEN35_QK_PREPARE_THREADS || tokens == 0) return;

    const uint q_rows = tokens * QWEN35_NUM_Q_HEADS;
    const uint kv_rows = tokens * QWEN35_NUM_KV_HEADS;
    if (row_group >= q_rows + kv_rows) return;

    threadgroup float partial[QWEN35_QK_PREPARE_THREADS];

    const bool is_q = row_group < q_rows;
    const uint logical_row = is_q ? row_group : row_group - q_rows;
    const uint token = is_q ? logical_row / QWEN35_NUM_Q_HEADS : logical_row / QWEN35_NUM_KV_HEADS;
    const uint head = is_q ? logical_row - token * QWEN35_NUM_Q_HEADS : logical_row - token * QWEN35_NUM_KV_HEADS;
    const device bfloat *src = is_q ?
        q + (ulong)logical_row * QWEN35_HEAD_DIM :
        k + (ulong)logical_row * QWEN35_HEAD_DIM;
    const device bfloat *weight = is_q ? q_weight : k_weight;

    float local_ss = 0.0f;
    for (uint d = tid; d < QWEN35_HEAD_DIM; d += threads_per_threadgroup) {
        const float x = float(src[d]);
        local_ss += x * x;
    }
    partial[tid] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(QWEN35_HEAD_DIM) + QWEN35_RMS_EPS);
    const uint pos = start_rope_pos + token;
    device bfloat *dst = is_q ?
        q + (ulong)logical_row * QWEN35_HEAD_DIM :
        cache_k + ((ulong)head * cache_stride + (start_cache_pos + token)) * QWEN35_HEAD_DIM;

    const uint half_rotary = QWEN35_ROTARY_DIM / 2;
    for (uint i = tid; i < half_rotary; i += threads_per_threadgroup) {
        const float a = float(src[i]) * scale * float(weight[i]);
        const float b = float(src[i + half_rotary]) * scale * float(weight[i + half_rotary]);
        const float inv_freq = pow(QWEN35_ROPE_BASE, -(2.0f * float(i)) / float(QWEN35_ROTARY_DIM));
        const float th = float(pos) * inv_freq;
        const float c = cos(th);
        const float s = sin(th);
        dst[i] = bfloat(a * c - b * s);
        dst[i + half_rotary] = bfloat(b * c + a * s);
    }

    for (uint d = QWEN35_ROTARY_DIM + tid; d < QWEN35_HEAD_DIM; d += threads_per_threadgroup) {
        dst[d] = bfloat(float(src[d]) * scale * float(weight[d]));
    }

    if (!is_q) {
        const device bfloat *vsrc = v + (ulong)logical_row * QWEN35_HEAD_DIM;
        for (uint d = tid; d < QWEN35_HEAD_DIM; d += threads_per_threadgroup) {
            cache_v[((ulong)head * QWEN35_HEAD_DIM + d) * cache_stride + (start_cache_pos + token)] = vsrc[d];
        }
    }
}
#endif

// Qwen3.5 full-attention decode setup. This ports Kestrel's fused q/gate
// prepare plus head-major K and value-dim-major V append for the one-token
// decode path. The projections remain separate; this replaces split_qg +
// q/k RMSNorm + q/k RoPE + kv_append with one dispatch.
kernel void qwen35_q_gate_prepare_kv_append_pair_head_major_value_dim_major_f32(
    device const float *packed_q [[buffer(0)]],
    device const ushort *q_weight [[buffer(1)]],
    device float *q_output [[buffer(2)]],
    device float *gate [[buffer(3)]],
    device const float *k_input [[buffer(4)]],
    device const float *v_input [[buffer(5)]],
    device const ushort *k_weight [[buffer(6)]],
    device float *key_dst [[buffer(7)]],
    device float *value_dst [[buffer(8)]],
    constant Qwen35DecodePrepareParams &params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    const uint num_heads = params.num_heads;
    const uint num_kv_heads = params.num_kv_heads;
    const uint head_dim = params.head_dim;
    const uint rotary_dim = params.rotary_dim;
    const uint threads_per_threadgroup = params.threads_per_threadgroup;
    const uint total_heads = num_heads + num_kv_heads;
    if (head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > QWEN35_QK_PREPARE_THREADS ||
        params.cache_stride_tokens <= params.dst_token_index ||
        threadgroup_index >= total_heads) return;

    threadgroup float partial[QWEN35_QK_PREPARE_THREADS];

    const bool use_q = threadgroup_index < num_heads;
    const uint head_index = use_q ? threadgroup_index : threadgroup_index - num_heads;
    const ulong row_base = (ulong)head_index * head_dim;
    const ulong packed_base = (ulong)head_index * (2 * head_dim);
    device const ushort *weight = use_q ? q_weight : k_weight;

    float sum = 0.0f;
    for (uint col = tid; col < head_dim; col += threads_per_threadgroup) {
        const float stored = use_q ? packed_q[packed_base + col] : k_input[row_base + col];
        sum += stored * stored;
        if (use_q) {
            gate[row_base + col] = packed_q[packed_base + head_dim + col];
        } else {
            value_dst[(row_base + col) * params.cache_stride_tokens + params.dst_token_index] =
                v_input[row_base + col];
        }
    }

    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / float(head_dim) + QWEN35_RMS_EPS);
    const uint half_rotary_dim = rotary_dim / 2;
    for (uint pair_index = tid; pair_index < half_rotary_dim; pair_index += threads_per_threadgroup) {
        const ulong idx1 = row_base + pair_index;
        const ulong idx2 = row_base + pair_index + half_rotary_dim;
        const float stored1 = use_q ? packed_q[packed_base + pair_index] : k_input[idx1];
        const float stored2 = use_q ?
            packed_q[packed_base + pair_index + half_rotary_dim] :
            k_input[idx2];
        const float x1 = stored1 * inv_rms * bf16_to_float(weight[pair_index]);
        const float x2 = stored2 * inv_rms * bf16_to_float(weight[pair_index + half_rotary_dim]);
        const float inv_freq = pow(params.theta, -(2.0f * float(pair_index)) / float(rotary_dim));
        const float angle = float(params.position_offset) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float rotated1 = x1 * c - x2 * s;
        const float rotated2 = x2 * c + x1 * s;
        if (use_q) {
            q_output[idx1] = rotated1;
            q_output[idx2] = rotated2;
        } else {
            const ulong key_base =
                ((ulong)head_index * params.cache_stride_tokens + params.dst_token_index) * head_dim;
            key_dst[key_base + pair_index] = rotated1;
            key_dst[key_base + pair_index + half_rotary_dim] = rotated2;
        }
    }

    for (uint col = rotary_dim + tid; col < head_dim; col += threads_per_threadgroup) {
        const ulong idx = row_base + col;
        const float stored = use_q ? packed_q[packed_base + col] : k_input[idx];
        const float normalized = stored * inv_rms * bf16_to_float(weight[col]);
        if (use_q) {
            q_output[idx] = normalized;
        } else {
            const ulong key_dst_index =
                ((ulong)head_index * params.cache_stride_tokens + params.dst_token_index) * head_dim + col;
            key_dst[key_dst_index] = normalized;
        }
    }
}

#if defined(__HAVE_BFLOAT__)
// BF16 decode setup sibling for the coherent Qwen3.5 BF16 route. This mirrors
// Kestrel's qwen35_q_gate_prepare_kv_append_pair_head_major_value_dim_major_bf16
// contract: split Q/gate from packed Q, RMSNorm+RoPE Q/K with BF16 rounding, and
// append K head-major plus V value-dim-major.
kernel void qwen35_q_gate_prepare_kv_append_pair_head_major_value_dim_major_bf16(
    device const bfloat *packed_q [[buffer(0)]],
    device const bfloat *q_weight [[buffer(1)]],
    device bfloat *q_output [[buffer(2)]],
    device bfloat *gate [[buffer(3)]],
    device const bfloat *k_input [[buffer(4)]],
    device const bfloat *v_input [[buffer(5)]],
    device const bfloat *k_weight [[buffer(6)]],
    device bfloat *key_dst [[buffer(7)]],
    device bfloat *value_dst [[buffer(8)]],
    constant Qwen35DecodePrepareParams &params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threadgroup_index [[threadgroup_position_in_grid]])
{
    const uint num_heads = params.num_heads;
    const uint num_kv_heads = params.num_kv_heads;
    const uint head_dim = params.head_dim;
    const uint rotary_dim = params.rotary_dim;
    const uint threads_per_threadgroup = params.threads_per_threadgroup;
    const uint total_heads = num_heads + num_kv_heads;
    if (head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > QWEN35_QK_PREPARE_THREADS ||
        params.cache_stride_tokens <= params.dst_token_index ||
        threadgroup_index >= total_heads) return;

    threadgroup float partial[QWEN35_QK_PREPARE_THREADS];

    const bool use_q = threadgroup_index < num_heads;
    const uint head_index = use_q ? threadgroup_index : threadgroup_index - num_heads;
    const ulong row_base = (ulong)head_index * head_dim;
    const ulong packed_base = (ulong)head_index * (2 * head_dim);
    device const bfloat *weight = use_q ? q_weight : k_weight;

    float sum = 0.0f;
    for (uint col = tid; col < head_dim; col += threads_per_threadgroup) {
        const bfloat stored = use_q ? packed_q[packed_base + col] : k_input[row_base + col];
        const float value = float(stored);
        sum += value * value;
        if (use_q) {
            gate[row_base + col] = packed_q[packed_base + head_dim + col];
        } else {
            value_dst[(row_base + col) * params.cache_stride_tokens + params.dst_token_index] =
                v_input[row_base + col];
        }
    }

    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / float(head_dim) + QWEN35_RMS_EPS);
    const uint half_rotary_dim = rotary_dim / 2;
    for (uint pair_index = tid; pair_index < half_rotary_dim; pair_index += threads_per_threadgroup) {
        const ulong idx1 = row_base + pair_index;
        const ulong idx2 = row_base + pair_index + half_rotary_dim;
        const bfloat stored1 = use_q ? packed_q[packed_base + pair_index] : k_input[idx1];
        const bfloat stored2 = use_q ?
            packed_q[packed_base + pair_index + half_rotary_dim] :
            k_input[idx2];
        const float x1 = float(bfloat(float(stored1) * inv_rms * float(weight[pair_index])));
        const float x2 = float(bfloat(float(stored2) * inv_rms * float(weight[pair_index + half_rotary_dim])));
        const float inv_freq = pow(params.theta, -(2.0f * float(pair_index)) / float(rotary_dim));
        const float angle = float(params.position_offset) * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const bfloat rotated1 = bfloat(x1 * c - x2 * s);
        const bfloat rotated2 = bfloat(x2 * c + x1 * s);
        if (use_q) {
            q_output[idx1] = rotated1;
            q_output[idx2] = rotated2;
        } else {
            const ulong key_base = ((ulong)head_index * params.cache_stride_tokens + params.dst_token_index) * head_dim;
            key_dst[key_base + pair_index] = rotated1;
            key_dst[key_base + pair_index + half_rotary_dim] = rotated2;
        }
    }

    for (uint col = rotary_dim + tid; col < head_dim; col += threads_per_threadgroup) {
        const ulong idx = row_base + col;
        const bfloat stored = use_q ? packed_q[packed_base + col] : k_input[idx];
        const bfloat normalized = bfloat(float(stored) * inv_rms * float(weight[col]));
        if (use_q) {
            q_output[idx] = normalized;
        } else {
            const ulong key_dst_index =
                ((ulong)head_index * params.cache_stride_tokens + params.dst_token_index) * head_dim + col;
            key_dst[key_dst_index] = normalized;
        }
    }
}
#endif

// Scaled-dot-product attention for a single decode query (L_q = 1), with GQA.
//   kv_head(h) = h / (num_q_heads / num_kv_heads)        [repeat-interleave]
//   score_t    = scale * dot(q[h], k[kv, t])
//   out[h]     = sum_t softmax(score)_t * v[kv, t]
// q/out: f32 [num_q_heads, head_dim]; k: f32 [num_kv_heads, kv_stride, head_dim],
// v: f32 [num_kv_heads, head_dim, kv_stride]
// where kv_stride is the allocated capacity (>= seq_len) and only the first
// seq_len positions per head are valid. Separating stride from length lets the
// caller over-allocate the cache once and grow seq_len in place across decode
// steps. All cached positions are valid (decode is the last position) so no
// masking. One thread per query head — correctness-first. TODO(perf): parallelize
// over seq_len within a head (flash-style, threadgroup-reduced) once stable.
kernel void sdpa_decode_f32(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &num_q_heads [[buffer(4)]],
    constant uint &num_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &seq_len [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &kv_stride [[buffer(9)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= num_q_heads) return;
    // Caller contract: num_q_heads is a multiple of num_kv_heads. Guard so a bad
    // pair can't silently read out-of-bounds KV (Metal has no bounds checks).
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) return;
    const uint group = num_q_heads / num_kv_heads;
    const uint kvh = h / group;
    const device float *qh = q + (ulong)h * head_dim;
    const device float *kbase = k + (ulong)kvh * kv_stride * head_dim;
    const device float *vbase = v + (ulong)kvh * head_dim * kv_stride;
    device float *oh = out + (ulong)h * head_dim;

    // No cached positions yet -> zero output (avoids 1/0 -> NaN).
    if (seq_len == 0) {
        for (uint d = 0; d < head_dim; d++) oh[d] = 0.0f;
        return;
    }

    // pass 1: max of the scaled scores (for a stable softmax)
    float m = -INFINITY;
    for (uint t = 0; t < seq_len; t++) {
        const device float *kt = kbase + (ulong)t * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
        m = max(m, dot * scale);
    }

    // pass 2: exp-sum + value accumulation
    for (uint d = 0; d < head_dim; d++) oh[d] = 0.0f;
    float denom = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        const device float *kt = kbase + (ulong)t * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
        const float w = exp(dot * scale - m);
        denom += w;
        for (uint d = 0; d < head_dim; d++) oh[d] += w * vbase[(ulong)d * kv_stride + t];
    }
    const float inv = 1.0f / denom;
    for (uint d = 0; d < head_dim; d++) oh[d] *= inv;
}

// Decode attention over a cached prompt prefix plus a per-request local suffix.
// Prefix/local K layout is [num_kv, stride, head_dim]. Prefix/local V layout is
// [num_kv, head_dim, stride].
// Tokens [0, prefix_len) are read from prefix_{k,v}; tokens
// [prefix_len, seq_len) are read from local_{k,v} at t - prefix_len.
kernel void sdpa_decode_prefix_f32(
    device const float *q [[buffer(0)]],
    device const float *prefix_k [[buffer(1)]],
    device const float *prefix_v [[buffer(2)]],
    device const float *local_k [[buffer(3)]],
    device const float *local_v [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &num_q_heads [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &seq_len [[buffer(9)]],
    constant uint &prefix_len [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant uint &prefix_stride [[buffer(12)]],
    constant uint &local_stride [[buffer(13)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= num_q_heads) return;
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) return;
    const uint group = num_q_heads / num_kv_heads;
    const uint kvh = h / group;
    const device float *qh = q + (ulong)h * head_dim;
    const device float *prefix_kbase = prefix_k + (ulong)kvh * prefix_stride * head_dim;
    const device float *prefix_vbase = prefix_v + (ulong)kvh * head_dim * prefix_stride;
    const device float *local_kbase = local_k + (ulong)kvh * local_stride * head_dim;
    const device float *local_vbase = local_v + (ulong)kvh * head_dim * local_stride;
    device float *oh = out + (ulong)h * head_dim;

    if (seq_len == 0) {
        for (uint d = 0; d < head_dim; d++) oh[d] = 0.0f;
        return;
    }

    float m = -INFINITY;
    for (uint t = 0; t < seq_len; t++) {
        const bool use_prefix = t < prefix_len;
        const uint local_t = use_prefix ? 0 : t - prefix_len;
        const device float *kt = use_prefix ?
            prefix_kbase + (ulong)t * head_dim :
            local_kbase + (ulong)local_t * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
        m = max(m, dot * scale);
    }

    for (uint d = 0; d < head_dim; d++) oh[d] = 0.0f;
    float denom = 0.0f;
    for (uint t = 0; t < seq_len; t++) {
        const bool use_prefix = t < prefix_len;
        const uint local_t = use_prefix ? 0 : t - prefix_len;
        const device float *kt = use_prefix ?
            prefix_kbase + (ulong)t * head_dim :
            local_kbase + (ulong)local_t * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
        const float w = exp(dot * scale - m);
        denom += w;
        for (uint d = 0; d < head_dim; d++) {
            const float vv = use_prefix ?
                prefix_vbase[(ulong)d * prefix_stride + t] :
                local_vbase[(ulong)d * local_stride + local_t];
            oh[d] += w * vv;
        }
    }
    const float inv = 1.0f / denom;
    for (uint d = 0; d < head_dim; d++) oh[d] *= inv;
}

// Causal prefill attention for a chunk of query rows. One threadgroup owns one
// (token, q_head) row and computes a tiled online softmax. q/out are
// [tokens, num_q_heads, head_dim]. local_k is [num_kv_heads, stride, head_dim];
// local_v is [num_kv_heads, head_dim, stride].
kernel void sdpa_prefill_f32(
    device const float *q [[buffer(0)]],
    device const float *local_k [[buffer(1)]],
    device const float *local_v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &num_q_heads [[buffer(4)]],
    constant uint &num_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    constant uint &start_local_pos [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant uint &local_stride [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (num_q_heads == 0 || group_index >= tokens * num_q_heads) return;
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) return;
    if (threads_per_threadgroup != SDPA_PREFILL_THREADS) return;
    if (head_dim > QWEN35_HEAD_DIM) return;

    threadgroup float q_cached[QWEN35_HEAD_DIM];
    threadgroup float scores[SDPA_PREFILL_TILE_SEQ];
    threadgroup float partial[SDPA_PREFILL_THREADS];

    const uint row = group_index / num_q_heads;
    const uint h = group_index - row * num_q_heads;
    const uint group = num_q_heads / num_kv_heads;
    const uint kvh = h / group;
    const uint local_len = start_local_pos + row + 1;
    const device float *qh = q + ((ulong)row * num_q_heads + h) * head_dim;
    const device float *kbase = local_k + (ulong)kvh * local_stride * head_dim;
    const device float *vbase = local_v + (ulong)kvh * head_dim * local_stride;
    device float *oh = out + ((ulong)row * num_q_heads + h) * head_dim;

    for (uint d = tid; d < head_dim; d += threads_per_threadgroup) {
        q_cached[d] = qh[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output_acc = 0.0f;
    const bool writes_output_dim = tid < head_dim;

    for (uint tile_start = 0; tile_start < local_len; tile_start += SDPA_PREFILL_TILE_SEQ) {
        const uint tile_len = min(SDPA_PREFILL_TILE_SEQ, local_len - tile_start);

        float local_max = -INFINITY;
        for (uint tile_offset = tid; tile_offset < tile_len; tile_offset += threads_per_threadgroup) {
            const device float *kt = kbase + (ulong)(tile_start + tile_offset) * head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) dot += q_cached[d] * kt[d];
            const float score = dot * scale;
            scores[tile_offset] = score;
            local_max = max(local_max, score);
        }

        partial[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_max = partial[0];

        float local_sum = 0.0f;
        for (uint tile_offset = tid; tile_offset < tile_len; tile_offset += threads_per_threadgroup) {
            const float score_exp = exp(scores[tile_offset] - tile_max);
            scores[tile_offset] = score_exp;
            local_sum += score_exp;
        }

        partial[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_sum = partial[0];

        const float next_max = max(running_max, tile_max);
        const float previous_scale = running_sum == 0.0f ? 0.0f : exp(running_max - next_max);
        const float tile_scale = exp(tile_max - next_max);
        const float next_sum = running_sum * previous_scale + tile_sum * tile_scale;

        if (writes_output_dim) {
            float tile_acc = 0.0f;
            for (uint tile_offset = 0; tile_offset < tile_len; tile_offset += 1) {
                tile_acc += scores[tile_offset] * vbase[(ulong)tid * local_stride + (tile_start + tile_offset)];
            }
            output_acc = (output_acc * running_sum * previous_scale + tile_acc * tile_scale) / next_sum;
        }

        running_max = next_max;
        running_sum = next_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (writes_output_dim) oh[tid] = output_acc;
}

METAL_FUNC uint prefill_tile_max_token_index(uint row_base, uint row_count, uint token_count) {
    if (row_count == 0 || token_count == 0) return 0;
    const uint last_row = row_base + row_count - 1;
    if (row_base / token_count == last_row / token_count) {
        return last_row % token_count;
    }
    return token_count - 1;
}

METAL_FUNC void prefill_store_scaled_masked_score(
    device float *out,
    uint index,
    float value,
    uint row,
    uint col,
    uint token_count,
    uint cache_prefix_length,
    float scale)
{
    const uint token_index = row % token_count;
    const uint max_visible_col = cache_prefix_length + token_index;
    out[index] = (col > max_visible_col) ? -INFINITY : value * scale;
}

METAL_FUNC void prefill_store_scaled_masked_score_for_batch(
    device float *out,
    uint index,
    float value,
    uint row,
    uint col,
    uint token_count,
    uint cache_length,
    float scale)
{
    const uint token_index = row % token_count;
    const uint cache_prefix_before_batch = cache_length - token_count;
    const uint max_visible_col = cache_prefix_before_batch + token_index;
    out[index] = (col > max_visible_col) ? -INFINITY : value * scale;
}

#if defined(__HAVE_BFLOAT__)
METAL_FUNC void prefill_store_scaled_masked_score_bf16(
    device bfloat *out,
    uint index,
    float value,
    uint row,
    uint col,
    uint token_count,
    uint cache_prefix_length,
    float scale)
{
    const uint token_index = row % token_count;
    const uint max_visible_col = cache_prefix_length + token_index;
    out[index] = (col > max_visible_col) ? bfloat(-INFINITY) : bfloat(value * scale);
}

METAL_FUNC void prefill_store_scaled_masked_score_for_batch_bf16(
    device bfloat *out,
    uint index,
    float value,
    uint row,
    uint col,
    uint token_count,
    uint cache_length,
    float scale)
{
    const uint token_index = row % token_count;
    const uint max_visible_col = (cache_length - token_count) + token_index;
    out[index] = (col > max_visible_col) ? bfloat(-INFINITY) : bfloat(value * scale);
}
#endif

// Materialized score stage for Qwen3.5 full-attention prefill. Rows are grouped
// by KV head as [group_q_head, token], matching Kestrel's production prefill
// chain and enabling a single score/probability scratch matrix per KV group.
[[kernel, max_total_threads_per_threadgroup(PREFILL_SCORE_THREADS)]]
kernel void prefill_score_head_major_f32(
    device const float *q [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &q_head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_start [[buffer(9)]],
    constant uint &kv_head_index [[buffer(10)]],
    constant uint &cache_prefix_length [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    constant uint &cache_stride_tokens [[buffer(13)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n == 0 || k != QWEN35_HEAD_DIM || token_count == 0 ||
        q_head_count == 0 || kv_head_count == 0 ||
        kv_head_index >= kv_head_count || cache_stride_tokens < n) return;

    const uint tiles_n = (n + PREFILL_SCORE_TILE_N - 1) / PREFILL_SCORE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_SCORE_TILE_M;
    const uint col_base = tile_col * PREFILL_SCORE_TILE_N;
    if (row_base >= m) return;

    const uint tile_row_count = min((uint)PREFILL_SCORE_TILE_M, m - row_base);
    const uint tile_max_visible_col = cache_prefix_length +
        prefill_tile_max_token_index(row_base, tile_row_count, token_count);
    const bool tile_fully_masked = col_base > tile_max_visible_col;

    threadgroup float lhs_tile[PREFILL_SCORE_TILE_M * PREFILL_SCORE_LHS_LD];
    threadgroup float rhs_tile[PREFILL_SCORE_TILE_K * PREFILL_SCORE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_SCORE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_SCORE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_SCORE_TM * PREFILL_SCORE_TN];

    for (uint i = 0; i < PREFILL_SCORE_TM * PREFILL_SCORE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_SCORE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_SCORE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_SCORE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_SCORE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_SCORE_TILE_M * PREFILL_SCORE_TILE_K) / PREFILL_SCORE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_SCORE_TILE_K * PREFILL_SCORE_TILE_N) / PREFILL_SCORE_THREADS;

    if (!tile_fully_masked) for (uint k_base = 0; k_base < k; k_base += PREFILL_SCORE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_SCORE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_SCORE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            const uint group_head = global_lhs_row / token_count;
            const uint q_token = global_lhs_row - group_head * token_count;
            const uint q_head = head_start + group_head;
            lhs_tile[lhs_row * PREFILL_SCORE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k && q_head < q_head_count) ?
                    q[(q_token * q_head_count + q_head) * k + global_lhs_col] : 0.0f;
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_SCORE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_SCORE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            rhs_tile[rhs_row * PREFILL_SCORE_RHS_LD + rhs_col] =
                (global_rhs_row < k && global_rhs_col < n) ?
                    key_cache[(kv_head_index * cache_stride_tokens + global_rhs_col) * k + global_rhs_row] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_SCORE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    rhs_block[j * PREFILL_SCORE_TN_STRIDE + 0];
                rhs_simd[j].thread_elements()[1] =
                    rhs_block[j * PREFILL_SCORE_TN_STRIDE + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_SCORE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_SCORE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_SCORE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_SCORE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_SCORE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_SCORE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
        if (i * PREFILL_SCORE_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_SCORE_TN + j].thread_elements();
            const uint row = row_offset + i * PREFILL_SCORE_TM_STRIDE;
            const uint col = col_offset + j * PREFILL_SCORE_TN_STRIDE;
            const uint output_index = row * n + col;
            if (j * PREFILL_SCORE_TN_STRIDE < dst_tile_dims.x) {
                prefill_store_scaled_masked_score(out, output_index, accum[0],
                    row, col, token_count, cache_prefix_length, scale);
            }
            if (j * PREFILL_SCORE_TN_STRIDE + 1 < dst_tile_dims.x) {
                prefill_store_scaled_masked_score(out, output_index + 1, accum[1],
                    row, col + 1, token_count, cache_prefix_length, scale);
            }
        }
    }
}

#if defined(__HAVE_BFLOAT__)
// BF16 score stage for the coherent Qwen3.5 full-attention route. Scores are
// materialized as BF16 probabilities, with float accumulation internally.
[[kernel, max_total_threads_per_threadgroup(PREFILL_SCORE_THREADS)]]
kernel void prefill_score_head_major_bf16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *key_cache [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &q_head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_start [[buffer(9)]],
    constant uint &kv_head_index [[buffer(10)]],
    constant uint &cache_prefix_length [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    constant uint &cache_stride_tokens [[buffer(13)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n == 0 || k != QWEN35_HEAD_DIM || token_count == 0 ||
        q_head_count == 0 || kv_head_count == 0 ||
        kv_head_index >= kv_head_count || cache_stride_tokens < n) return;

    const uint tiles_n = (n + PREFILL_SCORE_TILE_N - 1) / PREFILL_SCORE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_SCORE_TILE_M;
    const uint col_base = tile_col * PREFILL_SCORE_TILE_N;
    if (row_base >= m) return;

    const uint tile_row_count = min((uint)PREFILL_SCORE_TILE_M, m - row_base);
    const uint tile_max_visible_col = cache_prefix_length +
        prefill_tile_max_token_index(row_base, tile_row_count, token_count);
    const bool tile_fully_masked = col_base > tile_max_visible_col;

    threadgroup bfloat lhs_tile[PREFILL_SCORE_TILE_M * PREFILL_SCORE_LHS_LD];
    threadgroup bfloat rhs_tile[PREFILL_SCORE_TILE_K * PREFILL_SCORE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_SCORE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_SCORE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_SCORE_TM * PREFILL_SCORE_TN];

    for (uint i = 0; i < PREFILL_SCORE_TM * PREFILL_SCORE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_SCORE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_SCORE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_SCORE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_SCORE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_SCORE_TILE_M * PREFILL_SCORE_TILE_K) / PREFILL_SCORE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_SCORE_TILE_K * PREFILL_SCORE_TILE_N) / PREFILL_SCORE_THREADS;

    if (!tile_fully_masked) for (uint k_base = 0; k_base < k; k_base += PREFILL_SCORE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_SCORE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_SCORE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            const uint group_head = global_lhs_row / token_count;
            const uint q_token = global_lhs_row - group_head * token_count;
            const uint q_head = head_start + group_head;
            lhs_tile[lhs_row * PREFILL_SCORE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k && q_head < q_head_count) ?
                    q[(q_token * q_head_count + q_head) * k + global_lhs_col] : bfloat(0.0f);
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_SCORE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_SCORE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            rhs_tile[rhs_row * PREFILL_SCORE_RHS_LD + rhs_col] =
                (global_rhs_row < k && global_rhs_col < n) ?
                    key_cache[(kv_head_index * cache_stride_tokens + global_rhs_col) * k + global_rhs_row] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup bfloat *lhs_block = lhs_tile + lhs_offset;
        const threadgroup bfloat *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_SCORE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    float(lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 0]);
                lhs_simd[i].thread_elements()[1] =
                    float(lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    float(rhs_block[j * PREFILL_SCORE_TN_STRIDE + 0]);
                rhs_simd[j].thread_elements()[1] =
                    float(rhs_block[j * PREFILL_SCORE_TN_STRIDE + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_SCORE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_SCORE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_SCORE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_SCORE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_SCORE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_SCORE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
        if (i * PREFILL_SCORE_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_SCORE_TN + j].thread_elements();
            const uint row = row_offset + i * PREFILL_SCORE_TM_STRIDE;
            const uint col = col_offset + j * PREFILL_SCORE_TN_STRIDE;
            const uint output_index = row * n + col;
            if (j * PREFILL_SCORE_TN_STRIDE < dst_tile_dims.x) {
                prefill_store_scaled_masked_score_bf16(out, output_index, accum[0],
                    row, col, token_count, cache_prefix_length, scale);
            }
            if (j * PREFILL_SCORE_TN_STRIDE + 1 < dst_tile_dims.x) {
                prefill_store_scaled_masked_score_bf16(out, output_index + 1, accum[1],
                    row, col + 1, token_count, cache_prefix_length, scale);
            }
        }
    }
}
#endif

// Prefix-aware score stage for prompt suffix prefill. Columns [0, prefix_len)
// read the cached static prefix; columns [prefix_len, n) read the request-local
// suffix cache. The causal mask is based on the full cache length before this
// batch, so a query row can see all prefix tokens, previous local tokens, and
// earlier tokens in the current batch.
[[kernel, max_total_threads_per_threadgroup(PREFILL_SCORE_THREADS)]]
kernel void prefill_score_prefix_head_major_f32(
    device const float *q [[buffer(0)]],
    device const float *prefix_key_cache [[buffer(1)]],
    device const float *local_key_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &m [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &token_count [[buffer(7)]],
    constant uint &q_head_count [[buffer(8)]],
    constant uint &kv_head_count [[buffer(9)]],
    constant uint &head_start [[buffer(10)]],
    constant uint &kv_head_index [[buffer(11)]],
    constant uint &prefix_len [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    constant uint &prefix_stride_tokens [[buffer(14)]],
    constant uint &local_stride_tokens [[buffer(15)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n == 0 || k != QWEN35_HEAD_DIM || token_count == 0 ||
        q_head_count == 0 || kv_head_count == 0 ||
        kv_head_index >= kv_head_count || prefix_len > n ||
        prefix_stride_tokens < prefix_len ||
        local_stride_tokens < n - prefix_len) return;

    const uint tiles_n = (n + PREFILL_SCORE_TILE_N - 1) / PREFILL_SCORE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_SCORE_TILE_M;
    const uint col_base = tile_col * PREFILL_SCORE_TILE_N;
    if (row_base >= m) return;

    const uint tile_row_count = min((uint)PREFILL_SCORE_TILE_M, m - row_base);
    const uint tile_max_visible_col = (n - token_count) +
        prefill_tile_max_token_index(row_base, tile_row_count, token_count);
    const bool tile_fully_masked = col_base > tile_max_visible_col;

    threadgroup float lhs_tile[PREFILL_SCORE_TILE_M * PREFILL_SCORE_LHS_LD];
    threadgroup float rhs_tile[PREFILL_SCORE_TILE_K * PREFILL_SCORE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_SCORE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_SCORE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_SCORE_TM * PREFILL_SCORE_TN];

    for (uint i = 0; i < PREFILL_SCORE_TM * PREFILL_SCORE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_SCORE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_SCORE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_SCORE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_SCORE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_SCORE_TILE_M * PREFILL_SCORE_TILE_K) / PREFILL_SCORE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_SCORE_TILE_K * PREFILL_SCORE_TILE_N) / PREFILL_SCORE_THREADS;

    if (!tile_fully_masked) for (uint k_base = 0; k_base < k; k_base += PREFILL_SCORE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_SCORE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_SCORE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            const uint group_head = global_lhs_row / token_count;
            const uint q_token = global_lhs_row - group_head * token_count;
            const uint q_head = head_start + group_head;
            lhs_tile[lhs_row * PREFILL_SCORE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k && q_head < q_head_count) ?
                    q[(q_token * q_head_count + q_head) * k + global_lhs_col] : 0.0f;
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_SCORE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_SCORE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            float value = 0.0f;
            if (global_rhs_row < k && global_rhs_col < n) {
                if (global_rhs_col < prefix_len) {
                    value = prefix_key_cache[(kv_head_index * prefix_stride_tokens + global_rhs_col) * k + global_rhs_row];
                } else {
                    const uint local_col = global_rhs_col - prefix_len;
                    value = local_key_cache[(kv_head_index * local_stride_tokens + local_col) * k + global_rhs_row];
                }
            }
            rhs_tile[rhs_row * PREFILL_SCORE_RHS_LD + rhs_col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_SCORE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    rhs_block[j * PREFILL_SCORE_TN_STRIDE + 0];
                rhs_simd[j].thread_elements()[1] =
                    rhs_block[j * PREFILL_SCORE_TN_STRIDE + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_SCORE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_SCORE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_SCORE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_SCORE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_SCORE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_SCORE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
        if (i * PREFILL_SCORE_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_SCORE_TN + j].thread_elements();
            const uint row = row_offset + i * PREFILL_SCORE_TM_STRIDE;
            const uint col = col_offset + j * PREFILL_SCORE_TN_STRIDE;
            const uint output_index = row * n + col;
            if (j * PREFILL_SCORE_TN_STRIDE < dst_tile_dims.x) {
                prefill_store_scaled_masked_score_for_batch(out, output_index, accum[0],
                    row, col, token_count, n, scale);
            }
            if (j * PREFILL_SCORE_TN_STRIDE + 1 < dst_tile_dims.x) {
                prefill_store_scaled_masked_score_for_batch(out, output_index + 1, accum[1],
                    row, col + 1, token_count, n, scale);
            }
        }
    }
}

#if defined(__HAVE_BFLOAT__)
[[kernel, max_total_threads_per_threadgroup(PREFILL_SCORE_THREADS)]]
kernel void prefill_score_prefix_head_major_bf16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *prefix_key_cache [[buffer(1)]],
    device const bfloat *local_key_cache [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &m [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &token_count [[buffer(7)]],
    constant uint &q_head_count [[buffer(8)]],
    constant uint &kv_head_count [[buffer(9)]],
    constant uint &head_start [[buffer(10)]],
    constant uint &kv_head_index [[buffer(11)]],
    constant uint &prefix_len [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    constant uint &prefix_stride_tokens [[buffer(14)]],
    constant uint &local_stride_tokens [[buffer(15)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n == 0 || n < token_count || k != QWEN35_HEAD_DIM || token_count == 0 ||
        q_head_count == 0 || kv_head_count == 0 ||
        kv_head_index >= kv_head_count || prefix_len > n ||
        prefix_stride_tokens < prefix_len ||
        local_stride_tokens < n - prefix_len) return;

    const uint tiles_n = (n + PREFILL_SCORE_TILE_N - 1) / PREFILL_SCORE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_SCORE_TILE_M;
    const uint col_base = tile_col * PREFILL_SCORE_TILE_N;
    if (row_base >= m) return;

    const uint tile_row_count = min((uint)PREFILL_SCORE_TILE_M, m - row_base);
    const uint tile_max_visible_col = (n - token_count) +
        prefill_tile_max_token_index(row_base, tile_row_count, token_count);
    const bool tile_fully_masked = col_base > tile_max_visible_col;

    threadgroup bfloat lhs_tile[PREFILL_SCORE_TILE_M * PREFILL_SCORE_LHS_LD];
    threadgroup bfloat rhs_tile[PREFILL_SCORE_TILE_K * PREFILL_SCORE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_SCORE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_SCORE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_SCORE_TM * PREFILL_SCORE_TN];

    for (uint i = 0; i < PREFILL_SCORE_TM * PREFILL_SCORE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_SCORE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_SCORE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_SCORE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_SCORE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_SCORE_TILE_M * PREFILL_SCORE_TILE_K) / PREFILL_SCORE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_SCORE_TILE_K * PREFILL_SCORE_TILE_N) / PREFILL_SCORE_THREADS;

    if (!tile_fully_masked) for (uint k_base = 0; k_base < k; k_base += PREFILL_SCORE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_SCORE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_SCORE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            const uint group_head = global_lhs_row / token_count;
            const uint q_token = global_lhs_row - group_head * token_count;
            const uint q_head = head_start + group_head;
            lhs_tile[lhs_row * PREFILL_SCORE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k && q_head < q_head_count) ?
                    q[(q_token * q_head_count + q_head) * k + global_lhs_col] : bfloat(0.0f);
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_SCORE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_SCORE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            bfloat value = bfloat(0.0f);
            if (global_rhs_row < k && global_rhs_col < n) {
                if (global_rhs_col < prefix_len) {
                    value = prefix_key_cache[(kv_head_index * prefix_stride_tokens + global_rhs_col) * k + global_rhs_row];
                } else {
                    const uint local_col = global_rhs_col - prefix_len;
                    value = local_key_cache[(kv_head_index * local_stride_tokens + local_col) * k + global_rhs_row];
                }
            }
            rhs_tile[rhs_row * PREFILL_SCORE_RHS_LD + rhs_col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup bfloat *lhs_block = lhs_tile + lhs_offset;
        const threadgroup bfloat *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_SCORE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    float(lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 0]);
                lhs_simd[i].thread_elements()[1] =
                    float(lhs_block[i * PREFILL_SCORE_TM_STRIDE * PREFILL_SCORE_LHS_LD + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    float(rhs_block[j * PREFILL_SCORE_TN_STRIDE + 0]);
                rhs_simd[j].thread_elements()[1] =
                    float(rhs_block[j * PREFILL_SCORE_TN_STRIDE + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
                for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_SCORE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_SCORE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_SCORE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_SCORE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_SCORE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_SCORE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_SCORE_TM; ++i) {
        if (i * PREFILL_SCORE_TM_STRIDE >= dst_tile_dims.y) continue;
        for (uint j = 0; j < PREFILL_SCORE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_SCORE_TN + j].thread_elements();
            const uint row = row_offset + i * PREFILL_SCORE_TM_STRIDE;
            const uint col = col_offset + j * PREFILL_SCORE_TN_STRIDE;
            const uint output_index = row * n + col;
            if (j * PREFILL_SCORE_TN_STRIDE < dst_tile_dims.x) {
                prefill_store_scaled_masked_score_for_batch_bf16(out, output_index, accum[0],
                    row, col, token_count, n, scale);
            }
            if (j * PREFILL_SCORE_TN_STRIDE + 1 < dst_tile_dims.x) {
                prefill_store_scaled_masked_score_for_batch_bf16(out, output_index + 1, accum[1],
                    row, col + 1, token_count, n, scale);
            }
        }
    }
}
#endif

[[kernel, max_total_threads_per_threadgroup(PREFILL_SOFTMAX_THREADS)]]
kernel void prefill_softmax_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &row_count [[buffer(2)]],
    constant uint &col_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= row_count || col_count == 0 ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > PREFILL_SOFTMAX_THREADS) return;

    threadgroup float scratch_max[PREFILL_SOFTMAX_THREADS];
    threadgroup float scratch_sum[PREFILL_SOFTMAX_THREADS];
    const uint row_offset = row * col_count;
    const float log2e = M_LOG2E_F;

    float local_max = -INFINITY;
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        local_max = max(local_max, input[row_offset + col]);
    }
    scratch_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch_max[tid] = max(scratch_max[tid], scratch_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float row_max = scratch_max[0];
    float local_sum = 0.0f;
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        local_sum += fast::exp2((input[row_offset + col] - row_max) * log2e);
    }
    scratch_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch_sum[tid] += scratch_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float denom = scratch_sum[0] == 0.0f ? 1.0f : scratch_sum[0];
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        output[row_offset + col] = fast::exp2((input[row_offset + col] - row_max) * log2e) / denom;
    }
}

#if defined(__HAVE_BFLOAT__)
[[kernel, max_total_threads_per_threadgroup(PREFILL_SOFTMAX_THREADS)]]
kernel void prefill_softmax_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &row_count [[buffer(2)]],
    constant uint &col_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= row_count || col_count == 0 ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > PREFILL_SOFTMAX_THREADS) return;

    threadgroup float scratch_max[PREFILL_SOFTMAX_THREADS];
    threadgroup float scratch_sum[PREFILL_SOFTMAX_THREADS];
    const uint row_offset = row * col_count;
    const float log2e = M_LOG2E_F;

    float local_max = -INFINITY;
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        local_max = max(local_max, float(input[row_offset + col]));
    }
    scratch_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch_max[tid] = max(scratch_max[tid], scratch_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float row_max = scratch_max[0];
    float local_sum = 0.0f;
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        local_sum += fast::exp2((float(input[row_offset + col]) - row_max) * log2e);
    }
    scratch_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch_sum[tid] += scratch_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float denom = scratch_sum[0] == 0.0f ? 1.0f : scratch_sum[0];
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        output[row_offset + col] = bfloat(fast::exp2((float(input[row_offset + col]) - row_max) * log2e) / denom);
    }
}

[[kernel, max_total_threads_per_threadgroup(PREFILL_SOFTMAX_THREADS)]]
kernel void prefill_softmax_scaled_masked_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &row_count [[buffer(2)]],
    constant uint &col_count [[buffer(3)]],
    constant uint &token_count [[buffer(4)]],
    constant uint &cache_prefix_length [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= row_count || col_count == 0 || token_count == 0 ||
        threads_per_threadgroup == 0 ||
        threads_per_threadgroup > PREFILL_SOFTMAX_THREADS) return;

    threadgroup float scratch_max[PREFILL_SOFTMAX_THREADS];
    threadgroup float scratch_sum[PREFILL_SOFTMAX_THREADS];
    const uint row_offset = row * col_count;
    const uint token_index = row % token_count;
    const uint max_visible_col = cache_prefix_length + token_index;
    const float log2e = M_LOG2E_F;

    float local_max = -INFINITY;
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        const float value = (col > max_visible_col) ? -INFINITY : float(input[row_offset + col]) * scale;
        local_max = max(local_max, value);
    }
    scratch_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch_max[tid] = max(scratch_max[tid], scratch_max[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float row_max = scratch_max[0];
    float local_sum = 0.0f;
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        if (col <= max_visible_col) {
            local_sum += fast::exp2((float(input[row_offset + col]) * scale - row_max) * log2e);
        }
    }
    scratch_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch_sum[tid] += scratch_sum[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float denom = scratch_sum[0] == 0.0f ? 1.0f : scratch_sum[0];
    for (uint col = tid; col < col_count; col += threads_per_threadgroup) {
        output[row_offset + col] = (col > max_visible_col) ?
            bfloat(0.0f) :
            bfloat(fast::exp2((float(input[row_offset + col]) * scale - row_max) * log2e) / denom);
    }
}

#endif

[[kernel, max_total_threads_per_threadgroup(PREFILL_VALUE_THREADS)]]
kernel void prefill_value_head_major_value_dim_major_f32(
    device const float *lhs [[buffer(0)]],
    device const float *value_cache [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &head_count [[buffer(7)]],
    constant uint &kv_head_count [[buffer(8)]],
    constant uint &head_start [[buffer(9)]],
    constant uint &kv_head_index [[buffer(10)]],
    constant uint &cache_stride_tokens [[buffer(11)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n != QWEN35_HEAD_DIM || k == 0 || token_count == 0 ||
        kv_head_count == 0 || kv_head_index >= kv_head_count ||
        cache_stride_tokens < k) return;

    const uint tiles_n = (n + PREFILL_VALUE_TILE_N - 1) / PREFILL_VALUE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_VALUE_TILE_M;
    const uint col_base = tile_col * PREFILL_VALUE_TILE_N;
    if (row_base >= m) return;

    threadgroup float lhs_tile[PREFILL_VALUE_TILE_M * PREFILL_VALUE_LHS_LD];
    threadgroup float rhs_tile[PREFILL_VALUE_TILE_K * PREFILL_VALUE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_VALUE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_VALUE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_VALUE_TM * PREFILL_VALUE_TN];

    for (uint i = 0; i < PREFILL_VALUE_TM * PREFILL_VALUE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_VALUE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_VALUE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_VALUE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_VALUE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_VALUE_TILE_M * PREFILL_VALUE_TILE_K) / PREFILL_VALUE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_VALUE_TILE_K * PREFILL_VALUE_TILE_N) / PREFILL_VALUE_THREADS;

    for (uint k_base = 0; k_base < k; k_base += PREFILL_VALUE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_VALUE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_VALUE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            lhs_tile[lhs_row * PREFILL_VALUE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k) ?
                    lhs[global_lhs_row * k + global_lhs_col] : 0.0f;
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_VALUE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_VALUE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            rhs_tile[rhs_row * PREFILL_VALUE_RHS_LD + rhs_col] =
                (global_rhs_row < k && global_rhs_col < n) ?
                    value_cache[(kv_head_index * n + global_rhs_col) *
                        cache_stride_tokens + global_rhs_row] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_VALUE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    lhs_block[i * PREFILL_VALUE_TM_STRIDE * PREFILL_VALUE_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    lhs_block[i * PREFILL_VALUE_TM_STRIDE * PREFILL_VALUE_LHS_LD + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    rhs_block[j * PREFILL_VALUE_TN_STRIDE + 0];
                rhs_simd[j].thread_elements()[1] =
                    rhs_block[j * PREFILL_VALUE_TN_STRIDE + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
                for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_VALUE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_VALUE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_VALUE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_VALUE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_VALUE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_VALUE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
        if (i * PREFILL_VALUE_TM_STRIDE >= dst_tile_dims.y) continue;
        const uint row = row_offset + i * PREFILL_VALUE_TM_STRIDE;
        const uint row_group_head = row / token_count;
        const uint row_token = row - row_group_head * token_count;
        const uint row_head = head_start + row_group_head;
        if (row_head >= head_count) continue;
        for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_VALUE_TN + j].thread_elements();
            const uint col = col_offset + j * PREFILL_VALUE_TN_STRIDE;
            const uint output_index = (row_token * head_count + row_head) * n + col;
            if (j * PREFILL_VALUE_TN_STRIDE < dst_tile_dims.x) {
                out[output_index] = accum[0];
            }
            if (j * PREFILL_VALUE_TN_STRIDE + 1 < dst_tile_dims.x) {
                out[output_index + 1] = accum[1];
            }
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(PREFILL_VALUE_THREADS)]]
kernel void prefill_value_prefix_head_major_value_dim_major_f32(
    device const float *lhs [[buffer(0)]],
    device const float *prefix_value_cache [[buffer(1)]],
    device const float *local_value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &m [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &token_count [[buffer(7)]],
    constant uint &head_count [[buffer(8)]],
    constant uint &kv_head_count [[buffer(9)]],
    constant uint &head_start [[buffer(10)]],
    constant uint &kv_head_index [[buffer(11)]],
    constant uint &prefix_len [[buffer(12)]],
    constant uint &prefix_stride_tokens [[buffer(13)]],
    constant uint &local_stride_tokens [[buffer(14)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n != QWEN35_HEAD_DIM || k == 0 || token_count == 0 ||
        kv_head_count == 0 || kv_head_index >= kv_head_count ||
        prefix_len > k || prefix_stride_tokens < prefix_len ||
        local_stride_tokens < k - prefix_len) return;

    const uint tiles_n = (n + PREFILL_VALUE_TILE_N - 1) / PREFILL_VALUE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_VALUE_TILE_M;
    const uint col_base = tile_col * PREFILL_VALUE_TILE_N;
    if (row_base >= m) return;

    threadgroup float lhs_tile[PREFILL_VALUE_TILE_M * PREFILL_VALUE_LHS_LD];
    threadgroup float rhs_tile[PREFILL_VALUE_TILE_K * PREFILL_VALUE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_VALUE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_VALUE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_VALUE_TM * PREFILL_VALUE_TN];

    for (uint i = 0; i < PREFILL_VALUE_TM * PREFILL_VALUE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_VALUE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_VALUE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_VALUE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_VALUE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_VALUE_TILE_M * PREFILL_VALUE_TILE_K) / PREFILL_VALUE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_VALUE_TILE_K * PREFILL_VALUE_TILE_N) / PREFILL_VALUE_THREADS;

    for (uint k_base = 0; k_base < k; k_base += PREFILL_VALUE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_VALUE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_VALUE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            lhs_tile[lhs_row * PREFILL_VALUE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k) ?
                    lhs[global_lhs_row * k + global_lhs_col] : 0.0f;
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_VALUE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_VALUE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            float value = 0.0f;
            if (global_rhs_row < k && global_rhs_col < n) {
                if (global_rhs_row < prefix_len) {
                    value = prefix_value_cache[(kv_head_index * n + global_rhs_col) *
                        prefix_stride_tokens + global_rhs_row];
                } else {
                    const uint local_row = global_rhs_row - prefix_len;
                    value = local_value_cache[(kv_head_index * n + global_rhs_col) *
                        local_stride_tokens + local_row];
                }
            }
            rhs_tile[rhs_row * PREFILL_VALUE_RHS_LD + rhs_col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup float *lhs_block = lhs_tile + lhs_offset;
        const threadgroup float *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_VALUE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    lhs_block[i * PREFILL_VALUE_TM_STRIDE * PREFILL_VALUE_LHS_LD + 0];
                lhs_simd[i].thread_elements()[1] =
                    lhs_block[i * PREFILL_VALUE_TM_STRIDE * PREFILL_VALUE_LHS_LD + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    rhs_block[j * PREFILL_VALUE_TN_STRIDE + 0];
                rhs_simd[j].thread_elements()[1] =
                    rhs_block[j * PREFILL_VALUE_TN_STRIDE + 1];
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
                for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_VALUE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_VALUE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_VALUE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_VALUE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_VALUE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_VALUE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
        if (i * PREFILL_VALUE_TM_STRIDE >= dst_tile_dims.y) continue;
        const uint row = row_offset + i * PREFILL_VALUE_TM_STRIDE;
        const uint row_group_head = row / token_count;
        const uint row_token = row - row_group_head * token_count;
        const uint row_head = head_start + row_group_head;
        if (row_head >= head_count) continue;
        for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_VALUE_TN + j].thread_elements();
            const uint col = col_offset + j * PREFILL_VALUE_TN_STRIDE;
            const uint output_index = (row_token * head_count + row_head) * n + col;
            if (j * PREFILL_VALUE_TN_STRIDE < dst_tile_dims.x) {
                out[output_index] = accum[0];
            }
            if (j * PREFILL_VALUE_TN_STRIDE + 1 < dst_tile_dims.x) {
                out[output_index + 1] = accum[1];
            }
        }
    }
}

#if defined(__HAVE_BFLOAT__)
[[kernel, max_total_threads_per_threadgroup(PREFILL_VALUE_THREADS)]]
kernel void prefill_value_prefix_head_major_value_dim_major_bf16(
    device const bfloat *lhs [[buffer(0)]],
    device const bfloat *prefix_value_cache [[buffer(1)]],
    device const bfloat *local_value_cache [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &m [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]],
    constant uint &token_count [[buffer(7)]],
    constant uint &head_count [[buffer(8)]],
    constant uint &kv_head_count [[buffer(9)]],
    constant uint &head_start [[buffer(10)]],
    constant uint &kv_head_index [[buffer(11)]],
    constant uint &prefix_len [[buffer(12)]],
    constant uint &prefix_stride_tokens [[buffer(13)]],
    constant uint &local_stride_tokens [[buffer(14)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (m == 0 || n != QWEN35_HEAD_DIM || k == 0 || token_count == 0 ||
        kv_head_count == 0 || kv_head_index >= kv_head_count ||
        prefix_len > k || prefix_stride_tokens < prefix_len ||
        local_stride_tokens < k - prefix_len) return;

    const uint tiles_n = (n + PREFILL_VALUE_TILE_N - 1) / PREFILL_VALUE_TILE_N;
    const uint tile_row = group_index / tiles_n;
    const uint tile_col = group_index - tile_row * tiles_n;
    const uint row_base = tile_row * PREFILL_VALUE_TILE_M;
    const uint col_base = tile_col * PREFILL_VALUE_TILE_N;
    if (row_base >= m) return;

    threadgroup bfloat lhs_tile[PREFILL_VALUE_TILE_M * PREFILL_VALUE_LHS_LD];
    threadgroup bfloat rhs_tile[PREFILL_VALUE_TILE_K * PREFILL_VALUE_RHS_LD];
    simdgroup_matrix<float, 8, 8> lhs_simd[PREFILL_VALUE_TM];
    simdgroup_matrix<float, 8, 8> rhs_simd[PREFILL_VALUE_TN];
    simdgroup_matrix<float, 8, 8> results[PREFILL_VALUE_TM * PREFILL_VALUE_TN];

    for (uint i = 0; i < PREFILL_VALUE_TM * PREFILL_VALUE_TN; ++i) {
        results[i] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const ushort tm = 8 * (simd_group_id / PREFILL_VALUE_WN);
    const ushort tn = 8 * (simd_group_id % PREFILL_VALUE_WN);
    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort lhs_offset = sn + (tm + sm) * PREFILL_VALUE_LHS_LD;
    const ushort rhs_offset = sm * PREFILL_VALUE_RHS_LD + (tn + sn);
    const uint lhs_loads_per_thread =
        (PREFILL_VALUE_TILE_M * PREFILL_VALUE_TILE_K) / PREFILL_VALUE_THREADS;
    const uint rhs_loads_per_thread =
        (PREFILL_VALUE_TILE_K * PREFILL_VALUE_TILE_N) / PREFILL_VALUE_THREADS;

    for (uint k_base = 0; k_base < k; k_base += PREFILL_VALUE_TILE_K) {
        for (uint load = 0; load < lhs_loads_per_thread; ++load) {
            const uint lhs_index = local_tid * lhs_loads_per_thread + load;
            const uint lhs_row = lhs_index / PREFILL_VALUE_TILE_K;
            const uint lhs_col = lhs_index - lhs_row * PREFILL_VALUE_TILE_K;
            const uint global_lhs_row = row_base + lhs_row;
            const uint global_lhs_col = k_base + lhs_col;
            lhs_tile[lhs_row * PREFILL_VALUE_LHS_LD + lhs_col] =
                (global_lhs_row < m && global_lhs_col < k) ?
                    lhs[global_lhs_row * k + global_lhs_col] : bfloat(0.0f);
        }
        for (uint load = 0; load < rhs_loads_per_thread; ++load) {
            const uint rhs_index = local_tid * rhs_loads_per_thread + load;
            const uint rhs_row = rhs_index / PREFILL_VALUE_TILE_N;
            const uint rhs_col = rhs_index - rhs_row * PREFILL_VALUE_TILE_N;
            const uint global_rhs_row = k_base + rhs_row;
            const uint global_rhs_col = col_base + rhs_col;
            bfloat value = bfloat(0.0f);
            if (global_rhs_row < k && global_rhs_col < n) {
                if (global_rhs_row < prefix_len) {
                    value = prefix_value_cache[(kv_head_index * n + global_rhs_col) *
                        prefix_stride_tokens + global_rhs_row];
                } else {
                    const uint local_row = global_rhs_row - prefix_len;
                    value = local_value_cache[(kv_head_index * n + global_rhs_col) *
                        local_stride_tokens + local_row];
                }
            }
            rhs_tile[rhs_row * PREFILL_VALUE_RHS_LD + rhs_col] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const threadgroup bfloat *lhs_block = lhs_tile + lhs_offset;
        const threadgroup bfloat *rhs_block = rhs_tile + rhs_offset;
        for (uint kk = 0; kk < PREFILL_VALUE_TILE_K; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
                lhs_simd[i].thread_elements()[0] =
                    float(lhs_block[i * PREFILL_VALUE_TM_STRIDE * PREFILL_VALUE_LHS_LD + 0]);
                lhs_simd[i].thread_elements()[1] =
                    float(lhs_block[i * PREFILL_VALUE_TM_STRIDE * PREFILL_VALUE_LHS_LD + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
                rhs_simd[j].thread_elements()[0] =
                    float(rhs_block[j * PREFILL_VALUE_TN_STRIDE + 0]);
                rhs_simd[j].thread_elements()[1] =
                    float(rhs_block[j * PREFILL_VALUE_TN_STRIDE + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
                for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
                    const uint j_serp = (i % 2) ? (PREFILL_VALUE_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * PREFILL_VALUE_TN + j_serp],
                        lhs_simd[i],
                        rhs_simd[j_serp],
                        results[i * PREFILL_VALUE_TN + j_serp]);
                }
            }
            lhs_block += 8;
            rhs_block += 8 * PREFILL_VALUE_RHS_LD;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row_offset = row_base + tm + sm;
    const uint col_offset = col_base + tn + sn;
    if (row_offset >= m || col_offset >= n) return;
    const uint2 dst_tile_dims = uint2(
        min((uint)PREFILL_VALUE_TILE_N, n - col_base) - (tn + sn),
        min((uint)PREFILL_VALUE_TILE_M, m - row_base) - (tm + sm));
    if (dst_tile_dims.x == 0 || dst_tile_dims.y == 0) return;

    for (uint i = 0; i < PREFILL_VALUE_TM; ++i) {
        if (i * PREFILL_VALUE_TM_STRIDE >= dst_tile_dims.y) continue;
        const uint row = row_offset + i * PREFILL_VALUE_TM_STRIDE;
        const uint row_group_head = row / token_count;
        const uint row_token = row - row_group_head * token_count;
        const uint row_head = head_start + row_group_head;
        if (row_head >= head_count) continue;
        for (uint j = 0; j < PREFILL_VALUE_TN; ++j) {
            thread const auto &accum = results[i * PREFILL_VALUE_TN + j].thread_elements();
            const uint col = col_offset + j * PREFILL_VALUE_TN_STRIDE;
            const uint output_index = (row_token * head_count + row_head) * n + col;
            if (j * PREFILL_VALUE_TN_STRIDE < dst_tile_dims.x) {
                out[output_index] = bfloat(accum[0]);
            }
            if (j * PREFILL_VALUE_TN_STRIDE + 1 < dst_tile_dims.x) {
                out[output_index + 1] = bfloat(accum[1]);
            }
        }
    }
}
#endif

kernel void attention_prefill_compact_value_dim_major_head_f32(
    device const float *value_cache [[buffer(0)]],
    device float *compact_value [[buffer(1)]],
    constant uint &cache_length [[buffer(2)]],
    constant uint &cache_stride_tokens [[buffer(3)]],
    constant uint &kv_head_index [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (cache_length == 0 || cache_stride_tokens < cache_length || head_dim == 0) return;
    const uint total = cache_length * head_dim;
    if (tid >= total) return;

    const uint token_index = tid % cache_length;
    const uint dim = tid / cache_length;
    compact_value[dim * cache_length + token_index] =
        value_cache[(kv_head_index * head_dim + dim) * cache_stride_tokens + token_index];
}

kernel void attention_prefill_scatter_group_output_f32(
    device const float *group_output [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &token_count [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_start [[buffer(4)]],
    constant uint &group_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (token_count == 0 || head_count == 0 || group_count == 0 || head_dim == 0) return;
    const uint total = token_count * group_count * head_dim;
    if (tid >= total) return;

    const uint dim = tid % head_dim;
    const uint row = tid / head_dim;
    const uint group_head = row / token_count;
    const uint token_index = row - group_head * token_count;
    const uint head_index = head_start + group_head;
    if (head_index >= head_count) return;

    out[(token_index * head_count + head_index) * head_dim + dim] = group_output[tid];
}

#if defined(__HAVE_BFLOAT__)
kernel void attention_prefill_compact_query_group_bf16(
    device const bfloat *queries [[buffer(0)]],
    device bfloat *compact_query [[buffer(1)]],
    constant uint &token_count [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_start [[buffer(4)]],
    constant uint &group_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (token_count == 0 || head_count == 0 || group_count == 0 || head_dim == 0) return;
    const uint total = token_count * group_count * head_dim;
    if (tid >= total) return;

    const uint dim = tid % head_dim;
    const uint row = tid / head_dim;
    const uint group_head = row / token_count;
    const uint token_index = row - group_head * token_count;
    const uint head_index = head_start + group_head;
    if (head_index >= head_count) return;

    compact_query[tid] = queries[(token_index * head_count + head_index) * head_dim + dim];
}

kernel void attention_prefill_compact_value_dim_major_head_bf16(
    device const bfloat *value_cache [[buffer(0)]],
    device bfloat *compact_value [[buffer(1)]],
    constant uint &cache_length [[buffer(2)]],
    constant uint &cache_stride_tokens [[buffer(3)]],
    constant uint &kv_head_index [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (cache_length == 0 || cache_stride_tokens < cache_length || head_dim == 0) return;
    const uint total = cache_length * head_dim;
    if (tid >= total) return;

    const uint token_index = tid % cache_length;
    const uint dim = tid / cache_length;
    compact_value[dim * cache_length + token_index] =
        value_cache[(kv_head_index * head_dim + dim) * cache_stride_tokens + token_index];
}

kernel void attention_prefill_scatter_group_output_bf16(
    device const bfloat *group_output [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &token_count [[buffer(2)]],
    constant uint &head_count [[buffer(3)]],
    constant uint &head_start [[buffer(4)]],
    constant uint &group_count [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (token_count == 0 || head_count == 0 || group_count == 0 || head_dim == 0) return;
    const uint total = token_count * group_count * head_dim;
    if (tid >= total) return;

    const uint dim = tid % head_dim;
    const uint row = tid / head_dim;
    const uint group_head = row / token_count;
    const uint token_index = row - group_head * token_count;
    const uint head_index = head_start + group_head;
    if (head_index >= head_count) return;

    out[(token_index * head_count + head_index) * head_dim + dim] = group_output[tid];
}
#endif

METAL_FUNC float sdpa_prefill_d256_row_max(float2 values) {
    float reduced = max(values.x, values.y);
    reduced = max(reduced, simd_shuffle_xor(reduced, ushort(1)));
    reduced = max(reduced, simd_shuffle_xor(reduced, ushort(8)));
    return reduced;
}

METAL_FUNC float sdpa_prefill_d256_row_sum(float2 values) {
    float reduced = values.x + values.y;
    reduced += simd_shuffle_xor(reduced, ushort(1));
    reduced += simd_shuffle_xor(reduced, ushort(8));
    return reduced;
}

// Kestrel-style D-split d=256 prefill attention for the non-prefix path. The
// 1-D group index is flattened from (q_head, row_tile) so Peregrine can keep its
// minimal Metal runtime boundary.
[[kernel, max_total_threads_per_threadgroup(SDPA_PREFILL_D256_THREADS)]]
kernel void sdpa_prefill_d256_dsplit_f32(
    device const float *q [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device const float *value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &token_count [[buffer(4)]],
    constant uint &q_head_count [[buffer(5)]],
    constant uint &kv_head_count [[buffer(6)]],
    constant uint &cache_prefix_length [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &cache_stride_tokens [[buffer(9)]],
    ushort simd_lane_id [[thread_index_in_simdgroup]],
    ushort simd_group_id [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (token_count == 0 || q_head_count == 0 || kv_head_count == 0 ||
        q_head_count % kv_head_count != 0 ||
        cache_stride_tokens < cache_prefix_length + token_count ||
        simd_group_id >= 1) return;

    const uint row_tiles = (token_count + SDPA_PREFILL_D256_TILE_M - 1) / SDPA_PREFILL_D256_TILE_M;
    if (row_tiles == 0) return;
    const uint q_head = group_index / row_tiles;
    const uint tile_index = group_index - q_head * row_tiles;
    if (q_head >= q_head_count) return;

    const uint row_base = tile_index * SDPA_PREFILL_D256_TILE_M;
    if (row_base >= token_count) return;

    const uint group = q_head_count / kv_head_count;
    const uint kv_head = q_head / group;
    const uint cache_length = cache_prefix_length + token_count;
    const uint tile_rows = min(SDPA_PREFILL_D256_TILE_M, token_count - row_base);

    const ushort qid = simd_lane_id / 4;
    const ushort sm = (qid & 4) + (simd_lane_id / 2) % 4;
    const ushort sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    const ushort local_row = 8 * simd_group_id + sm;

    threadgroup float q_tile[SDPA_PREFILL_D256_TILE_M * SDPA_PREFILL_D256_Q_LD];
    threadgroup float kv_tile[SDPA_PREFILL_D256_TILE_N * SDPA_PREFILL_D256_DSPLIT_V_LD];

    for (uint idx = local_tid; idx < SDPA_PREFILL_D256_TILE_M *
        SDPA_PREFILL_D256_HEAD_DIM;
        idx += SDPA_PREFILL_D256_THREADS) {
        const uint row = idx / SDPA_PREFILL_D256_HEAD_DIM;
        const uint dim = idx - row * SDPA_PREFILL_D256_HEAD_DIM;
        const uint token = row_base + row;
        q_tile[row * SDPA_PREFILL_D256_Q_LD + dim] =
            (row < tile_rows) ?
                q[(token * q_head_count + q_head) * SDPA_PREFILL_D256_HEAD_DIM + dim] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> q_frag;
    simdgroup_matrix<float, 8, 8> k_frag[4];
    simdgroup_matrix<float, 8, 8> score_frag[4];
    simdgroup_matrix<float, 8, 8> v_frag;
    simdgroup_matrix<float, 8, 8> out_frag[32];

    for (uint dim_tile = 0; dim_tile < 32; ++dim_tile) {
        out_frag[dim_tile] = simdgroup_matrix<float, 8, 8>(0.0f);
    }

    float row_max = -3.402823466e+38F;
    float row_sum = 0.0f;
    const float scale_log2 = scale * M_LOG2E_F;

    for (uint col_base = 0; col_base < cache_length; col_base += SDPA_PREFILL_D256_TILE_N) {
        for (uint i = 0; i < 4; ++i) {
            score_frag[i] = simdgroup_matrix<float, 8, 8>(0.0f);
        }

        for (uint split = 0; split < 2; ++split) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = local_tid; idx < SDPA_PREFILL_D256_TILE_N *
                SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                idx += SDPA_PREFILL_D256_THREADS) {
                const uint key_row = idx / SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                const uint dim_half = idx - key_row * SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                const uint dim = split * SDPA_PREFILL_D256_DSPLIT_HEAD_DIM + dim_half;
                const uint key_token = col_base + key_row;
                kv_tile[dim_half * SDPA_PREFILL_D256_K_LD + key_row] =
                    (key_token < cache_length) ?
                        key_cache[(kv_head * cache_stride_tokens + key_token) *
                            SDPA_PREFILL_D256_HEAD_DIM + dim] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dim_half_base = 0; dim_half_base < SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                dim_half_base += 8) {
                const uint dim_base = split * SDPA_PREFILL_D256_DSPLIT_HEAD_DIM + dim_half_base;
                thread auto &q_elems = q_frag.thread_elements();
                q_elems[0] = q_tile[local_row * SDPA_PREFILL_D256_Q_LD + dim_base + sn];
                q_elems[1] = q_tile[local_row * SDPA_PREFILL_D256_Q_LD + dim_base + sn + 1];
                simdgroup_barrier(mem_flags::mem_none);

                for (uint key_tile = 0; key_tile < 4; ++key_tile) {
                    thread auto &k_elems = k_frag[key_tile].thread_elements();
                    k_elems[0] = kv_tile[(dim_half_base + sm) *
                        SDPA_PREFILL_D256_K_LD + key_tile * 8 + sn];
                    k_elems[1] = kv_tile[(dim_half_base + sm) *
                        SDPA_PREFILL_D256_K_LD + key_tile * 8 + sn + 1];
                }
                simdgroup_barrier(mem_flags::mem_none);

                for (uint key_tile = 0; key_tile < 4; ++key_tile) {
                    simdgroup_multiply_accumulate(
                        score_frag[key_tile],
                        q_frag,
                        k_frag[key_tile],
                        score_frag[key_tile]);
                }
            }
        }

        float next_max = row_max;
        for (uint key_tile = 0; key_tile < 4; ++key_tile) {
            thread auto &score_elems = score_frag[key_tile].thread_elements();
            const uint token0 = col_base + key_tile * 8 + sn;
            const uint query_token = row_base + local_row;
            score_elems[0] *= scale_log2;
            score_elems[1] *= scale_log2;
            if (local_row >= tile_rows || token0 >= cache_length ||
                token0 > cache_prefix_length + query_token) {
                score_elems[0] = -3.402823466e+38F;
            }
            if (local_row >= tile_rows || token0 + 1 >= cache_length ||
                token0 + 1 > cache_prefix_length + query_token) {
                score_elems[1] = -3.402823466e+38F;
            }
            next_max = max(next_max,
                sdpa_prefill_d256_row_max(float2(score_elems[0], score_elems[1])));
        }

        const float prev_factor = fast::exp2(row_max - next_max);
        float tile_sum = 0.0f;
        for (uint key_tile = 0; key_tile < 4; ++key_tile) {
            thread auto &score_elems = score_frag[key_tile].thread_elements();
            score_elems[0] = fast::exp2(score_elems[0] - next_max);
            score_elems[1] = fast::exp2(score_elems[1] - next_max);
            tile_sum += sdpa_prefill_d256_row_sum(float2(score_elems[0], score_elems[1]));
        }

        for (uint dim_tile = 0; dim_tile < 32; ++dim_tile) {
            thread auto &out_elems = out_frag[dim_tile].thread_elements();
            out_elems[0] *= prev_factor;
            out_elems[1] *= prev_factor;
        }

        for (uint split = 0; split < 2; ++split) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = local_tid; idx < SDPA_PREFILL_D256_TILE_N *
                SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                idx += SDPA_PREFILL_D256_THREADS) {
                const uint key_row = idx / SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                const uint dim_half = idx - key_row * SDPA_PREFILL_D256_DSPLIT_HEAD_DIM;
                const uint dim = split * SDPA_PREFILL_D256_DSPLIT_HEAD_DIM + dim_half;
                const uint key_token = col_base + key_row;
                kv_tile[key_row * SDPA_PREFILL_D256_DSPLIT_V_LD + dim_half] =
                    (key_token < cache_length) ?
                        value_cache[(kv_head * SDPA_PREFILL_D256_HEAD_DIM + dim) *
                            cache_stride_tokens + key_token] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint dim_tile_half = 0; dim_tile_half < 16; ++dim_tile_half) {
                const uint dim_base_half = dim_tile_half * 8;
                const uint out_tile = split * 16 + dim_tile_half;
                for (uint key_tile = 0; key_tile < 4; ++key_tile) {
                    thread auto &v_elems = v_frag.thread_elements();
                    v_elems[0] = kv_tile[(key_tile * 8 + sm) *
                        SDPA_PREFILL_D256_DSPLIT_V_LD + dim_base_half + sn];
                    v_elems[1] = kv_tile[(key_tile * 8 + sm) *
                        SDPA_PREFILL_D256_DSPLIT_V_LD + dim_base_half + sn + 1];
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(
                        out_frag[out_tile],
                        score_frag[key_tile],
                        v_frag,
                        out_frag[out_tile]);
                }
            }
        }

        row_max = next_max;
        row_sum = row_sum * prev_factor + tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_sum = row_sum == 0.0f ? 1.0f : 1.0f / row_sum;
    const uint query_token = row_base + local_row;
    if (local_row < tile_rows) {
        for (uint dim_tile = 0; dim_tile < 32; ++dim_tile) {
            thread auto &out_elems = out_frag[dim_tile].thread_elements();
            const uint dim = dim_tile * 8 + sn;
            out[(query_token * q_head_count + q_head) * SDPA_PREFILL_D256_HEAD_DIM + dim] =
                out_elems[0] * inv_sum;
            out[(query_token * q_head_count + q_head) * SDPA_PREFILL_D256_HEAD_DIM + dim + 1] =
                out_elems[1] * inv_sum;
        }
    }
}

// Causal prefill attention over a cached prefix plus the current request-local
// suffix. One threadgroup owns one (token, q_head) row.
kernel void sdpa_prefill_prefix_f32(
    device const float *q [[buffer(0)]],
    device const float *prefix_k [[buffer(1)]],
    device const float *prefix_v [[buffer(2)]],
    device const float *local_k [[buffer(3)]],
    device const float *local_v [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &num_q_heads [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &tokens [[buffer(9)]],
    constant uint &start_local_pos [[buffer(10)]],
    constant uint &prefix_len [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    constant uint &prefix_stride [[buffer(13)]],
    constant uint &local_stride [[buffer(14)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]])
{
    if (num_q_heads == 0 || group_index >= tokens * num_q_heads) return;
    if (num_kv_heads == 0 || num_q_heads % num_kv_heads != 0) return;
    if (threads_per_threadgroup != SDPA_PREFILL_THREADS) return;
    if (head_dim > QWEN35_HEAD_DIM) return;

    threadgroup float q_cached[QWEN35_HEAD_DIM];
    threadgroup float scores[SDPA_PREFILL_TILE_SEQ];
    threadgroup float partial[SDPA_PREFILL_THREADS];

    const uint row = group_index / num_q_heads;
    const uint h = group_index - row * num_q_heads;
    const uint group = num_q_heads / num_kv_heads;
    const uint kvh = h / group;
    const uint local_len = start_local_pos + row + 1;
    const uint seq_len = prefix_len + local_len;
    const device float *qh = q + ((ulong)row * num_q_heads + h) * head_dim;
    const device float *prefix_kbase = prefix_k + (ulong)kvh * prefix_stride * head_dim;
    const device float *prefix_vbase = prefix_v + (ulong)kvh * head_dim * prefix_stride;
    const device float *local_kbase = local_k + (ulong)kvh * local_stride * head_dim;
    const device float *local_vbase = local_v + (ulong)kvh * head_dim * local_stride;
    device float *oh = out + ((ulong)row * num_q_heads + h) * head_dim;

    for (uint d = tid; d < head_dim; d += threads_per_threadgroup) {
        q_cached[d] = qh[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output_acc = 0.0f;
    const bool writes_output_dim = tid < head_dim;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += SDPA_PREFILL_TILE_SEQ) {
        const uint tile_len = min(SDPA_PREFILL_TILE_SEQ, seq_len - tile_start);

        float local_max = -INFINITY;
        for (uint tile_offset = tid; tile_offset < tile_len; tile_offset += threads_per_threadgroup) {
            const uint t = tile_start + tile_offset;
            const bool use_prefix = t < prefix_len;
            const uint local_t = use_prefix ? 0 : t - prefix_len;
            const device float *kt = use_prefix ?
                prefix_kbase + (ulong)t * head_dim :
                local_kbase + (ulong)local_t * head_dim;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) dot += q_cached[d] * kt[d];
            const float score = dot * scale;
            scores[tile_offset] = score;
            local_max = max(local_max, score);
        }

        partial[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_max = partial[0];

        float local_sum = 0.0f;
        for (uint tile_offset = tid; tile_offset < tile_len; tile_offset += threads_per_threadgroup) {
            const float score_exp = exp(scores[tile_offset] - tile_max);
            scores[tile_offset] = score_exp;
            local_sum += score_exp;
        }

        partial[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float tile_sum = partial[0];

        const float next_max = max(running_max, tile_max);
        const float previous_scale = running_sum == 0.0f ? 0.0f : exp(running_max - next_max);
        const float tile_scale = exp(tile_max - next_max);
        const float next_sum = running_sum * previous_scale + tile_sum * tile_scale;

        if (writes_output_dim) {
            float tile_acc = 0.0f;
            for (uint tile_offset = 0; tile_offset < tile_len; tile_offset += 1) {
                const uint t = tile_start + tile_offset;
                const bool use_prefix = t < prefix_len;
                const uint local_t = use_prefix ? 0 : t - prefix_len;
                const float vv = use_prefix ?
                    prefix_vbase[(ulong)tid * prefix_stride + t] :
                    local_vbase[(ulong)tid * local_stride + local_t];
                tile_acc += scores[tile_offset] * vv;
            }
            output_acc = (output_acc * running_sum * previous_scale + tile_acc * tile_scale) / next_sum;
        }

        running_max = next_max;
        running_sum = next_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (writes_output_dim) oh[tid] = output_acc;
}

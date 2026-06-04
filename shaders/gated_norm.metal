#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

// Gated RMSNorm (Qwen3NextRMSNormGated): out = rms_norm(x, weight, eps) * silu(gate),
// in f32. Verified against MLX. x/gate/out: [rows, dim]; weight: bf16 [dim].
// One thread per row.
kernel void rmsnorm_gated_f32(
    device const float *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const device float *xr = x + (ulong)row * dim;
    const device float *gr = gate + (ulong)row * dim;
    device float *yr = out + (ulong)row * dim;

    float ss = 0.0f;
    for (uint i = 0; i < dim; i++) ss += xr[i] * xr[i];
    const float scale = rsqrt(ss / float(dim) + eps);
    for (uint i = 0; i < dim; i++) {
        const float n = xr[i] * scale * bf16_to_float(weight[i]);
        const float gg = gr[i];
        yr[i] = n * (gg / (1.0f + exp(-gg))); // * silu(gate)
    }
}

// Weightless RMSNorm scaled by a scalar: y = (x * rsqrt(mean(x^2)+eps)) * scale.
// The linear-attention q/k normalization: rms_norm(x, None) then * inv_scale^k
// (q uses inv_scale^2, k uses inv_scale, inv_scale = head_k_dim^-0.5). One thread
// per row (head).
kernel void rmsnorm_scale_f32(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &dim [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    constant float &scale_mul [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const device float *xr = x + (ulong)row * dim;
    device float *yr = y + (ulong)row * dim;
    float ss = 0.0f;
    for (uint i = 0; i < dim; i++) ss += xr[i] * xr[i];
    const float s = rsqrt(ss / float(dim) + eps) * scale_mul;
    for (uint i = 0; i < dim; i++) yr[i] = xr[i] * s;
}

constant uint QK_NORM_PREFILL_THREADS_PER_THREADGROUP = 128;

// Qwen3.5 linear-attention prefill q/k L2 norm, directly from conv_out:
// conv_out rows are [q, k, v], q_norm/key_norm rows are [tokens, Hk, Dk].
kernel void qk_l2norm_prefill_f32(
    device const float *conv_out [[buffer(0)]],
    device float *q_norm [[buffer(1)]],
    device float *k_norm [[buffer(2)]],
    constant uint &tokens [[buffer(3)]],
    constant uint &Hk [[buffer(4)]],
    constant uint &Dk [[buffer(5)]],
    constant uint &conv_dim [[buffer(6)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    const uint rows = tokens * Hk;
    if (rows == 0 || Dk == 0 || row >= rows) return;
    if (threads_per_threadgroup > QK_NORM_PREFILL_THREADS_PER_THREADGROUP) return;

    const uint token_index = row / Hk;
    const uint head_index = row - token_index * Hk;
    const uint key_dim = Hk * Dk;
    if (conv_dim < 2 * key_dim) return;

    const ulong conv_row_base = (ulong)token_index * conv_dim;
    const ulong head_base = (ulong)head_index * Dk;

    threadgroup float q_partial[QK_NORM_PREFILL_THREADS_PER_THREADGROUP];
    threadgroup float k_partial[QK_NORM_PREFILL_THREADS_PER_THREADGROUP];
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint dk = tid; dk < Dk; dk += threads_per_threadgroup) {
        const float q_value = conv_out[conv_row_base + head_base + dk];
        const float k_value = conv_out[conv_row_base + key_dim + head_base + dk];
        q_sum += q_value * q_value;
        k_sum += k_value * k_value;
    }

    q_partial[tid] = q_sum;
    k_partial[tid] = k_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_partial[tid] += q_partial[tid + stride];
            k_partial[tid] += k_partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv_norm = rsqrt(q_partial[0] + 1.0e-6f);
    const float k_inv_norm = rsqrt(k_partial[0] + 1.0e-6f);
    const float q_scale = rsqrt(float(Dk));
    const ulong norm_base = (ulong)row * Dk;
    for (uint dk = tid; dk < Dk; dk += threads_per_threadgroup) {
        q_norm[norm_base + dk] = conv_out[conv_row_base + head_base + dk] * q_inv_norm * q_scale;
        k_norm[norm_base + dk] = conv_out[conv_row_base + key_dim + head_base + dk] * k_inv_norm;
    }
}

// Gated-DeltaNet gate coefficients (per value head):
//   g    = exp(-exp(A_log) * softplus(a + dt_bias))
//   beta = sigmoid(b)
// A_log/a/b: f32 [Hv]; dt_bias: bf16 [Hv]. Verified against MLX compute_g/sigmoid.
kernel void gating_f32(
    device const float *A_log [[buffer(0)]],
    device const float *a [[buffer(1)]],
    device const ushort *dt_bias [[buffer(2)]],
    device const float *b [[buffer(3)]],
    device float *g [[buffer(4)]],
    device float *beta [[buffer(5)]],
    constant uint &Hv [[buffer(6)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= Hv) return;
    const float x = a[h] + bf16_to_float(dt_bias[h]);
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x))); // stable
    // A_log > ~88 would saturate exp() to +inf -> g=0 (not NaN); real values ~[-5,-2].
    g[h] = exp(-exp(A_log[h]) * softplus);
    beta[h] = 1.0f / (1.0f + exp(-b[h]));
}

// Batched gate coefficients for prompt prefill.
// a/b/g/beta: [tokens, Hv]; A_log/dt_bias: [Hv].
kernel void gating_many_f32(
    device const float *A_log [[buffer(0)]],
    device const float *a [[buffer(1)]],
    device const ushort *dt_bias [[buffer(2)]],
    device const float *b [[buffer(3)]],
    device float *g [[buffer(4)]],
    device float *beta [[buffer(5)]],
    constant uint &Hv [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (Hv == 0) return;
    const uint row = i / Hv;
    const uint h = i - row * Hv;
    if (row >= tokens) return;

    const ulong offset = (ulong)row * Hv + h;
    const float x = a[offset] + bf16_to_float(dt_bias[h]);
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    g[offset] = exp(-exp(A_log[h]) * softplus);
    beta[offset] = 1.0f / (1.0f + exp(-b[offset]));
}

#if defined(__HAVE_BFLOAT__)
kernel void rmsnorm_gated_bf16(
    device const bfloat *x [[buffer(0)]],
    device const ushort *weight [[buffer(1)]],
    device const bfloat *gate [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const device bfloat *xr = x + (ulong)row * dim;
    const device bfloat *gr = gate + (ulong)row * dim;
    device bfloat *yr = out + (ulong)row * dim;

    float ss = 0.0f;
    for (uint i = 0; i < dim; i++) {
        const float value = float(xr[i]);
        ss += value * value;
    }
    const float scale = rsqrt(ss / float(dim) + eps);
    for (uint i = 0; i < dim; i++) {
        const float n = float(xr[i]) * scale * bf16_to_float(weight[i]);
        const float gg = float(gr[i]);
        yr[i] = bfloat(n * (gg / (1.0f + exp(-gg))));
    }
}

kernel void qk_l2norm_prefill_bf16(
    device const bfloat *conv_out [[buffer(0)]],
    device float *q_norm [[buffer(1)]],
    device float *k_norm [[buffer(2)]],
    constant uint &tokens [[buffer(3)]],
    constant uint &Hk [[buffer(4)]],
    constant uint &Dk [[buffer(5)]],
    constant uint &conv_dim [[buffer(6)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    const uint rows = tokens * Hk;
    if (rows == 0 || Dk == 0 || row >= rows) return;
    if (threads_per_threadgroup > QK_NORM_PREFILL_THREADS_PER_THREADGROUP) return;

    const uint token_index = row / Hk;
    const uint head_index = row - token_index * Hk;
    const uint key_dim = Hk * Dk;
    if (conv_dim < 2 * key_dim) return;

    const ulong conv_row_base = (ulong)token_index * conv_dim;
    const ulong head_base = (ulong)head_index * Dk;

    threadgroup float q_partial[QK_NORM_PREFILL_THREADS_PER_THREADGROUP];
    threadgroup float k_partial[QK_NORM_PREFILL_THREADS_PER_THREADGROUP];
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint dk = tid; dk < Dk; dk += threads_per_threadgroup) {
        const float q_value = float(conv_out[conv_row_base + head_base + dk]);
        const float k_value = float(conv_out[conv_row_base + key_dim + head_base + dk]);
        q_sum += q_value * q_value;
        k_sum += k_value * k_value;
    }

    q_partial[tid] = q_sum;
    k_partial[tid] = k_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_threadgroup >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_partial[tid] += q_partial[tid + stride];
            k_partial[tid] += k_partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv_norm = rsqrt(q_partial[0] + 1.0e-6f);
    const float k_inv_norm = rsqrt(k_partial[0] + 1.0e-6f);
    const float q_scale = rsqrt(float(Dk));
    const ulong norm_base = (ulong)row * Dk;
    for (uint dk = tid; dk < Dk; dk += threads_per_threadgroup) {
        q_norm[norm_base + dk] = float(conv_out[conv_row_base + head_base + dk]) * q_inv_norm * q_scale;
        k_norm[norm_base + dk] = float(conv_out[conv_row_base + key_dim + head_base + dk]) * k_inv_norm;
    }
}

kernel void gating_many_bf16(
    device const float *A_log [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const ushort *dt_bias [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device float *g [[buffer(4)]],
    device float *beta [[buffer(5)]],
    constant uint &Hv [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (Hv == 0) return;
    const uint row = i / Hv;
    const uint h = i - row * Hv;
    if (row >= tokens) return;

    const ulong offset = (ulong)row * Hv + h;
    const float x = float(a[offset]) + bf16_to_float(dt_bias[h]);
    const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    g[offset] = exp(-exp(A_log[h]) * softplus);
    beta[offset] = 1.0f / (1.0f + exp(-float(b[offset])));
}
#endif

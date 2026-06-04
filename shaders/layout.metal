#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

// Layout/copy kernels that replace the host-side buffer reshuffles in the decode
// path. Moving them onto the GPU removes the mid-forward CPU reads that otherwise
// force a per-kernel sync, so the whole forward can be batched into one command
// buffer.

// Split interleaved [num_heads, 2*head_dim] (per head: query | gate) into
// separate queries[num_heads, head_dim] and gate[num_heads, head_dim].
kernel void split_qg_f32(
    device const float *qp [[buffer(0)]],
    device float *queries [[buffer(1)]],
    device float *gate [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant uint &num_heads [[buffer(4)]],
    uint i [[thread_position_in_grid]]) // grid = num_heads * head_dim
{
    if (i >= num_heads * head_dim) return;
    const uint h = i / head_dim;
    const uint d = i % head_dim;
    queries[i] = qp[h * 2 * head_dim + d];
    gate[i] = qp[h * 2 * head_dim + head_dim + d];
}

// Batched split for full-attention prefill. qp is
// [tokens, num_heads, 2*head_dim].
kernel void split_qg_many_f32(
    device const float *qp [[buffer(0)]],
    device float *queries [[buffer(1)]],
    device float *gate [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant uint &num_heads [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    const uint per_token = num_heads * head_dim;
    if (per_token == 0) return;
    const uint row = i / per_token;
    const uint rem = i - row * per_token;
    if (row >= tokens) return;

    const uint h = rem / head_dim;
    const uint d = rem - h * head_dim;
    const ulong src = ((ulong)row * num_heads + h) * (2 * head_dim);
    const ulong dst = (ulong)row * per_token + rem;
    queries[dst] = qp[src + d];
    gate[dst] = qp[src + head_dim + d];
}

#if defined(__HAVE_BFLOAT__)
kernel void split_qg_bf16(
    device const bfloat *qp [[buffer(0)]],
    device bfloat *queries [[buffer(1)]],
    device bfloat *gate [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant uint &num_heads [[buffer(4)]],
    uint i [[thread_position_in_grid]]) // grid = num_heads * head_dim
{
    if (i >= num_heads * head_dim) return;
    const uint h = i / head_dim;
    const uint d = i % head_dim;
    queries[i] = qp[h * 2 * head_dim + d];
    gate[i] = qp[h * 2 * head_dim + head_dim + d];
}

kernel void split_qg_many_bf16(
    device const bfloat *qp [[buffer(0)]],
    device bfloat *queries [[buffer(1)]],
    device bfloat *gate [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant uint &num_heads [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    const uint per_token = num_heads * head_dim;
    if (per_token == 0) return;
    const uint row = i / per_token;
    const uint rem = i - row * per_token;
    if (row >= tokens) return;

    const uint h = rem / head_dim;
    const uint d = rem - h * head_dim;
    const ulong src = ((ulong)row * num_heads + h) * (2 * head_dim);
    const ulong dst = (ulong)row * per_token + rem;
    queries[dst] = qp[src + d];
    gate[dst] = qp[src + head_dim + d];
}
#endif

// Append the new per-kv-head vectors. K uses [num_kv, cap, head_dim]; V uses
// Kestrel's value-dim-major layout [num_kv, head_dim, cap] so attention can read
// a fixed value dimension contiguously across tokens.
kernel void kv_append_f32(
    device const float *kbuf [[buffer(0)]],
    device const float *vbuf [[buffer(1)]],
    device float *cache_k [[buffer(2)]],
    device float *cache_v [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &cap [[buffer(5)]],
    constant uint &pos [[buffer(6)]],
    constant uint &num_kv [[buffer(7)]],
    uint i [[thread_position_in_grid]]) // grid = num_kv * head_dim
{
    if (i >= num_kv * head_dim) return;
    const uint kvh = i / head_dim;
    const uint d = i % head_dim;
    const uint k_dst = (kvh * cap + pos) * head_dim + d;
    const uint v_dst = (kvh * head_dim + d) * cap + pos;
    cache_k[k_dst] = kbuf[i];
    cache_v[v_dst] = vbuf[i];
}

// Batched KV publication for full-attention prefill. kbuf/vbuf are
// [tokens, num_kv, head_dim]. K cache is [num_kv, cap, head_dim]; V cache is
// [num_kv, head_dim, cap].
kernel void kv_append_many_f32(
    device const float *kbuf [[buffer(0)]],
    device const float *vbuf [[buffer(1)]],
    device float *cache_k [[buffer(2)]],
    device float *cache_v [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &cap [[buffer(5)]],
    constant uint &start_pos [[buffer(6)]],
    constant uint &num_kv [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    const uint per_token = num_kv * head_dim;
    if (per_token == 0) return;
    const uint row = i / per_token;
    const uint rem = i - row * per_token;
    if (row >= tokens) return;

    const uint kvh = rem / head_dim;
    const uint d = rem - kvh * head_dim;
    const ulong src = ((ulong)row * num_kv + kvh) * head_dim + d;
    const ulong k_dst = ((ulong)kvh * cap + (start_pos + row)) * head_dim + d;
    const ulong v_dst = ((ulong)kvh * head_dim + d) * cap + (start_pos + row);
    cache_k[k_dst] = kbuf[src];
    cache_v[v_dst] = vbuf[src];
}

// Split a contiguous [2*key_dim + val_dim] buffer into q[key_dim], k[key_dim],
// v[val_dim] — the in-proj qkv output of the linear-attention block.
kernel void split_qkv_f32(
    device const float *src [[buffer(0)]],
    device float *q [[buffer(1)]],
    device float *k [[buffer(2)]],
    device float *v [[buffer(3)]],
    constant uint &key_dim [[buffer(4)]],
    constant uint &val_dim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) // grid = 2*key_dim + val_dim
{
    if (i >= 2 * key_dim + val_dim) return;
    if (i < key_dim)
        q[i] = src[i];
    else if (i < 2 * key_dim)
        k[i - key_dim] = src[i];
    else
        v[i - 2 * key_dim] = src[i];
}

#if defined(__HAVE_BFLOAT__)
kernel void split_qkv_bf16(
    device const bfloat *src [[buffer(0)]],
    device bfloat *q [[buffer(1)]],
    device bfloat *k [[buffer(2)]],
    device bfloat *v [[buffer(3)]],
    constant uint &key_dim [[buffer(4)]],
    constant uint &val_dim [[buffer(5)]],
    uint i [[thread_position_in_grid]]) // grid = 2*key_dim + val_dim
{
    if (i >= 2 * key_dim + val_dim) return;
    if (i < key_dim)
        q[i] = src[i];
    else if (i < 2 * key_dim)
        k[i - key_dim] = src[i];
    else
        v[i - 2 * key_dim] = src[i];
}
#endif

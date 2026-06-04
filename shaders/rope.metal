#include <metal_stdlib>
using namespace metal;

// RoPE (traditional=false / NeoX half-split), in-place over the first `rot` dims
// of each head_dim-wide row; the tail dims pass through unchanged. Verified
// against mx.fast.rope.
//   inv_freq_i = base^(-2i/rot),  th = pos * inv_freq_i   (i in 0..rot/2)
//   x[i]       =  x[i]*cos(th) - x[i+rot/2]*sin(th)
//   x[i+rot/2] =  x[i+rot/2]*cos(th) + x[i]*sin(th)
// Grid: rows * (rot/2) — one thread per rotated pair; pairs are disjoint so the
// in-place writes never race.
kernel void rope_f32(
    device float *x [[buffer(0)]],
    constant uint &head_dim [[buffer(1)]],
    constant uint &rot [[buffer(2)]],
    constant float &base [[buffer(3)]],
    constant uint &pos [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_dim = rot / 2;
    const uint row = gid / half_dim;
    if (row >= rows) return;
    const uint i = gid % half_dim;

    const ulong off = (ulong)row * head_dim;
    const float inv_freq = pow(base, -(2.0f * float(i)) / float(rot));
    const float th = float(pos) * inv_freq;
    const float c = cos(th);
    const float s = sin(th);
    const float a = x[off + i];
    const float b = x[off + i + half_dim];
    x[off + i] = a * c - b * s;
    x[off + i + half_dim] = b * c + a * s;
}

// Batched RoPE for prompt prefill. x is [tokens, rows_per_token, head_dim], and
// each token row uses logical position start_pos + token_index.
kernel void rope_many_f32(
    device float *x [[buffer(0)]],
    constant uint &head_dim [[buffer(1)]],
    constant uint &rot [[buffer(2)]],
    constant float &base [[buffer(3)]],
    constant uint &start_pos [[buffer(4)]],
    constant uint &rows_per_token [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_dim = rot / 2;
    if (half_dim == 0 || rows_per_token == 0) return;
    const uint row = gid / half_dim;
    if (row >= rows) return;
    const uint i = gid % half_dim;
    const uint token = row / rows_per_token;
    const uint pos = start_pos + token;

    const ulong off = (ulong)row * head_dim;
    const float inv_freq = pow(base, -(2.0f * float(i)) / float(rot));
    const float th = float(pos) * inv_freq;
    const float c = cos(th);
    const float s = sin(th);
    const float a = x[off + i];
    const float b = x[off + i + half_dim];
    x[off + i] = a * c - b * s;
    x[off + i + half_dim] = b * c + a * s;
}

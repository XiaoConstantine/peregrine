#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

// q4 embedding gather: dequantize row `token` of embed_tokens into out[hidden].
//   out[h] = scale[token, h/64] * q[token,h] + bias[token, h/64]
// weight: u32 [vocab, hidden/8] (8 nibbles/word, low-first); scales/biases bf16
// [vocab, hidden/64]. Same affine layout as the q4 qmv, gathering one row.
kernel void q4_embedding_gather_f32(
    device const uint *weight [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &token [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= hidden) return;
    const uint words = hidden / 8;
    const uint groups = hidden / 64;
    const uint q = (weight[(ulong)token * words + (h >> 3)] >> ((h & 7) * 4)) & 0xF;
    const uint g = h >> 6;
    out[h] = bf16_to_float(scales[(ulong)token * groups + g]) * float(q) +
        bf16_to_float(biases[(ulong)token * groups + g]);
}

// Batched q4 embedding gather for prompt prefill.
// token_ids: [tokens]; out: f32 [tokens, hidden].
kernel void q4_embedding_gather_many_f32(
    device const uint *weight [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device float *out [[buffer(3)]],
    device const uint *token_ids [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (hidden == 0) return;
    const uint row = tid / hidden;
    const uint h = tid - row * hidden;
    if (row >= tokens) return;

    const uint token = token_ids[row];
    const uint words = hidden / 8;
    const uint groups = hidden / 64;
    const uint q = (weight[(ulong)token * words + (h >> 3)] >> ((h & 7) * 4)) & 0xF;
    const uint g = h >> 6;
    out[(ulong)row * hidden + h] = bf16_to_float(scales[(ulong)token * groups + g]) * float(q) +
        bf16_to_float(biases[(ulong)token * groups + g]);
}

#if defined(__HAVE_BFLOAT__)
// BF16 activation-path variants for Kestrel-style layer-major prefill.
kernel void q4_embedding_gather_bf16(
    device const uint *weight [[buffer(0)]],
    device const bfloat *scales [[buffer(1)]],
    device const bfloat *biases [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &token [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= hidden) return;
    const uint words = hidden / 8;
    const uint groups = hidden / 64;
    const uint q = (weight[(ulong)token * words + (h >> 3)] >> ((h & 7) * 4)) & 0xF;
    const uint g = h >> 6;
    out[h] = bfloat(float(scales[(ulong)token * groups + g]) * float(q) +
        float(biases[(ulong)token * groups + g]));
}

kernel void q4_embedding_gather_many_bf16(
    device const uint *weight [[buffer(0)]],
    device const bfloat *scales [[buffer(1)]],
    device const bfloat *biases [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    device const uint *token_ids [[buffer(4)]],
    constant uint &hidden [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (hidden == 0) return;
    const uint row = tid / hidden;
    const uint h = tid - row * hidden;
    if (row >= tokens) return;

    const uint token = token_ids[row];
    const uint words = hidden / 8;
    const uint groups = hidden / 64;
    const uint q = (weight[(ulong)token * words + (h >> 3)] >> ((h & 7) * 4)) & 0xF;
    const uint g = h >> 6;
    out[(ulong)row * hidden + h] = bfloat(float(scales[(ulong)token * groups + g]) * float(q) +
        float(biases[(ulong)token * groups + g]));
}
#endif

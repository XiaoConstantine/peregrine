#include <metal_stdlib>
using namespace metal;

// out[i] = a[i] * sigmoid(b[i]) — the full-attention output gate
// (attn_output * sigmoid(gate)).
kernel void mul_sigmoid_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float bb = b[i];
    out[i] = a[i] * (1.0f / (1.0f + exp(-bb)));
}

// out[i] = a[i] + b[i] — the residual add (out may alias a or b).
kernel void add_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

// out[i] = silu(a[i]) * b[i] = (a[i] * sigmoid(a[i])) * b[i] — the SwiGLU MLP.
kernel void silu_mul_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float aa = a[i];
    out[i] = (aa / (1.0f + exp(-aa))) * b[i];
}

#if defined(__HAVE_BFLOAT__)
kernel void f32_to_bf16(
    device const float *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    output[i] = bfloat(input[i]);
}

kernel void bf16_to_f32(
    device const bfloat *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    output[i] = float(input[i]);
}

kernel void silu_mul_pair_inplace_bf16(
    device bfloat *gate [[buffer(0)]],
    device const bfloat *up [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float gate_value = float(gate[i]);
    const float up_value = float(up[i]);
    const float silu = gate_value / (1.0f + exp(-gate_value));
    gate[i] = bfloat(silu * up_value);
}

kernel void add_bf16(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    out[i] = bfloat(float(a[i]) + float(b[i]));
}

kernel void sigmoid_mul_pair_inplace_bf16(
    device bfloat *input [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const float input_value = float(input[i]);
    const float gate_value = float(gate[i]);
    const float sigmoid = 1.0f / (1.0f + exp(-gate_value));
    input[i] = bfloat(input_value * sigmoid);
}
#endif

// Copy one row out of a row-major f32 matrix.
kernel void copy_row_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &row [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= width) return;
    output[i] = input[(ulong)row * width + i];
}

#if defined(__HAVE_BFLOAT__)
// Copy one row out of a row-major bf16 matrix.
kernel void copy_row_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &row [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= width) return;
    output[i] = input[(ulong)row * width + i];
}

// Write a single bf16 row into a row-major matrix.
kernel void write_row_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &row [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= width) return;
    output[(ulong)row * width + i] = input[i];
}

// Row-wise concat for Qwen3.5 MTP pre-fc input:
// out[row, i] = a[row, i], out[row, width + i] = b[row, i].
kernel void concat_two_rows_bf16(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    const uint total = width * rows;
    if (i >= total) return;
    const uint row = i / width;
    const uint col = i - row * width;
    const ulong out_base = (ulong)row * (ulong)width * 2ul;
    out[out_base + col] = a[i];
    out[out_base + (ulong)width + col] = b[i];
}
#endif

#include <metal_stdlib>
#include "bf16.h"
using namespace metal;

// Depthwise causal conv1d (kernel K) + SiLU for one decode step, then roll the
// rolling state cache. Verified against mlx nn.Conv1d + nn.silu.
//   out[c] = silu( sum_{j<K-1} w[c,j]*state[j,c] + w[c,K-1]*qkv[c] )   (w[0] = oldest)
//   state <- [state[1], ..., state[K-2], qkv]                          (roll)
// state: f32 [K-1, conv_dim] (in-place); weight: bf16 [conv_dim, K]; qkv/out:
// f32 [conv_dim]. One thread per channel (each owns its column -> race-free).
kernel void conv1d_silu_decode_f32(
    device float *state [[buffer(0)]],
    device const float *qkv [[buffer(1)]],
    device const ushort *weight [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &conv_dim [[buffer(4)]],
    constant uint &kernel_size [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= conv_dim) return;
    const uint km1 = kernel_size - 1;
    const float xnew = qkv[c];

    float acc = 0.0f;
    for (uint j = 0; j < km1; j++) {
        acc += bf16_to_float(weight[c * kernel_size + j]) * state[j * conv_dim + c];
    }
    acc += bf16_to_float(weight[c * kernel_size + km1]) * xnew;
    out[c] = acc / (1.0f + exp(-acc)); // SiLU

    // Roll the cache: drop the oldest, append the new input. (km1==0 -> K=1 has
    // no history; the guard also avoids an unsigned underflow on (km1-1).)
    for (uint j = 0; j + 1 < km1; j++) state[j * conv_dim + c] = state[(j + 1) * conv_dim + c];
    if (km1 > 0) state[(km1 - 1) * conv_dim + c] = xnew;
}

#if defined(__HAVE_BFLOAT__)
// BF16 activation decode variant for the future coherent BF16 linear-attention
// route. The rolling conv state stays f32, matching Peregrine's state contract.
kernel void conv1d_silu_decode_bf16(
    device float *state [[buffer(0)]],
    device const bfloat *qkv [[buffer(1)]],
    device const ushort *weight [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &conv_dim [[buffer(4)]],
    constant uint &kernel_size [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= conv_dim || kernel_size == 0) return;
    const uint km1 = kernel_size - 1;
    const float xnew = float(qkv[c]);

    float acc = 0.0f;
    for (uint j = 0; j < km1; j++) {
        acc += bf16_to_float(weight[c * kernel_size + j]) * state[j * conv_dim + c];
    }
    acc += bf16_to_float(weight[c * kernel_size + km1]) * xnew;
    out[c] = bfloat(acc / (1.0f + exp(-acc)));

    for (uint j = 0; j + 1 < km1; j++) state[j * conv_dim + c] = state[(j + 1) * conv_dim + c];
    if (km1 > 0) state[(km1 - 1) * conv_dim + c] = xnew;
}
#endif

// Batched causal conv1d + SiLU for prompt prefill. Kestrel's Qwen3.5-9B
// default computes four prompt-token conv outputs per thread. Peregrine keeps
// its history-major state layout, so this uses the same math with that layout.
kernel void conv1d_silu_prefill_vec4_f32(
    device const float *input [[buffer(0)]], // [tokens, conv_dim]
    device const ushort *weight [[buffer(1)]],
    device const float *state [[buffer(2)]], // [kernel_size - 1, conv_dim]
    device float *out [[buffer(3)]],         // [tokens, conv_dim]
    constant uint &tokens [[buffer(4)]],
    constant uint &conv_dim [[buffer(5)]],
    constant uint &kernel_size [[buffer(6)]],
    constant uint &has_previous_state [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (conv_dim == 0 || kernel_size == 0) return;
    const uint tokens_per_thread = 4;
    const uint token_block = tid / conv_dim;
    const uint c = tid - token_block * conv_dim;
    const uint token_base = token_block * tokens_per_thread;
    if (c >= conv_dim || token_base >= tokens) return;

    const uint km1 = kernel_size - 1;
    for (uint offset_token = 0; offset_token < tokens_per_thread; offset_token++) {
        const uint t = token_base + offset_token;
        if (t >= tokens) return;

        float acc = 0.0f;
        for (uint j = 0; j < kernel_size; j++) {
            const uint source_index = t + j;
            float x = 0.0f;
            if (source_index < km1) {
                if (has_previous_state != 0) x = state[source_index * conv_dim + c];
            } else {
                const uint input_t = source_index - km1;
                if (input_t < tokens) x = input[(ulong)input_t * conv_dim + c];
            }
            acc += bf16_to_float(weight[c * kernel_size + j]) * x;
        }
        out[(ulong)t * conv_dim + c] = acc / (1.0f + exp(-acc));
    }
}

// Publish the rolling conv state after a prefill chunk. State order remains
// oldest-to-newest, matching conv1d_silu_decode_f32.
kernel void conv1d_silu_prefill_state_f32(
    device const float *input [[buffer(0)]], // [tokens, conv_dim]
    device float *state [[buffer(1)]],       // [kernel_size - 1, conv_dim]
    constant uint &tokens [[buffer(2)]],
    constant uint &conv_dim [[buffer(3)]],
    constant uint &kernel_size [[buffer(4)]],
    constant uint &has_previous_state [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= conv_dim || kernel_size == 0) return;
    const uint km1 = kernel_size - 1;
    for (uint j = 0; j < km1; j++) {
        const uint source_index = tokens + j;
        float x = 0.0f;
        if (source_index < km1) {
            if (has_previous_state != 0) x = state[source_index * conv_dim + c];
        } else {
            const uint input_t = source_index - km1;
            if (input_t < tokens) x = input[(ulong)input_t * conv_dim + c];
        }
        state[j * conv_dim + c] = x;
    }
}

#if defined(__HAVE_BFLOAT__)
// BF16 activation variant for the layer-major prompt path. Peregrine keeps its
// small history-major conv state in f32 while prompt activations stay bf16.
kernel void conv1d_silu_prefill_vec4_bf16(
    device const bfloat *input [[buffer(0)]], // [tokens, conv_dim]
    device const ushort *weight [[buffer(1)]],
    device const float *state [[buffer(2)]],  // [kernel_size - 1, conv_dim]
    device bfloat *out [[buffer(3)]],         // [tokens, conv_dim]
    constant uint &tokens [[buffer(4)]],
    constant uint &conv_dim [[buffer(5)]],
    constant uint &kernel_size [[buffer(6)]],
    constant uint &has_previous_state [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (conv_dim == 0 || kernel_size == 0) return;
    const uint tokens_per_thread = 4;
    const uint token_block = tid / conv_dim;
    const uint c = tid - token_block * conv_dim;
    const uint token_base = token_block * tokens_per_thread;
    if (c >= conv_dim || token_base >= tokens) return;

    const uint km1 = kernel_size - 1;
    for (uint offset_token = 0; offset_token < tokens_per_thread; offset_token++) {
        const uint t = token_base + offset_token;
        if (t >= tokens) return;

        float acc = 0.0f;
        for (uint j = 0; j < kernel_size; j++) {
            const uint source_index = t + j;
            float x = 0.0f;
            if (source_index < km1) {
                if (has_previous_state != 0) x = state[source_index * conv_dim + c];
            } else {
                const uint input_t = source_index - km1;
                if (input_t < tokens) x = float(input[(ulong)input_t * conv_dim + c]);
            }
            acc += bf16_to_float(weight[c * kernel_size + j]) * x;
        }
        out[(ulong)t * conv_dim + c] = bfloat(acc / (1.0f + exp(-acc)));
    }
}

kernel void conv1d_silu_prefill_state_bf16(
    device const bfloat *input [[buffer(0)]], // [tokens, conv_dim]
    device float *state [[buffer(1)]],        // [kernel_size - 1, conv_dim]
    constant uint &tokens [[buffer(2)]],
    constant uint &conv_dim [[buffer(3)]],
    constant uint &kernel_size [[buffer(4)]],
    constant uint &has_previous_state [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= conv_dim || kernel_size == 0) return;
    const uint km1 = kernel_size - 1;
    for (uint j = 0; j < km1; j++) {
        const uint source_index = tokens + j;
        float x = 0.0f;
        if (source_index < km1) {
            if (has_previous_state != 0) x = state[source_index * conv_dim + c];
        } else {
            const uint input_t = source_index - km1;
            if (input_t < tokens) x = float(input[(ulong)input_t * conv_dim + c]);
        }
        state[j * conv_dim + c] = x;
    }
}
#endif

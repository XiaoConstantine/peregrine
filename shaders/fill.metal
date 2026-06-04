#include <metal_stdlib>
using namespace metal;

// Trivial kernel: write a constant value into every element of `out`.
// Its only job right now is to prove the Metal toolchain wiring
// (compile -> peregrine.metallib). The runtime layer that loads and
// dispatches it lands in a later step.
kernel void fill_f32(
    device float *out [[buffer(0)]],
    constant float &value [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = value;
}

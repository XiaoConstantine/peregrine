#pragma once
#include <metal_stdlib>
using namespace metal;

// bf16 (stored as the high 16 bits of an f32) -> f32. `static` keeps it
// TU-private so each shader's copy can't collide when the .air files link
// into one metallib.
static inline float bf16_to_float(ushort b)
{
    return as_type<float>(uint(b) << 16);
}

//! Greedy argmax helpers for Kestrel-style q4 lm-head partial maxima on the GPU.

const std = @import("std");
const metal = @import("../runtime/metal.zig");

const q4_affine_fast_argmax_outputs_per_partial: u32 = 8;
const argmax_pairs_threads: u32 = 256;

pub fn q4AffineFastArgmaxPartialCount(vocab_size: u32) !u32 {
    if (vocab_size == 0 or vocab_size % q4_affine_fast_argmax_outputs_per_partial != 0) return error.UnsupportedQuantization;
    return vocab_size / q4_affine_fast_argmax_outputs_per_partial;
}

pub fn encodePairsF32U32(
    ws: *metal.Workspace,
    pipeline: metal.Pipeline,
    input_values: metal.Buffer,
    input_indices: metal.Buffer,
    output_index: metal.Buffer,
    pair_count: u32,
) !void {
    if (pair_count == 0) return error.InvalidArgmaxInput;
    const values_bytes = std.math.mul(usize, @as(usize, pair_count), @sizeOf(f32)) catch return error.ContextSizeOverflow;
    if (input_values.length < values_bytes) return error.InputBufferTooSmall;
    const indices_bytes = std.math.mul(usize, @as(usize, pair_count), @sizeOf(u32)) catch return error.ContextSizeOverflow;
    if (input_indices.length < indices_bytes) return error.InputBufferTooSmall;
    if (output_index.length < @sizeOf(u32)) return error.OutputBufferTooSmall;
    const pair_count_buf = try ws.u32buf(pair_count);
    try ws.cmd.dispatch1DWithThreadgroup(
        pipeline,
        &.{ input_values, input_indices, output_index, pair_count_buf },
        @as(usize, argmax_pairs_threads),
        @as(usize, argmax_pairs_threads),
    );
}

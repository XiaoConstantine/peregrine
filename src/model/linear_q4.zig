//! Q4Linear: a q4 group-64 affine weight resident on the GPU. The reusable
//! projection building block for Qwen q/k/v/o, gate/up/down, and lm_head.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const dims = @import("dims.zig");
const safetensors = @import("safetensors.zig");
const weight_upload = @import("upload.zig");

pub const QMV_FAST_THREADS_PER_THREADGROUP = 64;
pub const QMV_FAST_RESULTS_PER_THREADGROUP = 8;
pub const QMM_TILE_M = 32;
pub const QMM_TILE_N = 64;
pub const QMM_THREADS_PER_THREADGROUP = 256;
pub const BF16_BYTES = dims.bf16_bytes;

pub const Q4Linear = struct {
    weight: metal.Buffer,
    scales: metal.Buffer,
    biases: metal.Buffer,
    in_dim_buf: metal.Buffer,
    out_dim_buf: metal.Buffer,
    in_dim: u32,
    out_dim: u32,

    /// Upload `<base>.weight/.scales/.biases` to GPU. `base` is the tensor name
    /// without the suffix (e.g. "...self_attn.q_proj").
    pub fn upload(device: *metal.Device, queue: *metal.Queue, repo: *const safetensors.Repository, base: []const u8) !Q4Linear {
        // `key` is reused across the three lookups: repo.get hashes the name and
        // does not retain the slice.
        var key: [256]u8 = undefined;
        const w_info = repo.get(try std.fmt.bufPrint(&key, "{s}.weight", .{base})) orelse return error.TensorNotFound;
        const weight_dims = try q4WeightDims(w_info.shape);
        const out_dim = weight_dims.out_dim;
        const in_dim = weight_dims.in_dim;
        const s_info = repo.get(try std.fmt.bufPrint(&key, "{s}.scales", .{base})) orelse return error.TensorNotFound;
        const b_info = repo.get(try std.fmt.bufPrint(&key, "{s}.biases", .{base})) orelse return error.TensorNotFound;

        // pread into temporary staging, then blit into private GPU-resident
        // buffers. This matches Kestrel's interactive private-weight default.
        var weight = try weight_upload.tensorPrivate(device, queue, repo, w_info);
        errdefer weight.destroy();
        var scales = try weight_upload.tensorPrivate(device, queue, repo, s_info);
        errdefer scales.destroy();
        var biases = try weight_upload.tensorPrivate(device, queue, repo, b_info);
        errdefer biases.destroy();
        var in_dim_buf = try device.createSharedBuffer(@sizeOf(u32));
        errdefer in_dim_buf.destroy();
        in_dim_buf.slice(u32)[0] = in_dim;
        var out_dim_buf = try device.createSharedBuffer(@sizeOf(u32));
        errdefer out_dim_buf.destroy();
        out_dim_buf.slice(u32)[0] = out_dim;

        return .{
            .weight = weight,
            .scales = scales,
            .biases = biases,
            .in_dim_buf = in_dim_buf,
            .out_dim_buf = out_dim_buf,
            .in_dim = in_dim,
            .out_dim = out_dim,
        };
    }

    pub fn deinit(self: *Q4Linear) void {
        self.weight.destroy();
        self.scales.destroy();
        self.biases.destroy();
        self.in_dim_buf.destroy();
        self.out_dim_buf.destroy();
        self.* = undefined;
    }

    /// Encode bf16[token_count, in_dim] x q4[out_dim, in_dim] -> bf16[token_count, out_dim].
    /// This is the Kestrel fast one-token/decode projection shape: one 64-thread
    /// threadgroup computes 8 output rows.
    pub fn encodeBf16Fast(
        self: *const Q4Linear,
        ws: *metal.Workspace,
        pipeline: metal.Pipeline,
        input_buf: metal.Buffer,
        output_buf: metal.Buffer,
        token_count: u32,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        if (self.in_dim % 512 != 0 or self.out_dim % QMV_FAST_RESULTS_PER_THREADGROUP != 0) return error.UnsupportedQuantization;
        const token_count_buf = try ws.u32buf(token_count);
        const grid = try qmvFastGrid(token_count, self.out_dim);
        try ws.cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{ input_buf, self.weight, self.scales, self.biases, output_buf, token_count_buf, self.out_dim_buf, self.in_dim_buf },
            grid,
            QMV_FAST_THREADS_PER_THREADGROUP,
        );
    }

    pub fn encodeBf16ArgmaxPartial(
        self: *const Q4Linear,
        ws: *metal.Workspace,
        pipeline: metal.Pipeline,
        input_buf: metal.Buffer,
        partial_values: metal.Buffer,
        partial_indices: metal.Buffer,
    ) !void {
        if (self.in_dim % 512 != 0 or self.out_dim % QMV_FAST_RESULTS_PER_THREADGROUP != 0) return error.UnsupportedQuantization;
        const partial_count = self.out_dim / QMV_FAST_RESULTS_PER_THREADGROUP;
        const partial_values_bytes = std.math.mul(usize, @as(usize, partial_count), @sizeOf(f32)) catch return error.ContextSizeOverflow;
        if (partial_values.length < partial_values_bytes) return error.OutputBufferTooSmall;
        const partial_indices_bytes = std.math.mul(usize, @as(usize, partial_count), @sizeOf(u32)) catch return error.ContextSizeOverflow;
        if (partial_indices.length < partial_indices_bytes) return error.OutputBufferTooSmall;
        const grid = std.math.mul(usize, @as(usize, partial_count), @as(usize, QMV_FAST_THREADS_PER_THREADGROUP)) catch return error.ContextSizeOverflow;
        try ws.cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{ input_buf, self.weight, self.scales, self.biases, partial_values, partial_indices, self.out_dim_buf, self.in_dim_buf },
            grid,
            QMV_FAST_THREADS_PER_THREADGROUP,
        );
    }

    /// Encode bf16[token_count, in_dim] x q4[out_dim, in_dim] + residual -> bf16[token_count, out_dim].
    pub fn encodeBf16FastResidualAdd(
        self: *const Q4Linear,
        ws: *metal.Workspace,
        pipeline: metal.Pipeline,
        input_buf: metal.Buffer,
        residual_buf: metal.Buffer,
        output_buf: metal.Buffer,
        token_count: u32,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        if (self.in_dim % 512 != 0 or self.out_dim % QMV_FAST_RESULTS_PER_THREADGROUP != 0) return error.UnsupportedQuantization;
        const token_count_buf = try ws.u32buf(token_count);
        const grid = try qmvFastGrid(token_count, self.out_dim);
        try ws.cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{ input_buf, self.weight, self.scales, self.biases, residual_buf, output_buf, token_count_buf, self.out_dim_buf, self.in_dim_buf },
            grid,
            QMV_FAST_THREADS_PER_THREADGROUP,
        );
    }

    /// Encode bf16[token_count, in_dim] x q4[out_dim, in_dim] -> bf16[token_count, out_dim].
    pub fn encodePrefillQmmM32N64NtBf16(
        self: *const Q4Linear,
        ws: *metal.Workspace,
        pipeline: metal.Pipeline,
        input_buf: metal.Buffer,
        output_buf: metal.Buffer,
        token_count: u32,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const token_count_buf = try ws.u32buf(token_count);
        try ws.cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{ input_buf, self.weight, self.scales, self.biases, output_buf, token_count_buf, self.out_dim_buf, self.in_dim_buf },
            try qmmGrid(@intCast(token_count), self.out_dim),
            QMM_THREADS_PER_THREADGROUP,
        );
    }

    /// Encode bf16[token_count, in_dim] x q4[out_dim, in_dim] + residual -> bf16[token_count, out_dim].
    pub fn encodePrefillQmmM32N64NtBf16ResidualAdd(
        self: *const Q4Linear,
        ws: *metal.Workspace,
        pipeline: metal.Pipeline,
        input_buf: metal.Buffer,
        residual_buf: metal.Buffer,
        output_buf: metal.Buffer,
        token_count: u32,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const token_count_buf = try ws.u32buf(token_count);
        try ws.cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{ input_buf, self.weight, self.scales, self.biases, residual_buf, output_buf, token_count_buf, self.out_dim_buf, self.in_dim_buf },
            try qmmGrid(@intCast(token_count), self.out_dim),
            QMM_THREADS_PER_THREADGROUP,
        );
    }

    /// Encode q4[out_dim, in_dim] -> bf16[out_dim, in_dim] into an explicit
    /// command buffer. This avoids per-encode dimension constants so queued
    /// preparation command buffers can outlive their encode scope safely.
    pub fn encodeDequantizeBf16Command(
        self: *const Q4Linear,
        cmd: *metal.CommandBuffer,
        pipeline: metal.Pipeline,
        output_buf: metal.Buffer,
    ) !void {
        const byte_len = try denseRhsBf16ByteLen(self.out_dim, self.in_dim);
        if (output_buf.length < byte_len) return error.OutputBufferTooSmall;
        const total_values = try denseRhsValueCount(self.out_dim, self.in_dim);
        try cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{ self.weight, self.scales, self.biases, output_buf, self.out_dim_buf, self.in_dim_buf },
            total_values,
            QMM_THREADS_PER_THREADGROUP,
        );
    }
};

const Q4WeightDims = struct {
    out_dim: u32,
    in_dim: u32,
};

fn q4WeightDims(shape: []const u64) !Q4WeightDims {
    if (shape.len != 2) return error.UnsupportedQuantization;
    const out_dim = std.math.cast(u32, shape[0]) orelse return error.UnsupportedQuantization;
    const in_dim_words = std.math.cast(u32, shape[1]) orelse return error.UnsupportedQuantization;
    const in_dim = std.math.mul(u32, in_dim_words, 8) catch return error.UnsupportedQuantization;
    if (out_dim == 0 or in_dim == 0 or in_dim % 64 != 0) return error.UnsupportedQuantization;
    return .{ .out_dim = out_dim, .in_dim = in_dim };
}

test "q4WeightDims validates safetensors q4 weight shape" {
    try std.testing.expectEqual(Q4WeightDims{ .out_dim = 4096, .in_dim = 4096 }, try q4WeightDims(&.{ 4096, 512 }));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{}));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{4096}));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{ 4096, 512, 1 }));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{ 0, 512 }));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{ 4096, 0 }));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{ 4096, std.math.maxInt(u64) }));
    try std.testing.expectError(error.UnsupportedQuantization, q4WeightDims(&.{ std.math.maxInt(u64), 512 }));
}

pub fn denseRhsValueCount(out_dim: u32, in_dim: u32) !usize {
    if (in_dim == 0 or in_dim % 64 != 0 or out_dim == 0) return error.UnsupportedQuantization;
    return std.math.mul(usize, @as(usize, out_dim), @as(usize, in_dim));
}

pub fn denseRhsBf16ByteLen(out_dim: u32, in_dim: u32) !usize {
    return std.math.mul(usize, try denseRhsValueCount(out_dim, in_dim), BF16_BYTES);
}

pub fn qmmGrid(token_count: usize, out_dim: u32) !usize {
    if (token_count == 0 or out_dim == 0) return error.UnsupportedQuantization;
    const row_groups = std.math.divCeil(usize, token_count, QMM_TILE_M) catch return error.ContextSizeOverflow;
    const col_groups = std.math.divCeil(usize, @as(usize, out_dim), QMM_TILE_N) catch return error.ContextSizeOverflow;
    const groups = std.math.mul(usize, row_groups, col_groups) catch return error.ContextSizeOverflow;
    return std.math.mul(usize, groups, QMM_THREADS_PER_THREADGROUP) catch return error.ContextSizeOverflow;
}

pub fn qmvFastGrid(token_count: u32, out_dim: u32) !usize {
    if (token_count == 0 or out_dim == 0 or out_dim % QMV_FAST_RESULTS_PER_THREADGROUP != 0) return error.UnsupportedQuantization;
    const groups = std.math.mul(usize, @as(usize, token_count), @as(usize, out_dim / QMV_FAST_RESULTS_PER_THREADGROUP)) catch return error.ContextSizeOverflow;
    return std.math.mul(usize, groups, QMV_FAST_THREADS_PER_THREADGROUP) catch return error.ContextSizeOverflow;
}

test "qmmGrid matches tiled prefill dispatch shape" {
    try std.testing.expectEqual(@as(usize, 256), try qmmGrid(1, 1));
    try std.testing.expectEqual(@as(usize, 256), try qmmGrid(32, 32));
    try std.testing.expectEqual(@as(usize, 512), try qmmGrid(33, 33));
    try std.testing.expectEqual(@as(usize, 819_200), try qmmGrid(1600, 4096));
}

test "fast bf16 qmv constants match Kestrel decode kernel shape" {
    try std.testing.expectEqual(@as(usize, 64), QMV_FAST_THREADS_PER_THREADGROUP);
    try std.testing.expectEqual(@as(usize, 8), QMV_FAST_RESULTS_PER_THREADGROUP);
    try std.testing.expectEqual(@as(usize, 32_768), try qmvFastGrid(1, 4096));
    try std.testing.expectEqual(@as(usize, 98_304), try qmvFastGrid(1, 12288));
}

test "denseRhsBf16ByteLen matches Qwen projection sizes" {
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * BF16_BYTES), try denseRhsBf16ByteLen(4096, 4096));
    try std.testing.expectEqual(@as(usize, 12288 * 4096 * BF16_BYTES), try denseRhsBf16ByteLen(12288, 4096));
    try std.testing.expectEqual(@as(usize, 4096 * 12288 * BF16_BYTES), try denseRhsBf16ByteLen(4096, 12288));
    try std.testing.expectError(error.UnsupportedQuantization, denseRhsBf16ByteLen(4096, 63));
}

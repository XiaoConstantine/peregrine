//! Per-request model state and prefix-cache copy helpers.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const decoder_layer = @import("decoder_layer.zig");
const block_attn = @import("block_attn.zig");
const block_linear = @import("block_linear.zig");

pub const NUM_LAYERS = 32;
pub const MAX_CONTEXT_TOKENS = 64 * 1024;
pub const qwen35_9b_value_dim_major_cache_stride_alignment: usize = 256;
pub const qwen35_9b_value_dim_major_long_cache_stride_alignment: usize = 64;
pub const qwen35_9b_value_dim_major_long_cache_stride_min_sequence_length: usize = 16_000;

pub const FullCacheDType = enum {
    bf16,

    pub fn sizeInBytes(self: FullCacheDType) usize {
        _ = self;
        return 2;
    }
};

pub fn fullCacheStrideForMaxSeq(max_seq: usize) !usize {
    if (max_seq == 0 or max_seq > MAX_CONTEXT_TOKENS) return error.InvalidContextLength;
    const alignment = if (max_seq >= qwen35_9b_value_dim_major_long_cache_stride_min_sequence_length)
        qwen35_9b_value_dim_major_long_cache_stride_alignment
    else
        qwen35_9b_value_dim_major_cache_stride_alignment;
    return alignForwardUsize(max_seq, alignment);
}

fn alignForwardUsize(value: usize, alignment: usize) !usize {
    std.debug.assert(alignment > 0);
    const remainder = value % alignment;
    if (remainder == 0) return value;
    return std.math.add(usize, value, alignment - remainder);
}

/// Per-layer recurrent/KV state for the whole stack, bounded to `max_seq` tokens.
/// Full layers hold K as [NUM_KV, full_cache_stride, HEAD_DIM] and V as
/// [NUM_KV, HEAD_DIM, full_cache_stride], matching Kestrel's value-dim-major path.
/// Linear layers hold the conv ring buffer and recurrent state. Caller owns this.
pub const ModelState = struct {
    layers: [NUM_LAYERS]decoder_layer.LayerState,
    max_seq: usize,
    full_cache_stride: usize,
    full_cache_dtype: FullCacheDType,

    pub fn initBf16FullCaches(device: *metal.Device, max_seq: usize) !ModelState {
        if (max_seq == 0 or max_seq > MAX_CONTEXT_TOKENS) return error.InvalidContextLength;
        const full_cache_stride = try fullCacheStrideForMaxSeq(max_seq);
        const linear_conv_bytes = try checked_math.product(.{ block_linear.CONV_K - 1, block_linear.CONV_DIM, @sizeOf(f32) });
        const linear_recur_bytes = try checked_math.product(.{ block_linear.HV, block_linear.DV, block_linear.DK, @sizeOf(f32) });
        const full_cache_bytes = try checked_math.product(.{ block_attn.NUM_KV, full_cache_stride, block_attn.HEAD_DIM, FullCacheDType.bf16.sizeInBytes() });
        var self: ModelState = .{
            .layers = undefined,
            .max_seq = max_seq,
            .full_cache_stride = full_cache_stride,
            .full_cache_dtype = .bf16,
        };
        var made: usize = 0;
        errdefer self.deinitPartial(made);
        while (made < NUM_LAYERS) : (made += 1) {
            if (decoder_layer.isLinear(made)) {
                var conv = try device.createSharedBuffer(linear_conv_bytes);
                errdefer conv.destroy();
                @memset(conv.slice(f32), 0);
                var recur = try device.createSharedBuffer(linear_recur_bytes);
                @memset(recur.slice(f32), 0);
                self.layers[made] = .{ .linear = .{ .conv = conv, .recur = recur } };
            } else {
                var ck = try device.createSharedBuffer(full_cache_bytes);
                errdefer ck.destroy();
                @memset(ck.slice(u8), 0);
                var cv = try device.createSharedBuffer(full_cache_bytes);
                @memset(cv.slice(u8), 0);
                self.layers[made] = .{ .full = .{ .cache_k = ck, .cache_v = cv } };
            }
        }
        return self;
    }

    fn destroyOne(s: *decoder_layer.LayerState) void {
        switch (s.*) {
            .full => |*f| {
                f.cache_k.destroy();
                f.cache_v.destroy();
            },
            .linear => |*l| {
                l.conv.destroy();
                l.recur.destroy();
            },
        }
    }

    fn deinitPartial(self: *ModelState, n: usize) void {
        for (0..n) |i| destroyOne(&self.layers[i]);
    }

    pub fn deinit(self: *ModelState) void {
        for (&self.layers) |*s| destroyOne(s);
        self.* = undefined;
    }

    /// Start a new sequence without reallocating the cache buffers. Full-attn KV
    /// slots are overwritten before use because attention is bounded by seq_len;
    /// linear-attn recurrent state must be explicitly cleared.
    pub fn resetForNewSequence(self: *ModelState) void {
        for (&self.layers) |*s| switch (s.*) {
            .full => {},
            .linear => |*l| {
                @memset(l.conv.slice(f32), 0);
                @memset(l.recur.slice(f32), 0);
            },
        };
    }

    pub fn requireFullCacheDType(self: *const ModelState, dtype: FullCacheDType) !void {
        if (self.full_cache_dtype != dtype) return error.UnsupportedCacheDType;
    }

    pub fn copyPrefixFrom(self: *ModelState, source: *const ModelState, token_count: usize) !void {
        if (token_count > self.max_seq or token_count > source.max_seq) return error.SequenceTooLong;
        try self.copyFullAttentionRangeFrom(0, source, 0, token_count);
        try self.copyLinearStateFrom(source);
    }

    pub fn copyLinearStateFrom(self: *ModelState, source: *const ModelState) !void {
        for (&self.layers, &source.layers) |*dst, *src| switch (dst.*) {
            .full => if (src.* != .full) return error.LayerStateMismatch,
            .linear => |*dst_linear| {
                if (src.* != .linear) return error.LayerStateMismatch;
                std.debug.assert(dst_linear.conv.length == src.linear.conv.length);
                std.debug.assert(dst_linear.recur.length == src.linear.recur.length);
                @memcpy(dst_linear.conv.slice(f32), src.linear.conv.slice(f32));
                @memcpy(dst_linear.recur.slice(f32), src.linear.recur.slice(f32));
            },
        };
    }

    pub fn copyFullAttentionRangeFrom(
        self: *ModelState,
        dst_token_start: usize,
        source: *const ModelState,
        src_token_start: usize,
        token_count: usize,
    ) !void {
        if (self.full_cache_dtype != source.full_cache_dtype) return error.UnsupportedCacheDType;
        if (dst_token_start > self.max_seq or token_count > self.max_seq - dst_token_start) return error.SequenceTooLong;
        if (src_token_start > source.max_seq or token_count > source.max_seq - src_token_start) return error.SequenceTooLong;
        for (&self.layers, &source.layers) |*dst, *src| switch (dst.*) {
            .full => |*dst_full| {
                if (src.* != .full) return error.LayerStateMismatch;
                copyFullAttentionKeyRange(
                    dst_full.cache_k,
                    dst_token_start,
                    src.full.cache_k,
                    src_token_start,
                    self.full_cache_stride,
                    source.full_cache_stride,
                    token_count,
                    self.full_cache_dtype,
                );
                copyFullAttentionValueRange(
                    dst_full.cache_v,
                    dst_token_start,
                    src.full.cache_v,
                    src_token_start,
                    self.full_cache_stride,
                    source.full_cache_stride,
                    token_count,
                    self.full_cache_dtype,
                );
            },
            .linear => if (src.* != .linear) return error.LayerStateMismatch,
        };
    }
};

fn copyFullAttentionKeyRange(
    dst: metal.Buffer,
    dst_token_start: usize,
    src: metal.Buffer,
    src_token_start: usize,
    dst_stride: usize,
    src_stride: usize,
    token_count: usize,
    dtype: FullCacheDType,
) void {
    if (token_count == 0) return;
    const value_bytes = dtype.sizeInBytes();
    const dst_values = dst.slice(u8);
    const src_values = src.slice(u8);
    const copy_bytes = token_count * block_attn.HEAD_DIM * value_bytes;
    std.debug.assert(dst_values.len >= fullAttentionKeyEndByte(block_attn.NUM_KV - 1, dst_stride, dst_token_start + token_count, value_bytes));
    std.debug.assert(src_values.len >= fullAttentionKeyEndByte(block_attn.NUM_KV - 1, src_stride, src_token_start + token_count, value_bytes));
    for (0..block_attn.NUM_KV) |head| {
        const dst_start = (head * dst_stride + dst_token_start) * block_attn.HEAD_DIM * value_bytes;
        const src_start = (head * src_stride + src_token_start) * block_attn.HEAD_DIM * value_bytes;
        @memcpy(dst_values[dst_start..][0..copy_bytes], src_values[src_start..][0..copy_bytes]);
    }
}

fn copyFullAttentionValueRange(
    dst: metal.Buffer,
    dst_token_start: usize,
    src: metal.Buffer,
    src_token_start: usize,
    dst_stride: usize,
    src_stride: usize,
    token_count: usize,
    dtype: FullCacheDType,
) void {
    if (token_count == 0) return;
    const value_bytes = dtype.sizeInBytes();
    const dst_values = dst.slice(u8);
    const src_values = src.slice(u8);
    const copy_bytes = token_count * value_bytes;
    std.debug.assert(dst_values.len >= fullAttentionValueEndByte(block_attn.NUM_KV - 1, block_attn.HEAD_DIM - 1, dst_stride, dst_token_start + token_count, value_bytes));
    std.debug.assert(src_values.len >= fullAttentionValueEndByte(block_attn.NUM_KV - 1, block_attn.HEAD_DIM - 1, src_stride, src_token_start + token_count, value_bytes));
    for (0..block_attn.NUM_KV) |head| {
        for (0..block_attn.HEAD_DIM) |dim| {
            const dst_start = ((head * block_attn.HEAD_DIM + dim) * dst_stride + dst_token_start) * value_bytes;
            const src_start = ((head * block_attn.HEAD_DIM + dim) * src_stride + src_token_start) * value_bytes;
            @memcpy(dst_values[dst_start..][0..copy_bytes], src_values[src_start..][0..copy_bytes]);
        }
    }
}

fn fullAttentionKeyEndByte(head: usize, stride: usize, token_end: usize, value_bytes: usize) usize {
    return (((head * stride) + token_end) * block_attn.HEAD_DIM) * value_bytes;
}

fn fullAttentionValueEndByte(head: usize, dim: usize, stride: usize, token_end: usize, value_bytes: usize) usize {
    return (((head * block_attn.HEAD_DIM + dim) * stride) + token_end) * value_bytes;
}

test "full cache dtype sizes are explicit" {
    try std.testing.expectEqual(@as(usize, 2), FullCacheDType.bf16.sizeInBytes());
}

test "Qwen3.5 9B value-major full cache stride follows Kestrel alignment" {
    try std.testing.expectError(error.InvalidContextLength, fullCacheStrideForMaxSeq(0));
    try std.testing.expectEqual(@as(usize, 256), try fullCacheStrideForMaxSeq(1));
    try std.testing.expectEqual(@as(usize, 4_096), try fullCacheStrideForMaxSeq(4_073));
    try std.testing.expectEqual(@as(usize, 8_192), try fullCacheStrideForMaxSeq(8_073));
    try std.testing.expectEqual(@as(usize, 16_000), try fullCacheStrideForMaxSeq(16_000));
    try std.testing.expectEqual(@as(usize, 16_064), try fullCacheStrideForMaxSeq(16_017));
    try std.testing.expectEqual(@as(usize, 16_384), try fullCacheStrideForMaxSeq(16_384));
    try std.testing.expectEqual(@as(usize, MAX_CONTEXT_TOKENS), try fullCacheStrideForMaxSeq(MAX_CONTEXT_TOKENS));
    try std.testing.expectError(error.InvalidContextLength, fullCacheStrideForMaxSeq(MAX_CONTEXT_TOKENS + 1));
}

//! Kestrel-derived graph arena for grouped Qwen3.5-9B q4 prefill.

const std = @import("std");

const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const block_attn = @import("block_attn.zig");

pub const qwen35_9b_graph_arena_bytes: usize = 469762048;
const heap_alignment: usize = 256;
const bf16_bytes: usize = 2;

pub const LayerMajorArena = struct {
    heap: ?metal.Heap = null,

    pub fn init(device: *metal.Device, chunk_count: usize) !LayerMajorArena {
        if (chunk_count <= 1) return .{};
        return .{ .heap = try device.createPrivateHeap(qwen35_9b_graph_arena_bytes) };
    }

    pub fn deinit(self: *LayerMajorArena) void {
        if (self.heap) |*heap| heap.destroy();
        self.* = undefined;
    }

    pub fn createPrivateBuffer(self: *LayerMajorArena, device: *metal.Device, length: usize) !metal.Buffer {
        if (self.heap) |*heap| {
            if (heap.maxAvailableSize(heap_alignment) >= length) {
                return heap.createPrivateBuffer(length) catch |err| switch (err) {
                    error.BufferCreateFailed => try device.createPrivateBuffer(length),
                    else => return err,
                };
            }
        }
        return device.createPrivateBuffer(length);
    }

    pub fn createAttentionScratch(
        self: *LayerMajorArena,
        device: *metal.Device,
        token_counts: []const u32,
        token_offsets: []const u32,
        start_cache_pos: u32,
    ) !block_attn.PrefillAttentionScratch {
        if (token_counts.len == 0 or token_counts.len != token_offsets.len) return error.InvalidPrefillChunkGroup;

        var max_token_count: u32 = 0;
        var max_cache_len: u32 = 0;
        for (token_counts, token_offsets) |token_count, token_offset| {
            if (token_count == 0) return error.InvalidPrefillChunk;
            max_token_count = @max(max_token_count, token_count);
            const chunk_start = std.math.add(u32, start_cache_pos, token_offset) catch return error.SequenceTooLong;
            const chunk_end = std.math.add(u32, chunk_start, token_count) catch return error.SequenceTooLong;
            max_cache_len = @max(max_cache_len, chunk_end);
        }
        if (max_token_count == 0 or max_cache_len == 0) return error.InvalidPrefillChunk;

        const max_group_rows = std.math.mul(usize, @as(usize, max_token_count), block_attn.NUM_Q / block_attn.NUM_KV) catch return error.ContextSizeOverflow;
        const max_cache_len_usize: usize = max_cache_len;
        const score_values = std.math.mul(usize, max_group_rows, max_cache_len_usize) catch return error.ContextSizeOverflow;
        const compact_values = std.math.mul(usize, max_cache_len_usize, block_attn.HEAD_DIM) catch return error.ContextSizeOverflow;
        const group_output_values = std.math.mul(usize, max_group_rows, block_attn.HEAD_DIM) catch return error.ContextSizeOverflow;

        var score_probs = try self.createPrivateBuffer(device, try checked_math.bytes(score_values, bf16_bytes));
        errdefer score_probs.destroy();
        var compact_value = try self.createPrivateBuffer(device, try checked_math.bytes(compact_values, bf16_bytes));
        errdefer compact_value.destroy();
        const group_output = try self.createPrivateBuffer(device, try checked_math.bytes(group_output_values, bf16_bytes));

        return .{
            .score_probs = score_probs,
            .compact_value = compact_value,
            .group_output = group_output,
            .element_bytes = bf16_bytes,
        };
    }
};

test "layer-major arena only enables heap for grouped prefill" {
    var device = metal.Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return,
        else => return err,
    };
    defer device.destroy();

    var single_arena = try LayerMajorArena.init(&device, 1);
    defer single_arena.deinit();
    try std.testing.expect(single_arena.heap == null);

    var grouped_arena = try LayerMajorArena.init(&device, 2);
    defer grouped_arena.deinit();
    try std.testing.expect(grouped_arena.heap != null);
}

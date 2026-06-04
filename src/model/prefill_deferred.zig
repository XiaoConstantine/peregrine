//! Deferred layer-major prefill command-buffer cleanup helpers.
//!
//! Kestrel's serving path overlaps adjacent layer-major command buffers. Keep
//! the command-buffer wait and full-attention scratch lifetime rules in one
//! small module so the current F32 path and the future BF16 port share them.

const std = @import("std");

const metal = @import("../runtime/metal.zig");
const block_attn = @import("block_attn.zig");

const log = std.log.scoped(.peregrine_prefill);

pub fn destroyBuffers(buffers: []metal.Buffer) void {
    for (buffers) |*buffer| buffer.destroy();
}

pub fn waitSlotWithScratchProfile(
    pending: *[2]?metal.PendingCommandBuffer,
    full_attention_scratch: *[2]?block_attn.PrefillAttentionScratch,
    index: usize,
) !?u64 {
    if (pending[index]) |*command| {
        defer {
            pending[index] = null;
            if (full_attention_scratch[index]) |*scratch| {
                scratch.deinit();
                full_attention_scratch[index] = null;
            }
        }
        return try command.waitProfile();
    } else if (full_attention_scratch[index]) |*scratch| {
        scratch.deinit();
        full_attention_scratch[index] = null;
    }
    return null;
}

pub fn drainAllWithScratch(
    pending: *[2]?metal.PendingCommandBuffer,
    full_attention_scratch: *[2]?block_attn.PrefillAttentionScratch,
) void {
    for (0..pending.len) |index| {
        if (pending[index]) |*command| {
            command.wait() catch |err| {
                log.warn("failed to drain pending prefill command buffer: {s}", .{@errorName(err)});
            };
            pending[index] = null;
        }
        if (full_attention_scratch[index]) |*scratch| {
            scratch.deinit();
            full_attention_scratch[index] = null;
        }
    }
}

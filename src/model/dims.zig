const std = @import("std");

pub const hidden: u32 = 4096;
pub const bf16_bytes: usize = 2;
pub const rmsnorm_eps: f32 = 1e-6;
pub const bf16_rmsnorm_threads: u32 = 512;
pub const bf16_add_rmsnorm_threads: u32 = 512;

/// Byte length of `token_count` rows of `[HIDDEN]` bf16 hidden states.
/// Shared by the prefill capture path, the prefix cache, persistence, the MTP
/// drafter, and the bench tool so they all agree on the layout.
pub fn hiddenRowsBytes(token_count: usize) !usize {
    return std.math.mul(usize, token_count, @as(usize, hidden) * bf16_bytes) catch
        return error.ContextSizeOverflow;
}

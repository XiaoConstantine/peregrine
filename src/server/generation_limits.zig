//! Helpers for fitting OpenAI-style generation requests into the local context.

const std = @import("std");

pub fn effectiveMaxNewTokens(prompt_tokens: usize, requested_max_new: u32, max_total_tokens: usize) !u32 {
    if (prompt_tokens == 0 or requested_max_new == 0) return error.EmptyInput;
    if (prompt_tokens >= max_total_tokens) return error.SequenceTooLong;
    const available = max_total_tokens - prompt_tokens;
    const clamped = @min(@as(usize, requested_max_new), available);
    return std.math.cast(u32, clamped) orelse error.SequenceTooLong;
}

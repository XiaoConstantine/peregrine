//! Small checked arithmetic helpers for byte and element-count calculations.

const std = @import("std");

pub fn product(factors: anytype) !usize {
    var total: usize = 1;
    inline for (factors) |factor| {
        total = std.math.mul(usize, total, @as(usize, factor)) catch return error.ContextSizeOverflow;
    }
    return total;
}

pub fn bytes(value_count: usize, element_bytes: usize) !usize {
    return product(.{ value_count, element_bytes });
}

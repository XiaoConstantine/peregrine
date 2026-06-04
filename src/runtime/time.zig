const std = @import("std");

pub fn monotonicNowNs() u64 {
    const posix = std.posix;
    var ts: posix.timespec = undefined;
    return switch (posix.errno(posix.system.clock_gettime(posix.CLOCK.MONOTONIC, &ts))) {
        .SUCCESS => blk: {
            std.debug.assert(ts.sec >= 0 and ts.nsec >= 0);
            break :blk @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        else => @panic("clock_gettime(CLOCK_MONOTONIC) failed"),
    };
}

pub fn msFromNs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

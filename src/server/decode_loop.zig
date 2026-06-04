//! One-token server decode loop for the prefix-cache serving path.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const runtime_time = @import("../runtime/time.zig");
const model_mod = @import("../model/model.zig");
const stop_sequences = @import("../stop_sequences.zig");

const DeviceModel = model_mod.DeviceModel;
const ModelState = model_mod.ModelState;
const monotonicNowNs = runtime_time.monotonicNowNs;
const msFromNs = runtime_time.msFromNs;
const log = std.log.scoped(.peregrine_decode);

pub const Args = struct {
    device: *metal.Device,
    queue: *metal.Queue,
    model: *DeviceModel,
    eos_id: u32,
    prompt_len: usize,
    prefix_len: usize,
    state: *ModelState,
    prefix: ?DeviceModel.PrefixState,
    initial_next_token: u32,
    next_token: metal.Buffer,
    out: []u32,
    callback: ?DeviceModel.GeneratedTokenCallback,
    stop_token_sequences: []const []const u32,
    trace: bool = false,
};

pub fn decodeFromPrefixCache(args: Args) !usize {
    std.debug.assert(args.prefix_len <= args.prompt_len);
    var current_token = args.initial_next_token;
    var n: usize = 0;
    while (n < args.out.len) {
        const next = current_token;
        if (next == args.eos_id) break;
        if (args.trace) {
            log.info("trace: token index={d} token_id={d}", .{ n, next });
        }
        args.out[n] = next;
        n += 1;
        if (args.callback) |cb| {
            if (args.trace) {
                log.info("trace: emit start generated_tokens={d}", .{n});
            }
            try cb.emit(cb.context, args.out[0..n]);
            if (args.trace) {
                log.info("trace: emit done generated_tokens={d}", .{n});
            }
        }
        if (stop_sequences.generatedEndsWith(args.out[0..n], args.stop_token_sequences)) break;

        if (n < args.out.len) {
            const logical_pos = args.prompt_len + n - 1;
            const cache_pos: u32 = @intCast(logical_pos - args.prefix_len);
            if (args.trace) {
                const wall_start = monotonicNowNs();
                const gpu_ns = try args.model.forwardNextTokenBf16WithPrefixProfiled(
                    args.device,
                    args.queue,
                    next,
                    cache_pos,
                    @intCast(logical_pos),
                    @intCast(logical_pos + 1),
                    args.state,
                    args.prefix,
                    args.next_token,
                );
                const wall_ns = monotonicNowNs() - wall_start;
                const wall_ms = msFromNs(wall_ns);
                if (gpu_ns) |g| {
                    const gpu_ms = msFromNs(g);
                    log.info(
                        "trace: forward wall_ms={d:.3} gpu_ms={d:.3}",
                        .{ wall_ms, gpu_ms },
                    );
                } else {
                    log.info(
                        "trace: forward wall_ms={d:.3} gpu_ms=null",
                        .{wall_ms},
                    );
                }
            } else {
                try args.model.forwardNextTokenBf16WithPrefix(
                    args.device,
                    args.queue,
                    next,
                    cache_pos,
                    @intCast(logical_pos),
                    @intCast(logical_pos + 1),
                    args.state,
                    args.prefix,
                    args.next_token,
                );
            }
            current_token = args.next_token.slice(u32)[0];
        }
    }
    if (args.trace) {
        log.info("trace: done generated_tokens={d}", .{n});
    }
    return n;
}

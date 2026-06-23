//! One-token server decode loop for the prefix-cache serving path.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const runtime_time = @import("../runtime/time.zig");
const model_mod = @import("../model/model.zig");
const mtp_mod = @import("../model/mtp.zig");
const stop_sequences = @import("../stop_sequences.zig");

const DeviceModel = model_mod.DeviceModel;
const ModelState = model_mod.ModelState;
const monotonicNowNs = runtime_time.monotonicNowNs;
const msFromNs = runtime_time.msFromNs;
const log = std.log.scoped(.peregrine_decode);

pub const Metrics = struct {
    request_start_ns: u64,
    decode_start_ns: u64 = 0,
    first_emit_ns: ?u64 = null,
    last_emit_ns: ?u64 = null,
    emitted_tokens: usize = 0,
    decode_steps: usize = 0,
    accepted: usize = 0,
    attempted: usize = 0,

    pub fn recordEmit(self: *Metrics, generated_tokens: usize) void {
        const now = monotonicNowNs();
        if (self.first_emit_ns == null) self.first_emit_ns = now;
        self.last_emit_ns = now;
        self.emitted_tokens = generated_tokens;
    }
};

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
    metrics: ?*Metrics = null,
};

pub const SpeculativeArgs = struct {
    device: *metal.Device,
    queue: *metal.Queue,
    model: *DeviceModel,
    drafter: *mtp_mod.Drafter,
    eos_id: u32,
    prompt_len: usize,
    prefix_len: usize,
    state: *ModelState,
    verifier_state: *ModelState,
    mtp_state: *mtp_mod.State,
    prefix: ?DeviceModel.PrefixState,
    initial_next_token: u32,
    initial_hidden_rows: metal.Buffer,
    initial_hidden_row: u32,
    next_token: metal.Buffer,
    draft_token: metal.Buffer,
    verify_token: metal.Buffer,
    bonus_token: metal.Buffer,
    advance_token: metal.Buffer,
    verifier_hidden: metal.Buffer,
    single_hidden: metal.Buffer,
    out: []u32,
    callback: ?DeviceModel.GeneratedTokenCallback,
    stop_token_sequences: []const []const u32,
    trace: bool = false,
    metrics: ?*Metrics = null,
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
        try emitGenerated(args.callback, args.out, &n, next, args.trace, args.metrics);
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
                if (args.metrics) |metrics| metrics.decode_steps += 1;
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
                if (args.metrics) |metrics| metrics.decode_steps += 1;
            }
            current_token = args.next_token.slice(u32)[0];
        }
    }
    if (args.trace) {
        log.info("trace: done generated_tokens={d}", .{n});
    }
    return n;
}

pub fn decodeSpeculativeFromPrefixCache(args: SpeculativeArgs) !usize {
    std.debug.assert(args.prefix_len <= args.prompt_len);
    var current_token = args.initial_next_token;
    var current_hidden_rows = args.initial_hidden_rows;
    var current_hidden_row = args.initial_hidden_row;
    var n: usize = 0;
    var accepted: usize = 0;
    var attempted: usize = 0;

    while (n < args.out.len) {
        const next = current_token;
        if (next == args.eos_id) break;
        if (args.trace) {
            log.info("trace: mtp token index={d} token_id={d}", .{ n, next });
        }
        try emitGenerated(args.callback, args.out, &n, next, args.trace, args.metrics);
        if (stop_sequences.generatedEndsWith(args.out[0..n], args.stop_token_sequences)) break;
        if (n >= args.out.len) break;

        const logical_pos = args.prompt_len + n - 1;
        const cache_pos: u32 = @intCast(logical_pos - args.prefix_len);
        const mtp_cache_pos: u32 = @intCast(logical_pos - 1);
        _ = try args.drafter.forwardDraftTokenProfiled(
            args.device,
            args.queue,
            args.model,
            next,
            current_hidden_rows,
            current_hidden_row,
            args.mtp_state,
            mtp_cache_pos,
            mtp_cache_pos,
            mtp_cache_pos + 1,
            args.draft_token,
        );

        const drafted = args.draft_token.slice(u32)[0];
        _ = try args.model.forwardTwoTokenDecodeVerifierBf16WithPrefixProfiled(
            args.device,
            args.queue,
            .{ next, drafted },
            cache_pos,
            @intCast(logical_pos),
            @intCast(logical_pos + 1),
            args.verifier_state,
            args.prefix,
            args.verify_token,
            args.bonus_token,
            args.verifier_hidden,
        );
        attempted += 1;

        const verified = args.verify_token.slice(u32)[0];
        if (drafted == verified) {
            accepted += 1;
            std.mem.swap(ModelState, args.state, args.verifier_state);

            if (drafted == args.eos_id) break;
            try emitGenerated(args.callback, args.out, &n, drafted, args.trace, args.metrics);
            if (stop_sequences.generatedEndsWith(args.out[0..n], args.stop_token_sequences)) break;
            if (n >= args.out.len) break;

            _ = try args.drafter.forwardDraftTokenProfiled(
                args.device,
                args.queue,
                args.model,
                drafted,
                args.verifier_hidden,
                0,
                args.mtp_state,
                mtp_cache_pos + 1,
                mtp_cache_pos + 1,
                mtp_cache_pos + 2,
                args.advance_token,
            );
            try prepareVerifierState(args.verifier_state, args.state, cache_pos, 2, true);
            current_token = args.bonus_token.slice(u32)[0];
            current_hidden_rows = args.verifier_hidden;
            current_hidden_row = 1;
        } else {
            _ = try args.model.forwardNextTokenBf16WithPrefixAndHiddenProfiled(
                args.device,
                args.queue,
                next,
                cache_pos,
                @intCast(logical_pos),
                @intCast(logical_pos + 1),
                args.state,
                args.prefix,
                args.next_token,
                args.single_hidden,
            );
            try prepareVerifierState(args.verifier_state, args.state, cache_pos, 1, true);
            current_token = args.next_token.slice(u32)[0];
            current_hidden_rows = args.single_hidden;
            current_hidden_row = 0;
        }
    }

    if (args.trace) {
        log.info("trace: mtp done generated_tokens={d} accepted={d} attempted={d}", .{ n, accepted, attempted });
    }
    if (args.metrics) |metrics| {
        metrics.decode_steps = attempted;
        metrics.accepted = accepted;
        metrics.attempted = attempted;
    }
    return n;
}

fn emitGenerated(
    callback: ?DeviceModel.GeneratedTokenCallback,
    out: []u32,
    n: *usize,
    token: u32,
    trace: bool,
    metrics: ?*Metrics,
) !void {
    out[n.*] = token;
    n.* += 1;
    if (metrics) |m| m.recordEmit(n.*);
    if (callback) |cb| {
        if (trace) {
            log.info("trace: emit start generated_tokens={d}", .{n.*});
        }
        try cb.emit(cb.context, out[0..n.*]);
        if (trace) {
            log.info("trace: emit done generated_tokens={d}", .{n.*});
        }
    }
}

fn prepareVerifierState(
    verifier_state: *ModelState,
    committed_state: *const ModelState,
    full_attention_start: u32,
    full_attention_count: u32,
    copy_full_attention: bool,
) !void {
    if (copy_full_attention and full_attention_count != 0) {
        try verifier_state.copyFullAttentionRangeFrom(
            full_attention_start,
            committed_state,
            full_attention_start,
            full_attention_count,
        );
    }
    try verifier_state.copyLinearStateFrom(committed_state);
}

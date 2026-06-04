//! Prompt prefill backends for Qwen3.5-9B q4.
//!
//! The server uses the Kestrel-derived batched chunk path.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const runtime_time = @import("../runtime/time.zig");
const model_mod = @import("model.zig");

const DeviceModel = model_mod.DeviceModel;
const ModelState = model_mod.ModelState;
const PrefillNextTokenOutput = model_mod.PrefillNextTokenOutput;
const monotonicNowNs = runtime_time.monotonicNowNs;
const msFromNs = runtime_time.msFromNs;
const log = std.log.scoped(.peregrine_prefill);

pub const SegmentOptions = struct {
    prefix_len: usize,
    start_pos: usize,
    end_pos: usize,
    chunk_tokens: usize = model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS,
    chunk_group_size: usize = model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE,
    trace: bool = false,
    callback: ?DeviceModel.GeneratedTokenCallback = null,
    write_next_token: bool = true,
};

/// Prefill `prompt_ids[start_pos..end_pos]` and leave `next_token_output` holding
/// the argmax token after the segment's final token. Earlier rows skip lm_head work.
pub fn prefillSegment(
    model: *DeviceModel,
    device: *metal.Device,
    queue: *metal.Queue,
    prompt_ids: []const u32,
    state: *ModelState,
    prefix: ?DeviceModel.PrefixState,
    next_token_output: PrefillNextTokenOutput,
    options: SegmentOptions,
) !void {
    if (options.start_pos > options.end_pos or options.end_pos > prompt_ids.len) return error.InvalidPrefillSegment;
    if (options.start_pos < options.prefix_len) return error.InvalidPrefillSegment;
    if (options.chunk_tokens == 0 or options.chunk_tokens > model_mod.MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.InvalidPrefillChunk;
    if (options.chunk_group_size == 0 or options.chunk_group_size > model_mod.MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE) return error.InvalidPrefillChunkGroup;
    if (options.start_pos == options.end_pos) return;

    const prepare_policy_token_count = prompt_ids.len - options.prefix_len;
    var pos = options.start_pos;
    if (!options.write_next_token) {
        while (pos < options.end_pos) {
            const chunk_end = nextCacheOnlyGroupEnd(pos, options.end_pos, options.chunk_tokens, options.chunk_group_size);
            if (options.trace) {
                log.info(
                    "trace: batched cache group pos={d}..{d}/{d} prefix_len={d} chunk_tokens={d}",
                    .{ pos, chunk_end, prompt_ids.len, options.prefix_len, options.chunk_tokens },
                );
            }
            const start_ns = traceStartNs(options.trace);
            try forwardGroup(model, device, queue, prompt_ids, state, prefix, pos, chunk_end, prepare_policy_token_count, options, null);
            traceDone(options.trace, "prefill batched cache group", pos, chunk_end, start_ns);
            pos = chunk_end;
            try reportProgress(options.callback, pos, prompt_ids.len);
        }
        return;
    }

    while (pos < options.end_pos) {
        if (options.end_pos - pos <= options.chunk_tokens) break;
        const chunk_end = nextOutputGroupEnd(pos, options.end_pos, options.chunk_tokens, options.chunk_group_size);
        const writes_next_token = chunk_end == options.end_pos;
        if (options.trace) {
            log.info(
                "trace: batched group pos={d}..{d}/{d} prefix_len={d} chunk_tokens={d} next_token={}",
                .{ pos, chunk_end, prompt_ids.len, options.prefix_len, options.chunk_tokens, writes_next_token },
            );
        }
        const start_ns = traceStartNs(options.trace);
        try forwardGroup(model, device, queue, prompt_ids, state, prefix, pos, chunk_end, prepare_policy_token_count, options, if (writes_next_token) next_token_output else null);
        traceDone(options.trace, "prefill batched group", pos, chunk_end, start_ns);
        pos = chunk_end;
        try reportProgress(options.callback, pos, prompt_ids.len);
        if (writes_next_token) return;
    }

    const final_pos = options.end_pos - 1;
    const final_chunk = prompt_ids[pos..options.end_pos];
    if (options.trace) {
        log.info(
            "trace: batched next-token chunk pos={d}..{d}/{d} final_pos={d} prefix_len={d}",
            .{ pos, options.end_pos, prompt_ids.len, final_pos, options.prefix_len },
        );
    }
    const start_ns = traceStartNs(options.trace);
    if (finalOutputChunkUsesLayerMajorGroup(prefix != null, pos, options.prefix_len)) {
        try forwardGroup(model, device, queue, prompt_ids, state, prefix, pos, options.end_pos, prepare_policy_token_count, options, next_token_output);
    } else {
        try model.forwardPrefillBatchNextTokenWithPrefix(
            device,
            queue,
            final_chunk,
            try u32Cast(pos - options.prefix_len),
            try u32Cast(pos),
            state,
            prefix,
            next_token_output,
        );
    }
    traceDone(options.trace, "prefill batched next-token chunk", pos, options.end_pos, start_ns);
    try reportProgress(options.callback, options.end_pos, prompt_ids.len);
}

fn forwardGroup(
    model: *DeviceModel,
    device: *metal.Device,
    queue: *metal.Queue,
    prompt_ids: []const u32,
    state: *ModelState,
    prefix: ?DeviceModel.PrefixState,
    pos: usize,
    chunk_end: usize,
    prepare_policy_token_count: usize,
    options: SegmentOptions,
    next_token_output: ?PrefillNextTokenOutput,
) !void {
    try model.forwardPrefillBatchGroupWithPrefix(
        device,
        queue,
        prompt_ids[pos..chunk_end],
        try u32Cast(pos - options.prefix_len),
        try u32Cast(pos),
        options.chunk_tokens,
        prepare_policy_token_count,
        state,
        prefix,
        options.trace,
        next_token_output,
    );
}

fn finalOutputChunkUsesLayerMajorGroup(has_prefix: bool, pos: usize, prefix_len: usize) bool {
    return !has_prefix and pos > prefix_len;
}

fn nextOutputGroupEnd(pos: usize, end_pos: usize, chunk_tokens: usize, chunk_group_size: usize) usize {
    std.debug.assert(pos + chunk_tokens < end_pos);
    const remaining = end_pos - pos;
    const group_tokens = chunk_tokens * chunk_group_size;
    if (remaining <= group_tokens) return end_pos;
    return pos + group_tokens;
}

fn nextCacheOnlyGroupEnd(pos: usize, end_pos: usize, chunk_tokens: usize, chunk_group_size: usize) usize {
    std.debug.assert(pos < end_pos);
    std.debug.assert(chunk_tokens != 0);
    std.debug.assert(chunk_group_size != 0);
    const remaining = end_pos - pos;
    if (remaining <= chunk_tokens) return end_pos;
    const group_tokens = chunk_tokens * chunk_group_size;
    if (remaining < group_tokens) return pos + chunk_tokens;
    return pos + group_tokens;
}

fn traceStartNs(enabled: bool) u64 {
    return if (enabled) monotonicNowNs() else 0;
}

fn traceDone(enabled: bool, label: []const u8, start_pos: usize, end_pos: usize, start_ns: u64) void {
    if (!enabled) return;
    const elapsed_ns = monotonicNowNs() - start_ns;
    log.info(
        "trace: {s} done pos={d}..{d} elapsed_ms={d:.3}",
        .{ label, start_pos, end_pos, msFromNs(elapsed_ns) },
    );
}

fn reportProgress(callback: ?DeviceModel.GeneratedTokenCallback, done: usize, total: usize) !void {
    if (callback) |cb| {
        if (cb.progress) |progress| try progress(cb.context, done, total);
    }
}

fn u32Cast(value: usize) !u32 {
    return std.math.cast(u32, value) orelse error.SequenceTooLong;
}

test "nextOutputGroupEnd fuses final multi-chunk group" {
    try std.testing.expectEqual(@as(usize, 10), nextOutputGroupEnd(0, 10, 4, 4));
    try std.testing.expectEqual(@as(usize, 12), nextOutputGroupEnd(0, 12, 4, 4));
    try std.testing.expectEqual(@as(usize, 16), nextOutputGroupEnd(0, 24, 4, 4));
    try std.testing.expectEqual(@as(usize, 20), nextOutputGroupEnd(8, 20, 4, 4));
}

test "nextCacheOnlyGroupEnd avoids grouping partial tail" {
    try std.testing.expectEqual(@as(usize, 16), nextCacheOnlyGroupEnd(0, 20, 4, 4));
    try std.testing.expectEqual(@as(usize, 20), nextCacheOnlyGroupEnd(16, 20, 4, 4));
    try std.testing.expectEqual(@as(usize, 4), nextCacheOnlyGroupEnd(0, 14, 4, 4));
    try std.testing.expectEqual(@as(usize, 8), nextCacheOnlyGroupEnd(4, 14, 4, 4));
    try std.testing.expectEqual(@as(usize, 12), nextCacheOnlyGroupEnd(8, 14, 4, 4));
    try std.testing.expectEqual(@as(usize, 14), nextCacheOnlyGroupEnd(12, 14, 4, 4));
}

test "final output tail keeps raw suffix on layer-major group path" {
    try std.testing.expect(!finalOutputChunkUsesLayerMajorGroup(false, 0, 0));
    try std.testing.expect(finalOutputChunkUsesLayerMajorGroup(false, 16000, 0));
    try std.testing.expect(!finalOutputChunkUsesLayerMajorGroup(true, 16000, 0));
    try std.testing.expect(!finalOutputChunkUsesLayerMajorGroup(false, 16000, 16000));
}

test "captured Pi raw prompt keeps Kestrel layer-major prefix split" {
    const prompt_tokens: usize = 16_153;
    const first_group_end = nextOutputGroupEnd(
        0,
        prompt_tokens,
        model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS,
        model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE,
    );

    try std.testing.expectEqual(@as(usize, 16_000), first_group_end);
    try std.testing.expectEqual(@as(usize, 153), prompt_tokens - first_group_end);
    try std.testing.expect(finalOutputChunkUsesLayerMajorGroup(false, first_group_end, 0));
}

test "short no-prefix prompts keep the direct single-chunk output path" {
    const kestrel_short_request_tokens: usize = 512;
    try std.testing.expect(kestrel_short_request_tokens <= model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS);
    try std.testing.expect(!finalOutputChunkUsesLayerMajorGroup(false, 0, 0));
}

test "serving prefill defaults stay on the Kestrel Qwen3.5 9B route" {
    try std.testing.expectEqual(@as(usize, 1600), model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS);
    try std.testing.expectEqual(
        model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS,
        model_mod.MAX_BATCHED_PREFILL_CHUNK_TOKENS,
    );
    try std.testing.expectEqual(@as(usize, 10), model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE);
    try std.testing.expectEqual(
        model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE,
        model_mod.MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE,
    );
}

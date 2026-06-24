//! Structured in-process inference metrics for the dashboard.
//!
//! Cumulative process-local totals are kept as atomics so dashboard polling
//! never contends with the generation path. A small bounded ring buffer of
//! recent completed-request records is guarded by a separate mutex so charts
//! and request logs can show per-request trends without holding the decode
//! lock.
//!
//! Only numeric counts, timings, cache counters, and MTP counters are stored.
//! Prompts, completions, tool arguments, and decoded text are never recorded.

const std = @import("std");
const Io = std.Io;
const runtime_time = @import("../runtime/time.zig");

const monotonicNowNs = runtime_time.monotonicNowNs;
const unixNowMs = runtime_time.unixNowMs;

pub const HISTORY_CAPACITY: usize = 256;

/// Decode mode for a request, mirroring the trace `mode` field.
pub const DecodeMode = enum {
    mtp,
    greedy,

    pub fn jsonStringValue(self: DecodeMode, writer: anytype) !void {
        try writer.print("\"{s}\"", .{@tagName(self)});
    }
};

/// Failure category for a request. `none` means the request completed normally.
pub const ErrorKind = enum {
    none,
    validation,
    generation,
    client_disconnect,
    internal,

    pub fn jsonStringValue(self: ErrorKind, writer: anytype) !void {
        try writer.print("\"{s}\"", .{@tagName(self)});
    }
};

/// Fixed-size record for one completed (or failed) generation attempt. Stored
/// verbatim in the ring; no prompt/completion text is retained.
pub const RequestMetricsRecord = struct {
    /// Monotonic process-local id assigned at generation-path entry. Starts at
    /// 1. Not stable across restarts.
    request_id: u64,

    /// Monotonic timestamps (runtime_time.monotonicNowNs) used for internal
    /// duration math. Zero when the request failed before prefill timing began.
    started_mono_ns: u64,
    completed_mono_ns: u64,

    /// Wall-clock timestamps for dashboard labels, in milliseconds since the
    /// Unix epoch. Milliseconds avoid the JavaScript number-precision problems
    /// of nanosecond epoch values.
    started_unix_ms: i64,
    completed_unix_ms: i64,

    ok: bool,
    error_kind: ErrorKind,
    mode: DecodeMode,

    prompt_tokens: u32,
    max_new_tokens: u32,
    generated_tokens: u32,

    /// Prompt tokens reused from the prefix cache for this request.
    reuse_tokens: u32,
    /// Prompt tokens computed fresh for this request. For a direct fill of an
    /// empty cache, the full prompt is counted as computed even though it
    /// becomes a future cache hit.
    computed_tokens: u32,

    prefill_ms: f64,
    setup_ms: f64,
    /// Time to first emitted token. Approximately prefill_ms + setup_ms when
    /// the decode path reported a first-emit timestamp; zero otherwise.
    ttft_ms: f64,
    decode_ms: f64,
    tok_s: f64,
    itl_ms: f64,
    /// Throughput over the first-to-last emitted token span. Kept because it
    /// can differ from decode_ms-based tok_s.
    emit_tok_s: f64,

    decode_steps: u32,
    tokens_per_step: f64,
    accepted: u32,
    attempted: u32,
    /// accepted / attempted for MTP. Zero when attempted == 0; rendered as
    /// `null` in JSON when attempted == 0.
    acceptance: f64,

    /// Nullable in JSON. Recorded separately from setup_ms because it isolates
    /// MTP drafter-prefill cost from verifier/setup cost. Null when no MTP
    /// drafter seed was measured.
    drafter_seed_ms: ?f64,

    /// Cache occupancy snapshot taken at request completion while the decode
    /// lock is still held and prefix-cache access is safe.
    cache_cached_tokens: u32,
    cache_max_cache_tokens: u32,
    cache_resident_entries: u32,
    cache_reserved_tokens: u32,
    cache_resident_tokens: u32,
    cache_pinned_entries: u32,
    cache_cached_logits: bool,
    /// Cumulative process-local evictions as of request completion.
    cache_evictions_total: u64,
};

/// In-flight request handle. Created at generation-path entry and consumed at
/// completion/failure to populate a `RequestMetricsRecord` and append it.
///
/// Timing fields (`prefill_ms`, `setup_ms`, `ttft_ms`, `decode_ms`, ...) are
/// filled in by the server as the request progresses; the handle carries the
/// start timestamps and identity. Use `setDecodeMetrics` once the decode loop
/// returns, then `finish` (or `finishError`) to append the record.
pub const Request = struct {
    metrics: *MetricsState,
    record: RequestMetricsRecord,

    pub fn setMode(self: *Request, mode: DecodeMode) void {
        self.record.mode = mode;
    }

    /// Populate the timing/throughput/MTP fields from values computed by the
    /// server's decode-metrics path. `drafter_seed_ms` is nullable.
    pub fn setDecodeMetrics(
        self: *Request,
        prefill_ms: f64,
        setup_ms: f64,
        ttft_ms: f64,
        decode_ms: f64,
        tok_s: f64,
        itl_ms: f64,
        emit_tok_s: f64,
        decode_steps: u32,
        tokens_per_step: f64,
        accepted: u32,
        attempted: u32,
        acceptance: f64,
        drafter_seed_ms: ?f64,
        generated_tokens: u32,
    ) void {
        self.record.prefill_ms = prefill_ms;
        self.record.setup_ms = setup_ms;
        self.record.ttft_ms = ttft_ms;
        self.record.decode_ms = decode_ms;
        self.record.tok_s = tok_s;
        self.record.itl_ms = itl_ms;
        self.record.emit_tok_s = emit_tok_s;
        self.record.decode_steps = decode_steps;
        self.record.tokens_per_step = tokens_per_step;
        self.record.accepted = accepted;
        self.record.attempted = attempted;
        self.record.acceptance = acceptance;
        self.record.drafter_seed_ms = drafter_seed_ms;
        self.record.generated_tokens = generated_tokens;
    }

    /// Record the prompt-side fields known after prefix resolution.
    pub fn setPromptStats(
        self: *Request,
        prompt_tokens: u32,
        max_new_tokens: u32,
        reuse_tokens: u32,
        computed_tokens: u32,
    ) void {
        self.record.prompt_tokens = prompt_tokens;
        self.record.max_new_tokens = max_new_tokens;
        self.record.reuse_tokens = reuse_tokens;
        self.record.computed_tokens = computed_tokens;
    }

    /// Record a cache-occupancy snapshot taken while the decode lock is held.
    pub fn setCacheSnapshot(self: *Request, snap: CacheSnapshot) void {
        self.record.cache_cached_tokens = snap.cached_tokens;
        self.record.cache_max_cache_tokens = snap.max_cache_tokens;
        self.record.cache_resident_entries = snap.resident_entries;
        self.record.cache_reserved_tokens = snap.reserved_tokens;
        self.record.cache_resident_tokens = snap.resident_tokens;
        self.record.cache_pinned_entries = snap.pinned_entries;
        self.record.cache_cached_logits = snap.cached_logits;
        self.record.cache_evictions_total = snap.evictions_total;
    }

    /// Append the record as successful. Consumes the request.
    pub fn finish(self: *Request) void {
        self.record.ok = true;
        self.record.error_kind = .none;
        self.record.completed_mono_ns = monotonicNowNs();
        self.record.completed_unix_ms = unixNowMs();
        self.metrics.appendRecord(self.record);
    }

    /// Append the record as failed with the given category. Consumes the
    /// request. Safe to call before `setDecodeMetrics` (missing timing fields
    /// stay zero and render as zero / null in JSON).
    pub fn finishError(self: *Request, kind: ErrorKind) void {
        self.record.ok = false;
        self.record.error_kind = kind;
        self.record.completed_mono_ns = monotonicNowNs();
        self.record.completed_unix_ms = unixNowMs();
        self.metrics.appendRecord(self.record);
    }
};

/// Cache occupancy snapshot copied out while the decode lock is held.
pub const CacheSnapshot = struct {
    cached_tokens: u32,
    max_cache_tokens: u32,
    resident_entries: u32,
    reserved_tokens: u32,
    resident_tokens: u32,
    pinned_entries: u32,
    cached_logits: bool,
    evictions_total: u64,
};

pub const MetricsState = struct {
    io: Io,
    next_request_id: std.atomic.Value(u64) = .init(1),

    requests_total: std.atomic.Value(u64) = .init(0),
    requests_failed: std.atomic.Value(u64) = .init(0),
    prompt_tokens_total: std.atomic.Value(u64) = .init(0),
    generated_tokens_total: std.atomic.Value(u64) = .init(0),
    reuse_tokens_total: std.atomic.Value(u64) = .init(0),
    computed_tokens_total: std.atomic.Value(u64) = .init(0),

    history_lock: Io.Mutex = .init,
    history: [HISTORY_CAPACITY]RequestMetricsRecord = undefined,
    /// Index of the next write slot. Used modulo HISTORY_CAPACITY.
    history_head: usize = 0,
    /// Number of valid records stored (caps at HISTORY_CAPACITY).
    history_count: usize = 0,

    pub fn init(io: Io) MetricsState {
        return .{ .io = io };
    }

    /// Allocate a request id and capture start timestamps. Called when a
    /// generation attempt reaches the model path.
    pub fn beginRequest(self: *MetricsState, max_new_tokens: u32) Request {
        const id = self.next_request_id.fetchAdd(1, .monotonic);
        const started_mono = monotonicNowNs();
        const started_unix = unixNowMs();
        _ = self.requests_total.fetchAdd(1, .monotonic);
        return .{
            .metrics = self,
            .record = .{
                .request_id = id,
                .started_mono_ns = started_mono,
                .completed_mono_ns = 0,
                .started_unix_ms = started_unix,
                .completed_unix_ms = 0,
                .ok = false,
                .error_kind = .none,
                .mode = .greedy,
                .prompt_tokens = 0,
                .max_new_tokens = max_new_tokens,
                .generated_tokens = 0,
                .reuse_tokens = 0,
                .computed_tokens = 0,
                .prefill_ms = 0,
                .setup_ms = 0,
                .ttft_ms = 0,
                .decode_ms = 0,
                .tok_s = 0,
                .itl_ms = 0,
                .emit_tok_s = 0,
                .decode_steps = 0,
                .tokens_per_step = 0,
                .accepted = 0,
                .attempted = 0,
                .acceptance = 0,
                .drafter_seed_ms = null,
                .cache_cached_tokens = 0,
                .cache_max_cache_tokens = 0,
                .cache_resident_entries = 0,
                .cache_reserved_tokens = 0,
                .cache_resident_tokens = 0,
                .cache_pinned_entries = 0,
                .cache_cached_logits = false,
                .cache_evictions_total = 0,
            },
        };
    }

    /// Append a completed record to the ring and bump cumulative totals.
    /// O(1); no JSON formatting or allocation. Called by `Request.finish` and
    /// `Request.finishError`.
    fn appendRecord(self: *MetricsState, record: RequestMetricsRecord) void {
        if (!record.ok) _ = self.requests_failed.fetchAdd(1, .monotonic);
        _ = self.prompt_tokens_total.fetchAdd(record.prompt_tokens, .monotonic);
        _ = self.generated_tokens_total.fetchAdd(record.generated_tokens, .monotonic);
        _ = self.reuse_tokens_total.fetchAdd(record.reuse_tokens, .monotonic);
        _ = self.computed_tokens_total.fetchAdd(record.computed_tokens, .monotonic);

        self.history_lock.lockUncancelable(self.io);
        defer self.history_lock.unlock(self.io);
        self.history[self.history_head] = record;
        self.history_head = (self.history_head + 1) % HISTORY_CAPACITY;
        if (self.history_count < HISTORY_CAPACITY) self.history_count += 1;
    }

    /// Copy the current history into a caller-provided slice (oldest-first)
    /// and return the number of records copied. The lock is held only long
    /// enough to copy; the caller renders JSON after release.
    pub fn snapshotHistory(self: *MetricsState, out: []RequestMetricsRecord) usize {
        self.history_lock.lockUncancelable(self.io);
        defer self.history_lock.unlock(self.io);
        const n = @min(out.len, self.history_count);
        if (n == 0) return 0;
        // The ring is full: oldest is at history_head (the next overwrite
        // slot). Otherwise oldest is at index 0.
        const start: usize = if (self.history_count == HISTORY_CAPACITY)
            self.history_head
        else
            0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            out[i] = self.history[(start + i) % HISTORY_CAPACITY];
        }
        return n;
    }

    /// Cumulative counters as a value struct (atomic loads).
    pub const Totals = struct {
        requests_total: u64,
        requests_failed: u64,
        prompt_tokens_total: u64,
        generated_tokens_total: u64,
        reuse_tokens_total: u64,
        computed_tokens_total: u64,
    };

    pub fn totals(self: *const MetricsState) Totals {
        return .{
            .requests_total = self.requests_total.load(.monotonic),
            .requests_failed = self.requests_failed.load(.monotonic),
            .prompt_tokens_total = self.prompt_tokens_total.load(.monotonic),
            .generated_tokens_total = self.generated_tokens_total.load(.monotonic),
            .reuse_tokens_total = self.reuse_tokens_total.load(.monotonic),
            .computed_tokens_total = self.computed_tokens_total.load(.monotonic),
        };
    }

    /// Recent aggregates computed over the ring. Computed under the history
    /// lock; the caller should not hold the decode lock when calling this.
    pub const Recent = struct {
        window_capacity: usize,
        window_count: usize,
        decode_tok_s_avg: ?f64,
        ttft_ms_avg: ?f64,
        prefix_hit_ratio_avg: ?f64,
        mtp_acceptance: ?f64,
        tokens_per_step: ?f64,
    };

    pub fn recent(self: *MetricsState) Recent {
        var buf: [HISTORY_CAPACITY]RequestMetricsRecord = undefined;
        const n = self.snapshotHistory(&buf);
        if (n == 0) {
            return .{
                .window_capacity = HISTORY_CAPACITY,
                .window_count = 0,
                .decode_tok_s_avg = null,
                .ttft_ms_avg = null,
                .prefix_hit_ratio_avg = null,
                .mtp_acceptance = null,
                .tokens_per_step = null,
            };
        }
        var tok_s_sum: f64 = 0;
        var tok_s_n: usize = 0;
        var ttft_sum: f64 = 0;
        var ttft_n: usize = 0;
        var hit_ratio_sum: f64 = 0;
        var hit_ratio_n: usize = 0;
        var accepted_sum: u64 = 0;
        var attempted_sum: u64 = 0;
        var gen_sum: u64 = 0;
        var steps_sum: u64 = 0;
        for (buf[0..n]) |r| {
            if (!r.ok) continue;
            if (r.generated_tokens > 0 and r.tok_s > 0) {
                tok_s_sum += r.tok_s;
                tok_s_n += 1;
            }
            if (r.ttft_ms > 0) {
                ttft_sum += r.ttft_ms;
                ttft_n += 1;
            }
            if (r.prompt_tokens > 0) {
                hit_ratio_sum += @as(f64, @floatFromInt(r.reuse_tokens)) / @as(f64, @floatFromInt(r.prompt_tokens));
                hit_ratio_n += 1;
            }
            accepted_sum += r.accepted;
            attempted_sum += r.attempted;
            gen_sum += r.generated_tokens;
            steps_sum += r.decode_steps;
        }
        return .{
            .window_capacity = HISTORY_CAPACITY,
            .window_count = n,
            .decode_tok_s_avg = if (tok_s_n > 0) tok_s_sum / @as(f64, @floatFromInt(tok_s_n)) else null,
            .ttft_ms_avg = if (ttft_n > 0) ttft_sum / @as(f64, @floatFromInt(ttft_n)) else null,
            .prefix_hit_ratio_avg = if (hit_ratio_n > 0) hit_ratio_sum / @as(f64, @floatFromInt(hit_ratio_n)) else null,
            .mtp_acceptance = if (attempted_sum > 0) @as(f64, @floatFromInt(accepted_sum)) / @as(f64, @floatFromInt(attempted_sum)) else null,
            .tokens_per_step = if (steps_sum > 0) @as(f64, @floatFromInt(gen_sum)) / @as(f64, @floatFromInt(steps_sum)) else null,
        };
    }
};

/// Wall-clock milliseconds since the Unix epoch. Re-exported from
/// runtime/time.zig for dashboard labels.
pub const unixNowMsFn = runtime_time.unixNowMs;

// ---------------------------------------------------------------------------
// JSON rendering
// ---------------------------------------------------------------------------

/// Static prefix-cache config known without reading mutable cache internals.
/// Used to keep known fields non-null in the busy path.
pub const BusyPrefixCacheConfig = struct {
    max_cache_tokens: u32,
    prefill_chunk_tokens: u32,
    prefill_chunk_group_size: u32,
};

/// Render `GET /v1/metrics` as JSON into `out`. `busy` indicates the decode
/// lock was not acquired for the live prefix-cache snapshot; when true the
/// unsafe live cache fields are emitted as `null` but known static config is
/// preserved via `busy_config`.
pub fn renderMetricsSnapshot(
    out: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    totals: MetricsState.Totals,
    recent_aggr: MetricsState.Recent,
    cache_snap: ?CacheSnapshot,
    cache_stats: ?CumulativeCacheStats,
    busy: bool,
    busy_config: BusyPrefixCacheConfig,
    history_count: usize,
    oldest_id: ?u64,
    newest_id: ?u64,
) !void {
    try out.print(gpa, "{{\"object\":\"metrics.snapshot\",\"busy\":{},", .{busy});
    try out.print(gpa, "\"requests_total\":{d},\"requests_failed\":{d},", .{ totals.requests_total, totals.requests_failed });
    try out.print(gpa, "\"prompt_tokens_total\":{d},\"generated_tokens_total\":{d},", .{ totals.prompt_tokens_total, totals.generated_tokens_total });
    try out.print(gpa, "\"reuse_tokens_total\":{d},\"computed_tokens_total\":{d},", .{ totals.reuse_tokens_total, totals.computed_tokens_total });

    // recent
    try out.print(gpa, "\"recent\":{{\"window_capacity\":{d},\"window_count\":{d},", .{ recent_aggr.window_capacity, recent_aggr.window_count });
    try writeNullableF64(out, gpa, "decode_tok_s_avg", recent_aggr.decode_tok_s_avg, true);
    try writeNullableF64(out, gpa, "ttft_ms_avg", recent_aggr.ttft_ms_avg, true);
    try writeNullableF64(out, gpa, "prefix_hit_ratio_avg", recent_aggr.prefix_hit_ratio_avg, true);
    try writeNullableF64(out, gpa, "mtp_acceptance", recent_aggr.mtp_acceptance, true);
    try writeNullableF64(out, gpa, "tokens_per_step", recent_aggr.tokens_per_step, false);
    try out.appendSlice(gpa, "},\"prefix_cache\":");

    // prefix_cache object
    if (busy) {
        try renderBusyPrefixCache(out, gpa, busy_config);
    } else if (cache_snap) |snap| {
        try renderPrefixCacheObject(out, gpa, snap, cache_stats, false);
    } else {
        try renderBusyPrefixCache(out, gpa, busy_config);
    }
    try out.appendSlice(gpa, ",\"history\":{");
    try out.print(gpa, "\"capacity\":{d},\"count\":{d},", .{ HISTORY_CAPACITY, history_count });
    if (oldest_id) |id| {
        try out.print(gpa, "\"oldest_request_id\":{d},", .{id});
    } else {
        try out.appendSlice(gpa, "\"oldest_request_id\":null,");
    }
    if (newest_id) |id| {
        try out.print(gpa, "\"newest_request_id\":{d}", .{id});
        try out.appendSlice(gpa, "}}");
    } else {
        try out.appendSlice(gpa, "\"newest_request_id\":null}}");
    }
}

/// Cumulative cache counters for the `prefix_cache` object. These come from
/// `prefix_state_cache.Cache.Stats` plus the per-record snapshot fields.
pub const CumulativeCacheStats = struct {
    hits: u64,
    misses: u64,
    hit_tokens: u64,
    computed_tokens: u64,
    evictions: u64,
    prefill_chunk_tokens: u32,
    prefill_chunk_group_size: u32,
};

fn renderBusyPrefixCache(out: *std.ArrayList(u8), gpa: std.mem.Allocator, cfg: BusyPrefixCacheConfig) !void {
    try out.appendSlice(gpa, "{\"object\":\"prefix.status\",\"busy\":true,");
    try out.appendSlice(gpa, "\"cached_tokens\":null,");
    try out.print(gpa, "\"max_cache_tokens\":{d},", .{cfg.max_cache_tokens});
    try out.appendSlice(gpa, "\"resident_entries\":null,\"reserved_tokens\":null,");
    try out.appendSlice(gpa, "\"resident_tokens\":null,\"pinned_entries\":null,");
    try out.appendSlice(gpa, "\"cached_logits\":null,\"hits\":null,\"misses\":null,");
    try out.appendSlice(gpa, "\"hit_tokens\":null,\"computed_tokens\":null,");
    try out.appendSlice(gpa, "\"token_hit_ratio\":null,\"evictions\":null,");
    try out.print(gpa, "\"prefill_chunk_tokens\":{d},\"prefill_chunk_group_size\":{d}", .{ cfg.prefill_chunk_tokens, cfg.prefill_chunk_group_size });
    try out.appendSlice(gpa, "}");
}

fn renderPrefixCacheObject(
    out: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    snap: CacheSnapshot,
    cache_stats: ?CumulativeCacheStats,
    busy: bool,
) !void {
    if (busy) {
        // This branch is unreachable from renderMetricsSnapshot (which calls
        // renderBusyPrefixCache directly for the busy path), but kept for
        // completeness if this function is reused elsewhere.
        try out.appendSlice(gpa, "{\"object\":\"prefix.status\",\"busy\":true}");
        return;
    }
    try out.appendSlice(gpa, "{\"object\":\"prefix.status\",\"busy\":false,");
    try out.print(gpa, "\"cached_tokens\":{d},\"max_cache_tokens\":{d},", .{ snap.cached_tokens, snap.max_cache_tokens });
    try out.print(gpa, "\"resident_entries\":{d},\"reserved_tokens\":{d},", .{ snap.resident_entries, snap.reserved_tokens });
    try out.print(gpa, "\"resident_tokens\":{d},\"pinned_entries\":{d},", .{ snap.resident_tokens, snap.pinned_entries });
    try out.print(gpa, "\"cached_logits\":{},", .{snap.cached_logits});
    if (cache_stats) |cs| {
        const served = cs.hit_tokens + cs.computed_tokens;
        const ratio: ?f64 = if (served > 0) @as(f64, @floatFromInt(cs.hit_tokens)) / @as(f64, @floatFromInt(served)) else null;
        try out.print(gpa, "\"hits\":{d},\"misses\":{d},\"hit_tokens\":{d},\"computed_tokens\":{d},", .{ cs.hits, cs.misses, cs.hit_tokens, cs.computed_tokens });
        try writeNullableF64(out, gpa, "token_hit_ratio", ratio, true);
        try out.print(gpa, "\"evictions\":{d},", .{cs.evictions});
        try out.print(gpa, "\"prefill_chunk_tokens\":{d},\"prefill_chunk_group_size\":{d}", .{ cs.prefill_chunk_tokens, cs.prefill_chunk_group_size });
        try out.appendSlice(gpa, "}");
    } else {
        try out.appendSlice(gpa, "\"hits\":null,\"misses\":null,\"hit_tokens\":null,\"computed_tokens\":null,\"token_hit_ratio\":null,\"evictions\":null,\"prefill_chunk_tokens\":null,\"prefill_chunk_group_size\":null}");
    }
}

/// Render `GET /v1/metrics/history` as JSON into `out`. `records` is
/// oldest-first.
pub fn renderMetricsHistory(
    out: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    records: []const RequestMetricsRecord,
) !void {
    try out.print(gpa, "{{\"object\":\"metrics.history\",\"capacity\":{d},\"count\":{d},\"records\":[", .{ HISTORY_CAPACITY, records.len });
    for (records, 0..) |r, i| {
        if (i > 0) try out.appendSlice(gpa, ",");
        try renderRecord(out, gpa, r);
    }
    try out.appendSlice(gpa, "]}");
}

fn renderRecord(out: *std.ArrayList(u8), gpa: std.mem.Allocator, r: RequestMetricsRecord) !void {
    try out.print(gpa, "{{\"request_id\":{d},", .{r.request_id});
    try out.print(gpa, "\"started_unix_ms\":{d},\"completed_unix_ms\":{d},", .{ r.started_unix_ms, r.completed_unix_ms });
    const duration_ms: f64 = if (r.completed_mono_ns > r.started_mono_ns)
        @as(f64, @floatFromInt(r.completed_mono_ns - r.started_mono_ns)) / std.time.ns_per_ms
    else
        0;
    try out.print(gpa, "\"duration_ms\":{d:.3},", .{duration_ms});
    try out.print(gpa, "\"ok\":{},\"error_kind\":", .{r.ok});
    if (r.ok) {
        try out.appendSlice(gpa, "null,");
    } else {
        try out.print(gpa, "\"{s}\",", .{@tagName(r.error_kind)});
    }
    try out.print(gpa, "\"mode\":\"{s}\",", .{@tagName(r.mode)});
    try out.print(gpa, "\"prompt_tokens\":{d},\"max_new_tokens\":{d},\"generated_tokens\":{d},", .{ r.prompt_tokens, r.max_new_tokens, r.generated_tokens });
    try out.print(gpa, "\"reuse_tokens\":{d},\"computed_tokens\":{d},", .{ r.reuse_tokens, r.computed_tokens });
    if (r.prompt_tokens > 0) {
        const ratio = @as(f64, @floatFromInt(r.reuse_tokens)) / @as(f64, @floatFromInt(r.prompt_tokens));
        try out.print(gpa, "\"prefix_hit_ratio\":{d:.3},", .{ratio});
    } else {
        try out.appendSlice(gpa, "\"prefix_hit_ratio\":null,");
    }
    try out.print(gpa, "\"prefill_ms\":{d:.3},\"setup_ms\":{d:.3},", .{ r.prefill_ms, r.setup_ms });
    try writeNullableF64(out, gpa, "drafter_seed_ms", r.drafter_seed_ms, true);
    try out.print(gpa, "\"ttft_ms\":{d:.3},\"decode_ms\":{d:.3},\"tok_s\":{d:.3},\"itl_ms\":{d:.3},\"emit_tok_s\":{d:.3},", .{ r.ttft_ms, r.decode_ms, r.tok_s, r.itl_ms, r.emit_tok_s });
    try out.print(gpa, "\"decode_steps\":{d},\"tokens_per_step\":", .{r.decode_steps});
    if (r.decode_steps > 0) {
        try out.print(gpa, "{d:.3},", .{r.tokens_per_step});
    } else {
        try out.appendSlice(gpa, "null,");
    }
    try out.print(gpa, "\"accepted\":{d},\"attempted\":{d},\"acceptance\":", .{ r.accepted, r.attempted });
    if (r.attempted > 0) {
        try out.print(gpa, "{d:.3},", .{r.acceptance});
    } else {
        try out.appendSlice(gpa, "null,");
    }
    try out.print(gpa, "\"cache_cached_tokens\":{d},\"cache_max_cache_tokens\":{d},", .{ r.cache_cached_tokens, r.cache_max_cache_tokens });
    try out.print(gpa, "\"cache_resident_entries\":{d},\"cache_reserved_tokens\":{d},", .{ r.cache_resident_entries, r.cache_reserved_tokens });
    try out.print(gpa, "\"cache_resident_tokens\":{d},\"cache_pinned_entries\":{d},", .{ r.cache_resident_tokens, r.cache_pinned_entries });
    try out.print(gpa, "\"cache_cached_logits\":{},\"cache_evictions_total\":{d}", .{ r.cache_cached_logits, r.cache_evictions_total });
    try out.appendSlice(gpa, "}");
}

/// Write `"name":value,` or `"name":null,` depending on `v`. `trailing_comma`
/// controls whether a trailing comma is emitted.
fn writeNullableF64(
    out: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    name: []const u8,
    v: ?f64,
    trailing_comma: bool,
) !void {
    if (v) |val| {
        try out.print(gpa, "\"{s}\":{d:.3}", .{ name, val });
    } else {
        try out.print(gpa, "\"{s}\":null", .{name});
    }
    if (trailing_comma) try out.appendSlice(gpa, ",");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn testIo() Io {
    // Tests need an Io for the history mutex (futex-based). A single-threaded
    // Threaded Io is cheap and sufficient for lock/unlock without spawning
    // worker threads.
    const t = struct {
        var threaded: std.Io.Threaded = undefined;
    };
    t.threaded = std.Io.Threaded.init(std.heap.page_allocator, .{});
    return t.threaded.io();
}

test "MetricsState ring stores records oldest-first and caps at capacity" {
    var state = MetricsState.init(testIo());
    var i: u32 = 0;
    while (i < HISTORY_CAPACITY + 5) : (i += 1) {
        var req = state.beginRequest(256);
        req.setPromptStats(i + 1, 256, i, 1);
        req.setMode(.greedy);
        req.finish();
    }
    var buf: [HISTORY_CAPACITY]RequestMetricsRecord = undefined;
    const n = state.snapshotHistory(&buf);
    try std.testing.expectEqual(@as(usize, HISTORY_CAPACITY), n);
    // The oldest surviving record should be request id 6 (we wrote 261, cap 256).
    try std.testing.expectEqual(@as(u64, 6), buf[0].request_id);
    // Newest should be id 261.
    try std.testing.expectEqual(@as(u64, 261), buf[HISTORY_CAPACITY - 1].request_id);
}

test "MetricsState totals track cumulative counts" {
    var state = MetricsState.init(testIo());
    var req = state.beginRequest(256);
    req.setPromptStats(100, 256, 80, 20);
    req.setDecodeMetrics(10, 5, 15, 100, 24, 40, 24, 50, 1.5, 40, 50, 0.8, null, 75);
    req.setMode(.mtp);
    req.finish();

    const t = state.totals();
    try std.testing.expectEqual(@as(u64, 1), t.requests_total);
    try std.testing.expectEqual(@as(u64, 0), t.requests_failed);
    try std.testing.expectEqual(@as(u64, 100), t.prompt_tokens_total);
    try std.testing.expectEqual(@as(u64, 75), t.generated_tokens_total);
    try std.testing.expectEqual(@as(u64, 80), t.reuse_tokens_total);
    try std.testing.expectEqual(@as(u64, 20), t.computed_tokens_total);
}

test "MetricsState failed requests increment requests_failed" {
    var state = MetricsState.init(testIo());
    var req = state.beginRequest(256);
    req.finishError(.generation);
    const t = state.totals();
    try std.testing.expectEqual(@as(u64, 1), t.requests_total);
    try std.testing.expectEqual(@as(u64, 1), t.requests_failed);
    var buf: [HISTORY_CAPACITY]RequestMetricsRecord = undefined;
    const n = state.snapshotHistory(&buf);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(false, buf[0].ok);
    try std.testing.expectEqual(ErrorKind.generation, buf[0].error_kind);
}

test "MetricsState recent aggregates average over successful records" {
    var state = MetricsState.init(testIo());
    // Two successful MTP requests with known tok_s and ttft.
    {
        var req = state.beginRequest(256);
        req.setPromptStats(1000, 256, 900, 100);
        req.setDecodeMetrics(900, 100, 1000, 4000, 24.0, 40, 24.0, 100, 1.8, 80, 100, 0.8, 50.0, 180);
        req.setMode(.mtp);
        req.finish();
    }
    {
        var req = state.beginRequest(256);
        req.setPromptStats(500, 256, 400, 100);
        req.setDecodeMetrics(500, 50, 550, 2000, 30.0, 20, 30.0, 60, 2.0, 50, 60, 0.833, 25.0, 120);
        req.setMode(.mtp);
        req.finish();
    }
    // One failed request; should not contribute to tok_s / ttft averages.
    {
        var req = state.beginRequest(256);
        req.finishError(.generation);
    }
    const r = state.recent();
    try std.testing.expectEqual(@as(usize, 3), r.window_count);
    try std.testing.expectEqual(@as(usize, HISTORY_CAPACITY), r.window_capacity);
    // tok_s avg = (24 + 30) / 2 = 27
    try std.testing.expectApproxEqAbs(@as(f64, 27.0), r.decode_tok_s_avg.?, 1e-6);
    // ttft avg = (1000 + 550) / 2 = 775
    try std.testing.expectApproxEqAbs(@as(f64, 775.0), r.ttft_ms_avg.?, 1e-6);
    // hit ratio avg = (0.9 + 0.8) / 2 = 0.85
    try std.testing.expectApproxEqAbs(@as(f64, 0.85), r.prefix_hit_ratio_avg.?, 1e-6);
    // mtp acceptance = (80+50)/(100+60) = 130/160 = 0.8125
    try std.testing.expectApproxEqAbs(@as(f64, 0.8125), r.mtp_acceptance.?, 1e-6);
    // tokens_per_step = (180+120)/(100+60) = 300/160 = 1.875
    try std.testing.expectApproxEqAbs(@as(f64, 1.875), r.tokens_per_step.?, 1e-6);
}

test "MetricsState recent returns nulls when no successful records" {
    var state = MetricsState.init(testIo());
    {
        var req = state.beginRequest(256);
        req.finishError(.generation);
    }
    const r = state.recent();
    try std.testing.expectEqual(@as(usize, 1), r.window_count);
    try std.testing.expect(r.decode_tok_s_avg == null);
    try std.testing.expect(r.ttft_ms_avg == null);
    try std.testing.expect(r.prefix_hit_ratio_avg == null);
    try std.testing.expect(r.mtp_acceptance == null);
    try std.testing.expect(r.tokens_per_step == null);
}

test "renderMetricsHistory produces valid JSON with expected fields" {
    var state = MetricsState.init(testIo());
    {
        var req = state.beginRequest(256);
        req.setPromptStats(100, 256, 80, 20);
        req.setDecodeMetrics(10, 5, 15, 100, 24, 40, 24, 50, 1.5, 40, 50, 0.8, 2.5, 75);
        req.setMode(.mtp);
        const snap: CacheSnapshot = .{
            .cached_tokens = 1000,
            .max_cache_tokens = 24576,
            .resident_entries = 2,
            .reserved_tokens = 1500,
            .resident_tokens = 1000,
            .pinned_entries = 1,
            .cached_logits = true,
            .evictions_total = 0,
        };
        req.setCacheSnapshot(snap);
        req.finish();
    }
    var buf: [HISTORY_CAPACITY]RequestMetricsRecord = undefined;
    const n = state.snapshotHistory(&buf);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try renderMetricsHistory(&out, std.testing.allocator, buf[0..n]);
    // Spot-check key fields are present.
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"object\":\"metrics.history\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"request_id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"mode\":\"mtp\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"drafter_seed_ms\":2.500") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"acceptance\":0.800") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"cache_evictions_total\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"prefix_hit_ratio\":0.800") != null);
    // Validate the JSON actually parses (catches trailing commas, missing
    // commas, bad nullability rendering).
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    _ = try std.json.parseFromSlice(std.json.Value, arena.allocator(), out.items, .{});
}

test "renderMetricsSnapshot output is valid JSON (busy and non-busy)" {
    var state = MetricsState.init(testIo());
    const t = state.totals();
    const r = state.recent();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Busy path
    {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(std.testing.allocator);
        try renderMetricsSnapshot(&out, std.testing.allocator, t, r, null, null, true, .{ .max_cache_tokens = 24576, .prefill_chunk_tokens = 1600, .prefill_chunk_group_size = 10 }, 0, null, null);
        _ = try std.json.parseFromSlice(std.json.Value, arena.allocator(), out.items, .{});
    }
    // Non-busy path with live cache fields
    {
        const snap: CacheSnapshot = .{
            .cached_tokens = 12345,
            .max_cache_tokens = 24576,
            .resident_entries = 3,
            .reserved_tokens = 14000,
            .resident_tokens = 12345,
            .pinned_entries = 1,
            .cached_logits = true,
            .evictions_total = 2,
        };
        const cs: CumulativeCacheStats = .{
            .hits = 17,
            .misses = 4,
            .hit_tokens = 99212,
            .computed_tokens = 45678,
            .evictions = 2,
            .prefill_chunk_tokens = 1600,
            .prefill_chunk_group_size = 10,
        };
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(std.testing.allocator);
        try renderMetricsSnapshot(&out, std.testing.allocator, t, r, snap, cs, false, .{ .max_cache_tokens = 24576, .prefill_chunk_tokens = 1600, .prefill_chunk_group_size = 10 }, 1, 1, 1);
        _ = try std.json.parseFromSlice(std.json.Value, arena.allocator(), out.items, .{});
    }
}

test "renderMetricsSnapshot busy path emits null cache fields" {
    var state = MetricsState.init(testIo());
    const t = state.totals();
    const r = state.recent();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try renderMetricsSnapshot(&out, std.testing.allocator, t, r, null, null, true, .{ .max_cache_tokens = 24576, .prefill_chunk_tokens = 1600, .prefill_chunk_group_size = 10 }, 0, null, null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"busy\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"cached_tokens\":null") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"hits\":null") != null);
}

test "renderMetricsSnapshot non-busy path emits live cache fields" {
    var state = MetricsState.init(testIo());
    const t = state.totals();
    const r = state.recent();
    const snap: CacheSnapshot = .{
        .cached_tokens = 12345,
        .max_cache_tokens = 24576,
        .resident_entries = 3,
        .reserved_tokens = 14000,
        .resident_tokens = 12345,
        .pinned_entries = 1,
        .cached_logits = true,
        .evictions_total = 2,
    };
    const cs: CumulativeCacheStats = .{
        .hits = 17,
        .misses = 4,
        .hit_tokens = 99212,
        .computed_tokens = 45678,
        .evictions = 2,
        .prefill_chunk_tokens = 1600,
        .prefill_chunk_group_size = 10,
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try renderMetricsSnapshot(&out, std.testing.allocator, t, r, snap, cs, false, .{ .max_cache_tokens = 24576, .prefill_chunk_tokens = 1600, .prefill_chunk_group_size = 10 }, 0, null, null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"busy\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"cached_tokens\":12345") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"hits\":17") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"token_hit_ratio\":") != null);
}

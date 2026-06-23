//! Disk persistence for the startup-prewarmed prompt-prefix cache.
//!
//! Raw cold prefill of the captured ~16k-token Pi static prefix costs about a
//! minute of GPU compute on this hardware class, at parity with the
//! same-machine MLX reference. The serving fix is to stop paying it per process:
//! after startup prewarm the warmed prefix state is written to one bounded file,
//! and the next server start reloads it in seconds.
//!
//! The file holds exactly one prefix: the token ids, the cached next-token
//! argmax, the 24 linear-attention conv/recurrent states, and the 8
//! full-attention BF16 KV caches compacted to the cached token rows. It is
//! written atomically (temp file + rename) and validated on load against the
//! format version, the model fingerprint, and the exact expected byte length;
//! any mismatch is ignored and the server falls back to normal cold prefill.

const std = @import("std");
const checked_math = @import("../runtime/checked_math.zig");
const metal = @import("../runtime/metal.zig");
const runtime_time = @import("../runtime/time.zig");
const state_mod = @import("../model/state.zig");
const decoder_layer = @import("../model/decoder_layer.zig");
const block_attn = @import("../model/block_attn.zig");
const block_linear = @import("../model/block_linear.zig");
const prefix_state_cache = @import("prefix_state_cache.zig");
const dims = @import("../model/dims.zig");

const Cache = prefix_state_cache.Cache;
const monotonicNowNs = runtime_time.monotonicNowNs;
const log = std.log.scoped(.peregrine_serve);

const MAGIC: u64 = 0x3153_5846_5047_5250; // "PRGPFXS1" little-endian
const VERSION: u32 = 3;
const DTYPE_BF16: u32 = 0;
const hiddenRowsBytes = dims.hiddenRowsBytes;

pub const default_file_name = "qwen35-9b-q4-prefix-state.bin";

pub const StartupState = struct {
    path: []const u8,
    model_fingerprint: u64,
    loaded_tokens: usize,
};

/// Fixed-size little-endian header; this file format is a local same-machine
/// cache, so native (aarch64) byte order is part of the format.
const Header = extern struct {
    magic: u64,
    version: u32,
    dtype: u32,
    model_fingerprint: u64,
    token_count: u64,
    next_token: u32,
    next_token_valid: u32,
    /// 1 when the hidden section holds real captured normalized hiddens, 0 when
    /// the section is present but zero-filled (written by a non-MTP run).
    hidden_valid: u32,
};

const LINEAR_CONV_BYTES: usize = (block_linear.CONV_K - 1) * block_linear.CONV_DIM * @sizeOf(f32);
const LINEAR_RECUR_BYTES: usize = block_linear.HV * block_linear.DV * block_linear.DK * @sizeOf(f32);
const FULL_VALUE_BYTES: usize = state_mod.FullCacheDType.bf16.sizeInBytes();

fn layerCounts() struct { linear: usize, full: usize } {
    var linear: usize = 0;
    var full: usize = 0;
    for (0..state_mod.NUM_LAYERS) |i| {
        if (decoder_layer.isLinear(i)) linear += 1 else full += 1;
    }
    return .{ .linear = linear, .full = full };
}

fn expectedFileBytes(token_count: usize) !usize {
    const counts = layerCounts();
    const tokens_bytes = try checked_math.product(.{ token_count, @sizeOf(u32) });
    const key_bytes = try checked_math.product(.{ block_attn.NUM_KV, token_count, block_attn.HEAD_DIM, FULL_VALUE_BYTES });
    const value_bytes = key_bytes;
    const full_bytes = try checked_math.product(.{ counts.full, key_bytes + value_bytes });
    const linear_bytes = counts.linear * (LINEAR_CONV_BYTES + LINEAR_RECUR_BYTES);
    const hidden_bytes = try hiddenRowsBytes(token_count);
    return @sizeOf(Header) + tokens_bytes + linear_bytes + full_bytes + hidden_bytes;
}

fn keyRunStart(head: usize, stride: usize) usize {
    return head * stride * block_attn.HEAD_DIM * FULL_VALUE_BYTES;
}

fn valueRunStart(head: usize, dim: usize, stride: usize) usize {
    return (head * block_attn.HEAD_DIM + dim) * stride * FULL_VALUE_BYTES;
}

const FullCacheRun = struct {
    buffer: enum { key, value },
    start: usize,
    byte_len: usize,
};

const FullCacheRunIterator = struct {
    stride: usize,
    key_run_bytes: usize,
    value_run_bytes: usize,
    phase: enum { keys, values, done } = .keys,
    head: usize = 0,
    dim: usize = 0,

    fn next(self: *FullCacheRunIterator) ?FullCacheRun {
        switch (self.phase) {
            .keys => {
                if (self.head < block_attn.NUM_KV) {
                    const head = self.head;
                    self.head += 1;
                    return .{ .buffer = .key, .start = keyRunStart(head, self.stride), .byte_len = self.key_run_bytes };
                }
                self.phase = .values;
                self.head = 0;
                self.dim = 0;
                return self.next();
            },
            .values => {
                if (self.head < block_attn.NUM_KV) {
                    const head = self.head;
                    const dim = self.dim;
                    self.dim += 1;
                    if (self.dim == block_attn.HEAD_DIM) {
                        self.dim = 0;
                        self.head += 1;
                    }
                    return .{ .buffer = .value, .start = valueRunStart(head, dim, self.stride), .byte_len = self.value_run_bytes };
                }
                self.phase = .done;
                return null;
            },
            .done => return null,
        }
    }
};

fn fullCacheRuns(token_count: usize, stride: usize) FullCacheRunIterator {
    return .{
        .stride = stride,
        .key_run_bytes = token_count * block_attn.HEAD_DIM * FULL_VALUE_BYTES,
        .value_run_bytes = token_count * FULL_VALUE_BYTES,
    };
}

/// Identity of the served checkpoint for cache validity. The persisted state
/// is weight-dependent, so the fingerprint is a real content identity, not
/// path or filesystem metadata (paths can be mutable directories or
/// re-pointed symlinks, and tools like `cp -p` preserve sizes and mtimes
/// across a weight swap): the exact `config.json` bytes plus the full byte
/// content of every safetensors shard, hashed in sorted-name order. The
/// caller computes this once at startup after weight upload, so the shard
/// read is a page-cache-warm pass; it only runs when persistence is enabled.
/// Tokenizer changes need no coverage here: cached token ids are compared
/// against the freshly tokenized request, so a tokenizer swap already misses.
fn modelFingerprint(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);

    var hasher = std.hash.Wyhash.init(MAGIC ^ VERSION);
    const config_bytes = try dir.readFileAlloc(io, "config.json", gpa, .limited(4 * 1024 * 1024));
    defer gpa.free(config_bytes);
    hasher.update(config_bytes);

    var shard_names: std.ArrayList([]u8) = .empty;
    defer {
        for (shard_names.items) |name| gpa.free(name);
        shard_names.deinit(gpa);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try shard_names.append(gpa, try gpa.dupe(u8, entry.name));
    }
    // Directory iteration order is filesystem-defined; sort for determinism.
    std.mem.sort([]u8, shard_names.items, {}, struct {
        fn lessThan(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    const chunk = try gpa.alloc(u8, 4 * 1024 * 1024);
    defer gpa.free(chunk);
    for (shard_names.items) |name| {
        hasher.update(name);
        var file = try dir.openFile(io, name, .{});
        defer file.close(io);
        var read_buf: [64 * 1024]u8 = undefined;
        var file_reader = file.reader(io, &read_buf);
        while (true) {
            const n = try file_reader.interface.readSliceShort(chunk);
            if (n == 0) break;
            hasher.update(chunk[0..n]);
        }
    }
    return hasher.final();
}

/// Default persisted-state location: `$HOME/.cache/peregrine/<file>`.
/// Returns null when HOME is unavailable; the caller then skips persistence.
pub fn resolveDefaultPath(gpa: std.mem.Allocator) !?[]u8 {
    const home = std.c.getenv("HOME") orelse return null;
    const home_slice = std.mem.span(home);
    if (home_slice.len == 0) return null;
    return try std.fmt.allocPrint(gpa, "{s}/.cache/peregrine/{s}", .{ home_slice, default_file_name });
}

pub fn loadAtStartup(
    io: std.Io,
    gpa: std.mem.Allocator,
    device: *metal.Device,
    cache: *Cache,
    path: []const u8,
    model_dir: []const u8,
    vocab: usize,
) !StartupState {
    const fingerprint = try modelFingerprint(gpa, io, model_dir);
    var state: StartupState = .{
        .path = path,
        .model_fingerprint = fingerprint,
        .loaded_tokens = 0,
    };
    const load_start_ns = monotonicNowNs();
    const loaded = load(io, gpa, device, cache, path, fingerprint, vocab) catch |e| blk: {
        log.info("ignoring persisted prefix state {s} ({any})", .{ path, e });
        break :blk null;
    };
    if (loaded) |token_count| {
        state.loaded_tokens = token_count;
        log.info(
            "loaded persisted prefix state from {s} (cached_tokens={d}, cached_logits={}, elapsed_ms={d})",
            .{ path, token_count, cache.activeCachedNextTokenValid(), (monotonicNowNs() - load_start_ns) / std.time.ns_per_ms },
        );
    }
    return state;
}

pub fn saveAfterPrewarm(io: std.Io, cache: *const Cache, state: StartupState, computed_tokens: usize) void {
    const cached_tokens = cache.activeTokenCount();
    const state_is_new = computed_tokens > 0 or state.loaded_tokens == 0;
    if (cached_tokens == 0 or !state_is_new) return;

    const save_start_ns = monotonicNowNs();
    if (save(io, state.path, cache, state.model_fingerprint)) |_| {
        log.info(
            "persisted prefix state to {s} (cached_tokens={d}, elapsed_ms={d})",
            .{ state.path, cached_tokens, (monotonicNowNs() - save_start_ns) / std.time.ns_per_ms },
        );
    } else |e| {
        log.warn("failed to persist prefix state to {s} ({any})", .{ state.path, e });
    }
}

/// Atomically write the cached prefix to `path`. The cache must hold a
/// nonempty prefix.
fn save(io: std.Io, path: []const u8, cache: *const Cache, model_fingerprint: u64) !void {
    const entry = cache.persistedEntryConst() orelse return error.EmptyPrefix;
    const token_count = entry.tokens.items.len;
    if (token_count == 0) return error.EmptyPrefix;
    if (token_count > entry.cached_state.max_seq) return error.SequenceTooLong;

    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true, .make_path = true });
    defer atomic.deinit(io);

    var write_buf: [256 * 1024]u8 = undefined;
    var file_writer = atomic.file.writer(io, &write_buf);
    const w = &file_writer.interface;

    const header: Header = .{
        .magic = MAGIC,
        .version = VERSION,
        .dtype = DTYPE_BF16,
        .model_fingerprint = model_fingerprint,
        .token_count = token_count,
        .next_token = entry.cached_next_token,
        .next_token_valid = @intFromBool(entry.cached_next_token_valid),
        .hidden_valid = @intFromBool(entry.hidden_valid),
    };
    try w.writeAll(std.mem.asBytes(&header));
    try w.writeAll(std.mem.sliceAsBytes(entry.tokens.items));

    const state = &entry.cached_state;
    const stride = state.full_cache_stride;
    for (&state.layers) |*layer| switch (layer.*) {
        .linear => |*l| {
            try w.writeAll(std.mem.sliceAsBytes(l.conv.slice(f32)));
            try w.writeAll(std.mem.sliceAsBytes(l.recur.slice(f32)));
        },
        .full => |*f| {
            const keys = f.cache_k.slice(u8);
            const values = f.cache_v.slice(u8);
            var runs = fullCacheRuns(token_count, stride);
            while (runs.next()) |run| {
                const bytes = switch (run.buffer) {
                    .key => keys,
                    .value => values,
                };
                try w.writeAll(bytes[run.start..][0..run.byte_len]);
            }
        },
    };
    // Hidden section: normalized target hidden rows for the cached prefix.
    // Always `hiddenRowsBytes(token_count)` bytes so the file size is
    // deterministic. Real rows when `entry.hidden_valid`; zero-fill
    // otherwise (non-MTP run, or MTP run that hasn't captured hiddens yet).
    // The header's `hidden_valid` flag tells the loader which is which.
    const hidden_bytes = try hiddenRowsBytes(token_count);
    if (entry.hidden) |hidden| {
        if (hidden.length < hidden_bytes) return error.SequenceTooLong;
        try w.writeAll(hidden.slice(u8)[0..hidden_bytes]);
    } else {
        const zero = try std.heap.page_allocator.alloc(u8, @min(hidden_bytes, 1 << 20));
        defer std.heap.page_allocator.free(zero);
        @memset(zero, 0);
        var remaining = hidden_bytes;
        while (remaining > 0) {
            const n = @min(remaining, zero.len);
            try w.writeAll(zero[0..n]);
            remaining -= n;
        }
    }
    try w.flush();
    try atomic.replace(io);
}

/// Load a persisted prefix into `cache`. Returns the cached token count, or
/// null when the file is missing or fails validation (including a cached
/// next token outside the model's vocab); the cache is left unchanged on
/// rejected files and cleared on mid-read failures.
fn load(
    io: std.Io,
    gpa: std.mem.Allocator,
    device: *metal.Device,
    cache: *Cache,
    path: []const u8,
    model_fingerprint: u64,
    vocab: usize,
) !?usize {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer file.close(io);

    var read_buf: [256 * 1024]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const r = &file_reader.interface;

    var header: Header = undefined;
    r.readSliceAll(std.mem.asBytes(&header)) catch return null;
    if (header.magic != MAGIC or header.version != VERSION or header.dtype != DTYPE_BF16) return null;
    if (header.model_fingerprint != model_fingerprint) return null;
    if (header.next_token_valid > 1) return null;
    if (header.next_token_valid == 1 and header.next_token >= vocab) return null;
    if (header.hidden_valid > 1) return null;
    const token_count = std.math.cast(usize, header.token_count) orelse return null;
    if (token_count == 0 or token_count > cache.max_tokens) return null;
    const expected_bytes = expectedFileBytes(token_count) catch return null;
    const file_bytes = try file.length(io);
    if (file_bytes != expected_bytes) return null;

    const match = try cache.loadEntrySlot(gpa, device, token_count);
    errdefer cache.clearAll(gpa);
    const entry = cache.activeEntry() orelse return error.EmptyPrefix;
    std.debug.assert(match.index < cache.entries.items.len);
    try r.readSliceAll(std.mem.sliceAsBytes(entry.tokens.items));
    try cache.finishLoadedEntry(gpa, match);

    const state = &entry.cached_state;
    const stride = state.full_cache_stride;
    for (&state.layers) |*layer| switch (layer.*) {
        .linear => |*l| {
            try r.readSliceAll(std.mem.sliceAsBytes(l.conv.slice(f32)));
            try r.readSliceAll(std.mem.sliceAsBytes(l.recur.slice(f32)));
        },
        .full => |*f| {
            const keys = f.cache_k.slice(u8);
            const values = f.cache_v.slice(u8);
            var runs = fullCacheRuns(token_count, stride);
            while (runs.next()) |run| {
                const bytes = switch (run.buffer) {
                    .key => keys,
                    .value => values,
                };
                try r.readSliceAll(bytes[run.start..][0..run.byte_len]);
            }
        },
    };

    entry.cached_next_token = header.next_token;
    entry.cached_next_token_valid = header.next_token_valid == 1;

    // Hidden section: read into the entry's hidden buffer when present (MTP
    // enabled); otherwise discard the bytes so the stream stays in sync.
    // `hidden_valid` follows the header flag: a v3 file written by a non-MTP
    // run has the section zero-filled and the flag clear, so MTP drafter
    // seeding falls back instead of consuming all-zero hiddens.
    const hidden_bytes = try hiddenRowsBytes(token_count);
    if (entry.hidden) |hidden| {
        if (hidden.length < hidden_bytes) return error.SequenceTooLong;
        try r.readSliceAll(hidden.slice(u8)[0..hidden_bytes]);
        entry.hidden_valid = header.hidden_valid == 1;
    } else {
        const scratch = try std.heap.page_allocator.alloc(u8, @min(hidden_bytes, 1 << 20));
        defer std.heap.page_allocator.free(scratch);
        var remaining = hidden_bytes;
        while (remaining > 0) {
            const n = @min(remaining, scratch.len);
            try r.readSliceAll(scratch[0..n]);
            remaining -= n;
        }
    }
    return token_count;
}

test "model fingerprint tracks checkpoint content, not the caller path" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"a\":1}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001.safetensors", .data = "weights-v1" });
    var path_buf: [4096]u8 = undefined;
    const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);
    const dir_path = path_buf[0..path_len];

    const first = try modelFingerprint(gpa, std.testing.io, dir_path);
    // Stable for unchanged content at the same path.
    try std.testing.expectEqual(first, try modelFingerprint(gpa, std.testing.io, dir_path));

    // A weight swap with identical name and byte length must change it: this
    // is the `cp -p` case where size and mtime metadata are preserved.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001.safetensors", .data = "weights-v2" });
    const same_size_swap = try modelFingerprint(gpa, std.testing.io, dir_path);
    try std.testing.expect(first != same_size_swap);

    // Size changes and config changes must change it too.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001.safetensors", .data = "weights-v2-longer" });
    const reshard = try modelFingerprint(gpa, std.testing.io, dir_path);
    try std.testing.expect(same_size_swap != reshard);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"a\":2}" });
    try std.testing.expect(reshard != try modelFingerprint(gpa, std.testing.io, dir_path));
}

test "expected file bytes follow the one-model layer topology" {
    const counts = layerCounts();
    try std.testing.expectEqual(@as(usize, 24), counts.linear);
    try std.testing.expectEqual(@as(usize, 8), counts.full);

    const token_count: usize = 16_153;
    const expected = @sizeOf(Header) +
        token_count * @sizeOf(u32) +
        24 * (LINEAR_CONV_BYTES + LINEAR_RECUR_BYTES) +
        8 * 2 * (block_attn.NUM_KV * token_count * block_attn.HEAD_DIM * FULL_VALUE_BYTES) +
        try hiddenRowsBytes(token_count);
    try std.testing.expectEqual(expected, try expectedFileBytes(token_count));
}

test "save and load round-trip the cached prefix" {
    var device = metal.Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return error.SkipZigTest,
        else => return err,
    };
    defer device.destroy();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(std.testing.io, &path_buf);
    const dir_path = path_buf[0..dir_len];
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/prefix-state.bin", .{dir_path});
    defer std.testing.allocator.free(path);

    const token_count: usize = 7;
    var src = try Cache.init(64, 16);
    defer src.deinit(std.testing.allocator);
    _ = try src.loadEntrySlot(std.testing.allocator, &device, token_count);
    const src_entry = src.activeEntry().?;
    for (0..token_count) |i| src_entry.tokens.items[i] = @intCast(i + 100);
    src_entry.cached_next_token = 4242;
    src_entry.cached_next_token_valid = true;
    var fill: u8 = 1;
    for (&src_entry.cached_state.layers) |*layer| switch (layer.*) {
        .linear => |*l| {
            @memset(l.conv.slice(u8), fill);
            @memset(l.recur.slice(u8), fill +% 1);
            fill +%= 2;
        },
        .full => |*f| {
            @memset(f.cache_k.slice(u8), fill);
            @memset(f.cache_v.slice(u8), fill +% 1);
            fill +%= 2;
        },
    };

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"a\":1}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001.safetensors", .data = "weights" });
    const fingerprint = try modelFingerprint(std.testing.allocator, std.testing.io, dir_path);
    try save(std.testing.io, path, &src, fingerprint);

    var dst = try Cache.init(64, 16);
    defer dst.deinit(std.testing.allocator);
    const vocab: usize = 8192;
    const loaded = try load(std.testing.io, std.testing.allocator, &device, &dst, path, fingerprint, vocab);
    try std.testing.expectEqual(@as(?usize, token_count), loaded);
    const dst_entry = dst.activeEntryConst().?;
    try std.testing.expectEqualSlices(u32, src_entry.tokens.items, dst_entry.tokens.items);
    try std.testing.expectEqual(src_entry.cached_next_token, dst_entry.cached_next_token);
    try std.testing.expect(dst_entry.cached_next_token_valid);

    const stride = dst_entry.cached_state.full_cache_stride;
    try std.testing.expectEqual(src_entry.cached_state.full_cache_stride, stride);
    for (&src_entry.cached_state.layers, &dst_entry.cached_state.layers) |*a, *b| switch (a.*) {
        .linear => |*l| {
            try std.testing.expectEqualSlices(f32, l.conv.slice(f32), b.linear.conv.slice(f32));
            try std.testing.expectEqualSlices(f32, l.recur.slice(f32), b.linear.recur.slice(f32));
        },
        .full => |*f| {
            var runs = fullCacheRuns(token_count, stride);
            while (runs.next()) |run| {
                const src_bytes = switch (run.buffer) {
                    .key => f.cache_k.slice(u8),
                    .value => f.cache_v.slice(u8),
                };
                const dst_bytes = switch (run.buffer) {
                    .key => b.full.cache_k.slice(u8),
                    .value => b.full.cache_v.slice(u8),
                };
                try std.testing.expectEqualSlices(u8, src_bytes[run.start..][0..run.byte_len], dst_bytes[run.start..][0..run.byte_len]);
            }
        },
    };

    // Validation failures are rejected before the cache is touched.
    const wrong = try load(std.testing.io, std.testing.allocator, &device, &dst, path, fingerprint +% 1, vocab);
    try std.testing.expectEqual(@as(?usize, null), wrong);
    try std.testing.expectEqual(token_count, dst.activeTokenCount());

    // A cached next token outside the served vocab is rejected.
    const small_vocab = try load(std.testing.io, std.testing.allocator, &device, &dst, path, fingerprint, 100);
    try std.testing.expectEqual(@as(?usize, null), small_vocab);
    try std.testing.expectEqual(token_count, dst.activeTokenCount());
}

test "save and load round-trip the normalized hidden states" {
    var device = metal.Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return error.SkipZigTest,
        else => return err,
    };
    defer device.destroy();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(std.testing.io, &path_buf);
    const dir_path = path_buf[0..dir_len];
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/prefix-state.bin", .{dir_path});
    defer std.testing.allocator.free(path);

    const token_count: usize = 5;
    var src = try Cache.init(64, 16);
    defer src.deinit(std.testing.allocator);
    src.enableHiddenTracking();
    _ = try src.loadEntrySlot(std.testing.allocator, &device, token_count);
    const src_entry = src.activeEntry().?;
    for (0..token_count) |i| src_entry.tokens.items[i] = @intCast(i + 10);
    src_entry.cached_next_token = 777;
    src_entry.cached_next_token_valid = true;
    // Fill the hidden buffer with a recognizable pattern across the valid rows.
    const hidden_bytes = try hiddenRowsBytes(token_count);
    const src_hidden = src_entry.hidden.?;
    try std.testing.expect(hidden_bytes <= src_hidden.length);
    for (src_hidden.slice(u8)[0..hidden_bytes]) |*b| b.* = 0xAB;
    src_entry.hidden_valid = true;

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"a\":1}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001.safetensors", .data = "weights" });
    const fingerprint = try modelFingerprint(std.testing.allocator, std.testing.io, dir_path);
    try save(std.testing.io, path, &src, fingerprint);

    var dst = try Cache.init(64, 16);
    defer dst.deinit(std.testing.allocator);
    dst.enableHiddenTracking();
    const loaded = try load(std.testing.io, std.testing.allocator, &device, &dst, path, fingerprint, 8192);
    try std.testing.expectEqual(@as(?usize, token_count), loaded);
    const dst_entry = dst.activeEntryConst().?;
    try std.testing.expectEqualSlices(u32, src_entry.tokens.items, dst_entry.tokens.items);
    // Hidden rows survive the round-trip.
    const dst_hidden = dst_entry.hidden.?;
    try std.testing.expectEqualSlices(u8, src_hidden.slice(u8)[0..hidden_bytes], dst_hidden.slice(u8)[0..hidden_bytes]);
    try std.testing.expect(dst_entry.hidden_valid);
}

test "non-MTP v3 file loads with hidden_valid clear under MTP" {
    // A file written by a non-MTP run stores zero hiddens and clears
    // `hidden_valid`. Loading it into an MTP-enabled cache must allocate the
    // hidden buffer (so capacity is ready) but keep `hidden_valid` false, so
    // `seedMtpDrafter` falls back to greedy decode instead of consuming
    // all-zero hiddens.
    var device = metal.Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return error.SkipZigTest,
        else => return err,
    };
    defer device.destroy();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(std.testing.io, &path_buf);
    const dir_path = path_buf[0..dir_len];
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/prefix-state.bin", .{dir_path});
    defer std.testing.allocator.free(path);

    const token_count: usize = 4;
    // Source cache has no hidden tracking (non-MTP run).
    var src = try Cache.init(64, 16);
    defer src.deinit(std.testing.allocator);
    _ = try src.loadEntrySlot(std.testing.allocator, &device, token_count);
    const src_entry = src.activeEntry().?;
    for (0..token_count) |i| src_entry.tokens.items[i] = @intCast(i + 1);
    src_entry.cached_next_token = 99;
    src_entry.cached_next_token_valid = true;
    try std.testing.expect(src_entry.hidden == null);
    try std.testing.expect(!src_entry.hidden_valid);

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"a\":1}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001.safetensors", .data = "weights" });
    const fingerprint = try modelFingerprint(std.testing.allocator, std.testing.io, dir_path);
    try save(std.testing.io, path, &src, fingerprint);

    // Destination cache enables hidden tracking (MTP run).
    var dst = try Cache.init(64, 16);
    defer dst.deinit(std.testing.allocator);
    dst.enableHiddenTracking();
    const loaded = try load(std.testing.io, std.testing.allocator, &device, &dst, path, fingerprint, 8192);
    try std.testing.expectEqual(@as(?usize, token_count), loaded);
    const dst_entry = dst.activeEntryConst().?;
    // Buffer allocated (capacity ready) but not valid.
    try std.testing.expect(dst_entry.hidden != null);
    try std.testing.expect(!dst_entry.hidden_valid);
}

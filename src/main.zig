//! peregrine CLI: serve the Qwen3.5-9B q4 checkpoint.

const std = @import("std");
const Io = std.Io;
const peregrine = @import("peregrine");

pub const std_options: std.Options = .{ .log_level = .info };

const USAGE =
    \\usage:
    \\  peregrine serve <dir> [--host H] [--port P] [--agent-optimized] [--max-total-tokens N] [--default-new-tokens N] [--max-new-tokens N] [--prefix-cache-tokens N] [--prefill-chunk-tokens N] [--prefill-chunk-group N] [--max-active-connections N] [--socket-timeout-s N] [--prewarm-prompt-file FILE] [--prewarm-request-file FILE] [--prefix-state-file FILE] [--no-prefix-state] [--mtp-dir DIR]
    \\  peregrine mtp-bench <target-dir> <mtp-dir> [--prompt TEXT|--prompt-file FILE] [--tokens N]
    \\
;

const default_host = "127.0.0.1";
const default_port: u16 = 8080;
const agent_optimized_total_tokens: usize = peregrine.model.MAX_CONTEXT_TOKENS;
const agent_optimized_default_new_tokens: u32 = 512;
const agent_optimized_max_new_tokens: u32 = 4096;
const agent_optimized_prefix_cache_tokens: usize = 24 * 1024;
const agent_optimized_prefill_chunk_tokens: usize = peregrine.model.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS;
const agent_optimized_prefill_chunk_group_size: usize = peregrine.model.DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE;

const ServeArgs = struct {
    model_dir: []const u8,
    host: []const u8 = default_host,
    port: u16 = default_port,
    options: peregrine.server.Options = .{},
};

const MtpBenchArgs = struct {
    target_dir: []const u8,
    mtp_dir: []const u8,
    prompt: []const u8 = "The quick brown fox jumps over the lazy dog.",
    prompt_file: ?[]const u8 = null,
    tokens: u32 = 32,
};

const AgentOptimizedOverrides = struct {
    max_total_tokens: ?usize = null,
    default_new_tokens: ?u32 = null,
    max_new_tokens: ?u32 = null,
    prefix_cache_tokens: ?usize = null,
    prefill_chunk_tokens: ?usize = null,
    prefill_chunk_group_size: ?usize = null,

    fn apply(self: AgentOptimizedOverrides, options: *peregrine.server.Options) void {
        if (self.max_total_tokens == null) options.max_total_tokens = agent_optimized_total_tokens;
        if (self.default_new_tokens == null) options.default_new_tokens = agent_optimized_default_new_tokens;
        if (self.max_new_tokens == null) options.max_new_tokens = agent_optimized_max_new_tokens;
        if (self.prefix_cache_tokens == null) options.prefix_cache_tokens = agent_optimized_prefix_cache_tokens;
        if (self.prefill_chunk_tokens == null) options.prefill_chunk_tokens = agent_optimized_prefill_chunk_tokens;
        if (self.prefill_chunk_group_size == null) options.prefill_chunk_group_size = agent_optimized_prefill_chunk_group_size;
    }
};

fn nextArg(argv: []const []const u8, index: *usize) error{Usage}![]const u8 {
    index.* += 1;
    if (index.* >= argv.len) return error.Usage;
    return argv[index.*];
}

fn nextInt(comptime T: type, argv: []const []const u8, index: *usize) error{ Usage, BadNumber }!T {
    return std.fmt.parseInt(T, try nextArg(argv, index), 10) catch return error.BadNumber;
}

/// Parse `serve <model_dir> [--host H] [--port P] [--max-total-tokens N] [--max-new-tokens N]`.
fn parseServe(argv: []const []const u8) error{ Usage, BadNumber }!ServeArgs {
    if (argv.len < 3) return error.Usage;
    var out = ServeArgs{ .model_dir = argv[2] };
    var agent_optimized = false;
    var overrides = AgentOptimizedOverrides{};
    var i: usize = 3;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--agent-optimized")) {
            agent_optimized = true;
        } else if (std.mem.eql(u8, argv[i], "--host")) {
            out.host = try nextArg(argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--port")) {
            out.port = try nextInt(u16, argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--max-total-tokens")) {
            out.options.max_total_tokens = try nextInt(usize, argv, &i);
            overrides.max_total_tokens = out.options.max_total_tokens;
        } else if (std.mem.eql(u8, argv[i], "--default-new-tokens")) {
            out.options.default_new_tokens = try nextInt(u32, argv, &i);
            overrides.default_new_tokens = out.options.default_new_tokens;
        } else if (std.mem.eql(u8, argv[i], "--max-new-tokens")) {
            out.options.max_new_tokens = try nextInt(u32, argv, &i);
            overrides.max_new_tokens = out.options.max_new_tokens;
        } else if (std.mem.eql(u8, argv[i], "--prefix-cache-tokens")) {
            out.options.prefix_cache_tokens = try nextInt(usize, argv, &i);
            overrides.prefix_cache_tokens = out.options.prefix_cache_tokens;
        } else if (std.mem.eql(u8, argv[i], "--prefill-chunk-tokens")) {
            out.options.prefill_chunk_tokens = try nextInt(usize, argv, &i);
            overrides.prefill_chunk_tokens = out.options.prefill_chunk_tokens;
        } else if (std.mem.eql(u8, argv[i], "--prefill-chunk-group")) {
            out.options.prefill_chunk_group_size = try nextInt(usize, argv, &i);
            overrides.prefill_chunk_group_size = out.options.prefill_chunk_group_size;
        } else if (std.mem.eql(u8, argv[i], "--max-active-connections")) {
            out.options.max_active_connections = try nextInt(usize, argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--socket-timeout-s")) {
            out.options.socket_timeout_seconds = try nextInt(u32, argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--prewarm-prompt-file")) {
            out.options.prewarm_prompt_file = try nextArg(argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--prewarm-request-file")) {
            out.options.prewarm_request_file = try nextArg(argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--prefix-state-file")) {
            out.options.prefix_state_file = try nextArg(argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--no-prefix-state")) {
            out.options.persist_prefix_state = false;
        } else if (std.mem.eql(u8, argv[i], "--mtp-dir")) {
            out.options.mtp_dir = try nextArg(argv, &i);
        } else return error.Usage;
    }
    if (agent_optimized) {
        overrides.apply(&out.options);
    }
    if (overrides.max_new_tokens != null and overrides.default_new_tokens == null and out.options.default_new_tokens > out.options.max_new_tokens) {
        out.options.default_new_tokens = out.options.max_new_tokens;
    }
    out.options.validate() catch return error.Usage;
    return out;
}

fn parseMtpBench(argv: []const []const u8) error{ Usage, BadNumber }!MtpBenchArgs {
    if (argv.len < 4) return error.Usage;
    var out = MtpBenchArgs{
        .target_dir = argv[2],
        .mtp_dir = argv[3],
    };
    var i: usize = 4;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--prompt")) {
            out.prompt = try nextArg(argv, &i);
            out.prompt_file = null;
        } else if (std.mem.eql(u8, argv[i], "--prompt-file")) {
            out.prompt_file = try nextArg(argv, &i);
        } else if (std.mem.eql(u8, argv[i], "--tokens")) {
            out.tokens = try nextInt(u32, argv, &i);
        } else return error.Usage;
    }
    if (out.tokens < 3) return error.Usage;
    return out;
}

test "serve args parse agent optimized defaults" {
    const args = [_][]const u8{
        "peregrine",
        "serve",
        "/tmp/qwen3.5-9b-q4",
        "--agent-optimized",
    };
    const parsed = try parseServe(&args);

    try std.testing.expectEqualStrings("/tmp/qwen3.5-9b-q4", parsed.model_dir);
    try std.testing.expectEqual(@as(usize, peregrine.model.MAX_CONTEXT_TOKENS), parsed.options.max_total_tokens);
    try std.testing.expectEqual(@as(u32, agent_optimized_default_new_tokens), parsed.options.default_new_tokens);
    try std.testing.expectEqual(@as(u32, agent_optimized_max_new_tokens), parsed.options.max_new_tokens);
    try std.testing.expectEqual(@as(usize, agent_optimized_prefix_cache_tokens), parsed.options.prefix_cache_tokens);
    try std.testing.expectEqual(@as(usize, agent_optimized_prefill_chunk_tokens), parsed.options.prefill_chunk_tokens);
    try std.testing.expectEqual(@as(usize, agent_optimized_prefill_chunk_group_size), parsed.options.prefill_chunk_group_size);
}

test "serve args keep explicit agent optimized overrides" {
    const args = [_][]const u8{
        "peregrine",
        "serve",
        "/tmp/qwen3.5-9b-q4",
        "--agent-optimized",
        "--max-total-tokens",
        "32768",
        "--default-new-tokens",
        "128",
        "--max-new-tokens",
        "1024",
        "--prefix-cache-tokens",
        "8192",
        "--prefill-chunk-tokens",
        "800",
        "--prefill-chunk-group",
        "4",
    };
    const parsed = try parseServe(&args);

    try std.testing.expectEqual(@as(usize, 32_768), parsed.options.max_total_tokens);
    try std.testing.expectEqual(@as(u32, 128), parsed.options.default_new_tokens);
    try std.testing.expectEqual(@as(u32, 1_024), parsed.options.max_new_tokens);
    try std.testing.expectEqual(@as(usize, 8_192), parsed.options.prefix_cache_tokens);
    try std.testing.expectEqual(@as(usize, 800), parsed.options.prefill_chunk_tokens);
    try std.testing.expectEqual(@as(usize, 4), parsed.options.prefill_chunk_group_size);
}

fn exitUsage() noreturn {
    std.debug.print(USAGE, .{});
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io; // process-provided Threaded Io (no need for a second one)

    const raw = try init.minimal.args.toSlice(arena);
    const argv = try arena.alloc([]const u8, raw.len);
    for (raw, argv) |src, *dst| dst.* = src;

    // `peregrine serve <model_dir> [--host H] [--port P] [--max-total-tokens N] [--max-new-tokens N]`
    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "serve")) {
        const sv = parseServe(argv) catch {
            exitUsage();
        };
        return peregrine.server.run(gpa, io, sv.model_dir, sv.host, sv.port, sv.options);
    }

    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "mtp-bench")) {
        const bench = parseMtpBench(argv) catch {
            exitUsage();
        };
        return runMtpBench(gpa, arena, io, bench);
    }

    exitUsage();
}

fn runMtpBench(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: Io, args: MtpBenchArgs) !void {
    try peregrine.config.verifyFingerprint(gpa, io, args.target_dir);
    try peregrine.mtp.verifyFingerprint(gpa, io, args.mtp_dir);

    const prompt = if (args.prompt_file) |path|
        try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(4 * 1024 * 1024))
    else
        args.prompt;

    var target_repo = try peregrine.safetensors.Repository.open(gpa, io, args.target_dir);
    defer target_repo.deinit();
    var mtp_repo = try peregrine.safetensors.Repository.open(gpa, io, args.mtp_dir);
    defer mtp_repo.deinit();
    var device = try peregrine.metal.Device.create();
    defer device.destroy();
    var queue = try device.createQueue();
    defer queue.destroy();

    const target = try gpa.create(peregrine.model.DeviceModel);
    errdefer gpa.destroy(target);
    target.* = try peregrine.model.DeviceModel.upload(&device, &queue, &target_repo);
    defer {
        target.deinit();
        gpa.destroy(target);
    }

    var drafter = try peregrine.mtp.Drafter.upload(&device, &queue, target.library, &mtp_repo);
    defer drafter.deinit();

    var tok = try peregrine.tokenizer.Tokenizer.load(gpa, io, args.target_dir);
    defer tok.deinit();
    const prompt_ids = try tok.encode(gpa, prompt);
    defer gpa.free(prompt_ids);
    if (prompt_ids.len < 2) return error.EmptyInput;
    if (prompt_ids.len > peregrine.model.MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.PrefillChunkTooLarge;

    const one_hidden_bytes = try hiddenRowsBytes(1);
    const prompt_hidden_bytes = try hiddenRowsBytes(@intCast(prompt_ids.len));

    const max_seq = std.math.add(usize, prompt_ids.len, @as(usize, args.tokens) + 4) catch return error.SequenceTooLong;
    var baseline_state = try peregrine.model.ModelState.initBf16FullCaches(&device, max_seq);
    defer baseline_state.deinit();
    var spec_state = try peregrine.model.ModelState.initBf16FullCaches(&device, max_seq);
    defer spec_state.deinit();
    var mtp_state = try peregrine.mtp.State.initBf16(&device, max_seq);
    defer mtp_state.deinit();

    var baseline_next_token = try device.createSharedBuffer(@sizeOf(u32));
    defer baseline_next_token.destroy();
    var baseline_hidden = try device.createPrivateBuffer(one_hidden_bytes);
    defer baseline_hidden.destroy();
    var baseline_prompt_hidden = try device.createPrivateBuffer(prompt_hidden_bytes);
    defer baseline_prompt_hidden.destroy();

    try target.forwardPrefillBatchNextTokenAndHiddenWithPrefix(
        &device,
        &queue,
        prompt_ids,
        0,
        0,
        &baseline_state,
        null,
        baseline_next_token,
        baseline_prompt_hidden,
    );

    var baseline_gpu_ns_total: u128 = 0;
    var baseline_current = baseline_next_token.slice(u32)[0];
    var baseline_tokens: [2]u32 = undefined;
    for (&baseline_tokens, 0..) |*out_token, i| {
        const logical_pos = prompt_ids.len + i;
        const gpu_ns = try target.forwardNextTokenBf16WithPrefixAndHiddenProfiled(
            &device,
            &queue,
            baseline_current,
            @intCast(logical_pos),
            @intCast(logical_pos),
            @intCast(logical_pos + 1),
            &baseline_state,
            null,
            baseline_next_token,
            baseline_hidden,
        );
        baseline_gpu_ns_total += gpu_ns orelse 0;
        baseline_current = baseline_next_token.slice(u32)[0];
        out_token.* = baseline_current;
    }

    var spec_next_token = try device.createSharedBuffer(@sizeOf(u32));
    defer spec_next_token.destroy();
    var draft_token = try device.createSharedBuffer(@sizeOf(u32));
    defer draft_token.destroy();
    var verify_token = try device.createSharedBuffer(@sizeOf(u32));
    defer verify_token.destroy();
    var bonus_token = try device.createSharedBuffer(@sizeOf(u32));
    defer bonus_token.destroy();
    var spec_prompt_hidden = try device.createPrivateBuffer(prompt_hidden_bytes);
    defer spec_prompt_hidden.destroy();

    try target.forwardPrefillBatchNextTokenAndHiddenWithPrefix(
        &device,
        &queue,
        prompt_ids,
        0,
        0,
        &spec_state,
        null,
        spec_next_token,
        spec_prompt_hidden,
    );
    try drafter.forwardPrefill(
        &device,
        &queue,
        target,
        prompt_ids[1..],
        spec_prompt_hidden,
        &mtp_state,
        0,
        0,
    );

    const current_token = spec_next_token.slice(u32)[0];
    const logical_pos = prompt_ids.len;
    const mtp_cache_pos: u32 = @intCast(logical_pos - 1);
    const mtp_gpu_ns = try drafter.forwardDraftTokenProfiled(
        &device,
        &queue,
        target,
        current_token,
        spec_prompt_hidden,
        @intCast(prompt_ids.len - 1),
        &mtp_state,
        mtp_cache_pos,
        mtp_cache_pos,
        @intCast(mtp_cache_pos + 1),
        draft_token,
    );

    const drafted = draft_token.slice(u32)[0];
    const verify_gpu_ns = try target.forwardTwoTokenDecodeVerifierBf16WithPrefixProfiled(
        &device,
        &queue,
        .{ current_token, drafted },
        @intCast(logical_pos),
        @intCast(logical_pos),
        @intCast(logical_pos + 1),
        &spec_state,
        null,
        verify_token,
        bonus_token,
        null,
    );

    const verified = verify_token.slice(u32)[0];
    const bonus = bonus_token.slice(u32)[0];
    const first_draft_accepted = drafted == verified;
    const verify_token_matches_baseline = verified == baseline_tokens[0];
    const bonus_matches_baseline_if_accepted = first_draft_accepted and bonus == baseline_tokens[1];
    const mtp_ms = @as(f64, @floatFromInt(mtp_gpu_ns orelse 0)) / 1_000_000.0;
    const verify_ms = @as(f64, @floatFromInt(verify_gpu_ns orelse 0)) / 1_000_000.0;
    const spec_ms = mtp_ms + verify_ms;
    const baseline_ms = @as(f64, @floatFromInt(baseline_gpu_ns_total)) / 1_000_000.0;
    std.debug.print(
        \\mtp-bench
        \\  mode: two_token_verify_probe
        \\  prompt_tokens: {d}
        \\  requested_tokens: {d}
        \\  current_token: {d}
        \\  drafted_token: {d}
        \\  verifier_token: {d}
        \\  bonus_token: {d}
        \\  baseline_token_1: {d}
        \\  baseline_token_2: {d}
        \\  first_draft_accepted: {}
        \\  verify_token_matches_baseline: {}
        \\  bonus_matches_baseline_if_accepted: {}
        \\  baseline_two_decode_gpu_ms: {d:.3}
        \\  mtp_draft_gpu_ms: {d:.3}
        \\  two_token_decode_verifier_gpu_ms: {d:.3}
        \\  mtp_plus_two_token_verify_gpu_ms: {d:.3}
        \\  accepted_step_speedup_if_accepted: {d:.3}x
        \\
    , .{
        prompt_ids.len,
        args.tokens,
        current_token,
        drafted,
        verified,
        bonus,
        baseline_tokens[0],
        baseline_tokens[1],
        first_draft_accepted,
        verify_token_matches_baseline,
        bonus_matches_baseline_if_accepted,
        baseline_ms,
        mtp_ms,
        verify_ms,
        spec_ms,
        if (first_draft_accepted and spec_ms != 0) baseline_ms / spec_ms else 0,
    });
}

const hiddenRowsBytes = peregrine.model.dims.hiddenRowsBytes;

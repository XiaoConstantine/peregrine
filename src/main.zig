//! peregrine CLI: serve the Qwen3.5-9B q4 checkpoint.

const std = @import("std");
const Io = std.Io;
const peregrine = @import("peregrine");

pub const std_options: std.Options = .{ .log_level = .info };

const USAGE =
    \\usage:
    \\  peregrine serve <dir> [--host H] [--port P] [--agent-optimized] [--max-total-tokens N] [--default-new-tokens N] [--max-new-tokens N] [--prefix-cache-tokens N] [--prefill-chunk-tokens N] [--prefill-chunk-group N] [--max-active-connections N] [--socket-timeout-s N] [--prewarm-prompt-file FILE] [--prewarm-request-file FILE] [--prefix-state-file FILE] [--no-prefix-state]
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

    exitUsage();
}

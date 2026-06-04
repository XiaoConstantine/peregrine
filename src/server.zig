//! Minimal HTTP server for Qwen3.5-9B q4 OpenAI-compatible local inference.
//! Connections are handled on a bounded set of detached worker threads. The
//! shared model/runtime state is protected so only one decode uses the Metal
//! scratch pool at a time.
//! File-size exception: this remains the single minimal HTTP facade until the
//! serving surface stabilizes; split by route family before adding new surface.
const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const http = std.http;
const metal = @import("runtime/metal.zig");
const config = @import("model/config.zig");
const safetensors = @import("model/safetensors.zig");
const model_mod = @import("model/model.zig");
const prefill_mod = @import("model/prefill.zig");
const generation_limits = @import("server/generation_limits.zig");
const api_error = @import("api/error.zig");
const api_request = @import("api/request.zig");
const api_response = @import("api/response.zig");
const prefix_matching = @import("server/prefix_match.zig");
const decode_loop = @import("server/decode_loop.zig");
const errors = @import("server/errors.zig");
const prefix_state_cache = @import("server/prefix_state_cache.zig");
const prefix_persist = @import("server/prefix_persist.zig");
const stop_sequences = @import("stop_sequences.zig");
const DeviceModel = model_mod.DeviceModel;
const ModelState = model_mod.ModelState;
const Tokenizer = @import("model/tokenizer.zig").Tokenizer;
const Completion = api_response.Completion;
const log = std.log.scoped(.peregrine_serve);
pub const Options = struct {
    max_total_tokens: usize = 8192,
    default_new_tokens: u32 = 512,
    max_new_tokens: u32 = 512,
    prefix_cache_tokens: usize = 8192,
    prefill_chunk_tokens: usize = model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS,
    prefill_chunk_group_size: usize = model_mod.DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE,
    max_active_connections: usize = 32,
    socket_timeout_seconds: u32 = 30,
    prewarm_prompt_file: ?[]const u8 = null,
    prewarm_request_file: ?[]const u8 = null,
    /// Persist the startup-prewarmed prefix state to disk and reload it on the
    /// next start, so a fresh process serves the warmed agent prefix without
    /// recomputing the ~minute-long raw cold prefill.
    persist_prefix_state: bool = true,
    /// Override for the persisted prefix-state file; null uses
    /// `$HOME/.cache/peregrine/` with the one-model default file name.
    prefix_state_file: ?[]const u8 = null,

    pub fn validate(self: Options) !void {
        if (self.max_total_tokens == 0 or
            self.max_total_tokens > model_mod.MAX_CONTEXT_TOKENS or
            self.default_new_tokens == 0 or
            self.max_new_tokens == 0 or
            self.prefix_cache_tokens == 0 or
            self.prefill_chunk_tokens == 0 or
            self.prefill_chunk_tokens > model_mod.MAX_BATCHED_PREFILL_CHUNK_TOKENS or
            self.prefill_chunk_group_size == 0 or
            self.prefill_chunk_group_size > model_mod.MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE or
            self.default_new_tokens > self.max_new_tokens or
            self.max_new_tokens > self.max_total_tokens or
            self.max_active_connections == 0 or
            self.socket_timeout_seconds == 0 or
            self.socket_timeout_seconds > MAX_SOCKET_TIMEOUT_SECONDS)
        {
            return error.InvalidServerOptions;
        }
    }
};
const MAX_BODY = 1 << 20; // 1 MiB request-body ceiling
const MAX_PREWARM_PROMPT_BYTES = 1 << 20;
pub const MAX_SOCKET_TIMEOUT_SECONDS: u32 = 3600;
const health_body =
    "peregrine ok - GET /health, /v1/me, /v1/models, /v1/models/{id}, /v1/prefix/status; " ++
    "POST /v1/chat/completions, /v1/prefix/warmup\n";
const health_headers = api_error.cors_headers ++ [_]http.Header{
    .{ .name = "content-type", .value = "text/plain; charset=utf-8" },
};

const PostRoute = enum {
    chat,
    prefix_warmup,
};

fn postRouteForPath(path: []const u8) ?PostRoute {
    if (std.mem.eql(u8, path, "/v1/chat/completions")) return .chat;
    if (std.mem.eql(u8, path, "/v1/prefix/warmup")) return .prefix_warmup;
    return null;
}

fn modelIdFromPath(path: []const u8) ?[]const u8 {
    const prefix = "/v1/models/";
    if (!std.mem.startsWith(u8, path, prefix)) return null;
    return path[prefix.len..];
}

const Context = struct {
    io: Io,
    request_gpa: std.mem.Allocator,
    device: *metal.Device,
    queue: *metal.Queue,
    model: *DeviceModel,
    tok: *const Tokenizer,
    max_total: usize, // prompt + completion ceiling (KV capacity)
    default_new: u32, // request default when JSON omits output-token fields
    max_new_cap: u32, // per-request completion ceiling
    prefill_chunk_tokens: usize,
    prefill_chunk_group_size: usize,
    prefix_cache: prefix_state_cache.Cache,
    connection_slots: Io.Semaphore,
    socket_timeout_seconds: u32,
    decode_lock: Io.Mutex = .init,
    trace_prefill: bool,
};

pub fn run(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, host: []const u8, port: u16, options: Options) !void {
    try options.validate();
    try config.verifyFingerprint(gpa, io, model_dir); // reject non-Qwen3.5-9B checkpoints up front
    log.info("loading {s}", .{model_dir});
    var repo = try safetensors.Repository.open(gpa, io, model_dir);
    defer repo.deinit();
    var device = try metal.Device.create();
    defer device.destroy();
    var queue = try device.createQueue();
    defer queue.destroy();
    const model = try gpa.create(DeviceModel);
    errdefer gpa.destroy(model);
    model.* = try DeviceModel.upload(&device, &queue, &repo);
    defer {
        model.deinit();
        gpa.destroy(model);
    }
    var tok = try Tokenizer.load(gpa, io, model_dir);
    defer tok.deinit();

    var ctx: Context = .{
        .io = io,
        .request_gpa = std.heap.smp_allocator,
        .device = &device,
        .queue = &queue,
        .model = model,
        .tok = &tok,
        .max_total = options.max_total_tokens,
        .default_new = options.default_new_tokens,
        .max_new_cap = options.max_new_tokens,
        .prefill_chunk_tokens = options.prefill_chunk_tokens,
        .prefill_chunk_group_size = options.prefill_chunk_group_size,
        .prefix_cache = try prefix_state_cache.Cache.init(@min(options.prefix_cache_tokens, options.max_total_tokens), options.prefill_chunk_tokens),
        .connection_slots = .{ .permits = options.max_active_connections },
        .socket_timeout_seconds = options.socket_timeout_seconds,
        .trace_prefill = std.c.getenv("PEREGRINE_TRACE_PREFILL") != null,
    };
    defer ctx.prefix_cache.deinit(ctx.request_gpa);

    var default_prefix_state_path: ?[]u8 = null;
    defer if (default_prefix_state_path) |p| ctx.request_gpa.free(p);
    const prefix_state_path: ?[]const u8 = if (!options.persist_prefix_state)
        null
    else if (options.prefix_state_file) |path|
        path
    else blk: {
        default_prefix_state_path = try prefix_persist.resolveDefaultPath(ctx.request_gpa);
        break :blk default_prefix_state_path;
    };
    const prefix_persist_state = if (prefix_state_path) |path|
        try prefix_persist.loadAtStartup(io, ctx.request_gpa, &device, &ctx.prefix_cache, path, model_dir, model.vocab)
    else
        null;

    var prewarm_computed_tokens: usize = 0;
    if (options.prewarm_prompt_file) |path| {
        prewarm_computed_tokens += try startupPrewarmFile(io, &ctx, path, .prompt);
    }
    if (options.prewarm_request_file) |path| {
        prewarm_computed_tokens += try startupPrewarmFile(io, &ctx, path, .request);
    }
    if (prefix_persist_state) |state| {
        prefix_persist.saveAfterPrewarm(io, &ctx.prefix_cache, state, prewarm_computed_tokens);
    }
    const address = try net.IpAddress.parse(host, port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.close(io);
    log.info(
        "listening on http://{s}:{d} (max_total_tokens={d}, default_new_tokens={d}, max_new_tokens={d}, prefix_cache_tokens={d}, prefill_chunk_tokens={d}, prefill_chunk_group_size={d}, max_active_connections={d}, socket_timeout_s={d})",
        .{ host, port, ctx.max_total, ctx.default_new, ctx.max_new_cap, ctx.prefix_cache.max_tokens, ctx.prefill_chunk_tokens, ctx.prefill_chunk_group_size, options.max_active_connections, options.socket_timeout_seconds },
    );

    while (true) {
        const stream = server.accept(io) catch |e| {
            std.log.warn("accept failed: {}", .{e});
            continue;
        };
        ctx.connection_slots.waitUncancelable(io);
        const thread = std.Thread.spawn(.{}, connectionWorker, .{ io, stream, &ctx }) catch |e| {
            std.log.warn("spawn connection worker failed: {}", .{e});
            ctx.connection_slots.post(io);
            var s = stream;
            s.close(io);
            continue;
        };
        thread.detach();
    }
}
fn connectionWorker(io: Io, stream: net.Stream, ctx: *Context) void {
    defer ctx.connection_slots.post(io);
    handleConnection(io, stream, ctx) catch |e| {
        std.log.warn("connection error: {}", .{e});
    };
}

fn handleConnection(io: Io, stream: net.Stream, ctx: *Context) !void {
    var s = stream;
    defer s.close(io);
    try applyStreamTimeouts(&s, ctx.socket_timeout_seconds);
    var recv: [16 * 1024]u8 = undefined;
    var send: [16 * 1024]u8 = undefined;
    var reader = s.reader(io, &recv);
    var writer = s.writer(io, &send);
    var srv = http.Server.init(&reader.interface, &writer.interface);

    while (true) {
        var request = srv.receiveHead() catch |e| switch (e) {
            error.HttpConnectionClosing => return,
            else => return e,
        };
        try serveRequest(&request, ctx);
    }
}

fn applyStreamTimeouts(stream: *const net.Stream, seconds: u32) !void {
    var timeout: std.c.timeval = .{
        .sec = @intCast(seconds),
        .usec = 0,
    };
    const bytes = std.mem.asBytes(&timeout);
    try std.posix.setsockopt(stream.socket.handle, std.c.SOL.SOCKET, std.c.SO.RCVTIMEO, bytes);
    try std.posix.setsockopt(stream.socket.handle, std.c.SOL.SOCKET, std.c.SO.SNDTIMEO, bytes);
}

fn serveRequest(req: *http.Server.Request, ctx: *Context) !void {
    const t = req.head.target;
    const path = t[0 .. std.mem.indexOfScalar(u8, t, '?') orelse t.len];

    if (req.head.method == .OPTIONS) {
        try api_error.respondOptions(req);
        return;
    }

    if (req.head.method == .GET) {
        return serveGet(req, ctx, path);
    }

    if (std.mem.eql(u8, path, "/v1/me")) {
        try errors.respondMethodNotAllowed(ctx.request_gpa, req);
        return;
    }
    if (modelIdFromPath(path)) |id| {
        if (req.head.method == .DELETE) {
            if (!api_response.isServedModelId(id)) {
                try errors.respondModelNotFound(ctx.request_gpa, req);
                return;
            }
            try errors.respondUnsupportedOpenAIEndpoint(ctx.request_gpa, req);
            return;
        }
        try errors.respondMethodNotAllowed(ctx.request_gpa, req);
        return;
    }
    if (errors.isKnownUnsupportedOpenAIPath(path)) {
        try errors.respondUnsupportedOpenAIEndpoint(ctx.request_gpa, req);
        return;
    }
    if (req.head.method != .POST) {
        try errors.respondMethodNotAllowed(ctx.request_gpa, req);
        return;
    }

    const route = postRouteForPath(path) orelse {
        try errors.respondNotFound(ctx.request_gpa, req);
        return;
    };

    const body = try readLimitedBody(req, ctx) orelse return;
    defer ctx.request_gpa.free(body);

    try servePost(req, ctx, route, body);
}

fn serveGet(req: *http.Server.Request, ctx: *Context, path: []const u8) !void {
    if (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/health")) {
        try req.respond(health_body, .{ .keep_alive = true, .extra_headers = &health_headers });
    } else if (std.mem.eql(u8, path, "/v1/me")) {
        const body = try api_response.buildMeResponse(ctx.request_gpa);
        defer ctx.request_gpa.free(body);
        try req.respond(body, .{ .keep_alive = true, .extra_headers = &api_response.json_ct });
    } else if (std.mem.eql(u8, path, "/v1/models")) {
        const body = try api_response.buildModelListResponse(ctx.request_gpa);
        defer ctx.request_gpa.free(body);
        try req.respond(body, .{ .keep_alive = true, .extra_headers = &api_response.json_ct });
    } else if (modelIdFromPath(path)) |id| {
        const body = api_response.buildModelResponse(ctx.request_gpa, id) catch |e| switch (e) {
            error.ModelNotFound => return errors.respondModelNotFound(ctx.request_gpa, req),
            else => return e,
        };
        defer ctx.request_gpa.free(body);
        try req.respond(body, .{ .keep_alive = true, .extra_headers = &api_response.json_ct });
    } else if (std.mem.eql(u8, path, "/v1/prefix/status")) {
        try respondPrefixStatus(req, ctx);
    } else if (std.mem.eql(u8, path, "/v1/chat/completions")) {
        try errors.respondUnsupportedOpenAIEndpoint(ctx.request_gpa, req);
    } else if (postRouteForPath(path) != null) {
        try errors.respondMethodNotAllowed(ctx.request_gpa, req);
    } else if (errors.isKnownUnsupportedOpenAIPath(path)) {
        try errors.respondUnsupportedOpenAIEndpoint(ctx.request_gpa, req);
    } else {
        try errors.respondNotFound(ctx.request_gpa, req);
    }
}

fn readLimitedBody(req: *http.Server.Request, ctx: *Context) !?[]u8 {
    if (req.head.content_length) |cl| if (cl > MAX_BODY) {
        try errors.respondRequestBodyTooLarge(ctx.request_gpa, req);
        return null;
    };
    var body_buf: [64 * 1024]u8 = undefined;
    const body_reader = try req.readerExpectContinue(&body_buf);
    return body_reader.allocRemaining(ctx.request_gpa, .limited(MAX_BODY)) catch |e| switch (e) {
        error.StreamTooLong => {
            try errors.respondRequestBodyTooLarge(ctx.request_gpa, req);
            return null;
        },
        else => return e,
    };
}

fn servePost(req: *http.Server.Request, ctx: *Context, route: PostRoute, body: []const u8) !void {
    switch (route) {
        .chat => try handleChat(req, ctx, body),
        .prefix_warmup => try handlePrefixWarmup(req, ctx, body),
    }
}

fn tokenCaps(ctx: *const Context) errors.TokenCaps {
    return .{
        .default_new = ctx.default_new,
        .max_new_cap = ctx.max_new_cap,
        .max_total = ctx.max_total,
    };
}

fn respondGenError(req: *http.Server.Request, ctx: *const Context, e: anyerror) !void {
    try errors.respondGenError(ctx.request_gpa, req, tokenCaps(ctx), e);
}

fn respondRequestError(req: *http.Server.Request, ctx: *const Context, e: anyerror) !void {
    try errors.respondRequestError(ctx.request_gpa, req, tokenCaps(ctx), e);
}

fn handleChat(req: *http.Server.Request, ctx: *Context, body: []const u8) !void {
    const gpa = ctx.request_gpa;
    const params = api_request.parseChatParams(gpa, body, ctx.default_new, ctx.max_new_cap) catch |e| {
        return respondRequestError(req, ctx, e);
    };
    defer params.deinit(gpa);

    if (params.stream) {
        respondChatSseStreaming(req, ctx, &params) catch |e| return respondGenError(req, ctx, e);
        return;
    }

    const c = complete(ctx, params.prompt, params.cache_prefix_prompt, params.max_new_tokens, params.stop) catch |e| {
        return respondGenError(req, ctx, e);
    };
    defer gpa.free(c.text);

    return api_response.respondChatJson(gpa, req, c, !params.thinking_disabled, params.response_format_validation);
}

fn handlePrefixWarmup(req: *http.Server.Request, ctx: *Context, body: []const u8) !void {
    const gpa = ctx.request_gpa;
    const prompt = api_request.parseWarmupPrompt(gpa, body) catch |e| {
        return respondRequestError(req, ctx, e);
    };
    defer gpa.free(prompt);

    const result = prewarmPromptCache(ctx, prompt) catch |e| {
        return respondGenError(req, ctx, e);
    };

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try out.print(
        gpa,
        "{{\"object\":\"prefix.warmup\",\"prompt_tokens\":{d},\"cached_tokens\":{d},\"reused_tokens\":{d},\"computed_tokens\":{d},\"cached_logits\":{}}}",
        .{ result.prompt_tokens, result.cached_tokens, result.reused_tokens, result.computed_tokens, result.cached_logits },
    );
    try req.respond(out.items, .{ .keep_alive = true, .extra_headers = &api_response.json_ct });
}

fn respondPrefixStatus(req: *http.Server.Request, ctx: *Context) !void {
    if (!ctx.decode_lock.tryLock()) {
        return respondPrefixStatusJson(req, ctx, busyPrefixStatus(ctx));
    }
    defer ctx.decode_lock.unlock(ctx.io);

    return respondPrefixStatusJson(req, ctx, prefixStatus(ctx));
}

const PrefixStatus = struct {
    busy: bool = false,
    cached_tokens: usize,
    max_cache_tokens: usize,
    resident_entries: usize,
    resident_tokens: usize,
    pinned_entries: usize,
    cached_logits: bool,
    hits: usize,
    misses: usize,
    hit_tokens: usize,
    evictions: usize,
    prefill_chunk_tokens: usize,
    prefill_chunk_group_size: usize,
};

fn busyPrefixStatus(ctx: *const Context) PrefixStatus {
    return .{
        .busy = true,
        .cached_tokens = 0,
        .max_cache_tokens = ctx.prefix_cache.max_tokens,
        .resident_entries = 0,
        .resident_tokens = 0,
        .pinned_entries = 0,
        .cached_logits = false,
        .hits = 0,
        .misses = 0,
        .hit_tokens = 0,
        .evictions = 0,
        .prefill_chunk_tokens = ctx.prefill_chunk_tokens,
        .prefill_chunk_group_size = ctx.prefill_chunk_group_size,
    };
}

fn prefixStatus(ctx: *const Context) PrefixStatus {
    return .{
        .busy = false,
        .cached_tokens = ctx.prefix_cache.activeTokenCount(),
        .max_cache_tokens = ctx.prefix_cache.max_tokens,
        .resident_entries = ctx.prefix_cache.entryCount(),
        .resident_tokens = ctx.prefix_cache.resident_tokens,
        .pinned_entries = ctx.prefix_cache.pinnedEntryCount(),
        .cached_logits = ctx.prefix_cache.activeCachedNextTokenValid(),
        .hits = ctx.prefix_cache.hits,
        .misses = ctx.prefix_cache.misses,
        .hit_tokens = ctx.prefix_cache.hit_tokens,
        .evictions = ctx.prefix_cache.evictions,
        .prefill_chunk_tokens = ctx.prefill_chunk_tokens,
        .prefill_chunk_group_size = ctx.prefill_chunk_group_size,
    };
}

fn respondPrefixStatusJson(req: *http.Server.Request, ctx: *Context, status: PrefixStatus) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(ctx.request_gpa);
    if (status.busy) {
        try out.print(
            ctx.request_gpa,
            "{{\"object\":\"prefix.status\",\"busy\":true,\"cached_tokens\":null,\"max_cache_tokens\":{d},\"resident_entries\":null,\"resident_tokens\":null,\"pinned_entries\":null,\"cached_logits\":null,\"hits\":null,\"misses\":null,\"hit_tokens\":null,\"evictions\":null,\"prefill_chunk_tokens\":{d},\"prefill_chunk_group_size\":{d}}}",
            .{ status.max_cache_tokens, status.prefill_chunk_tokens, status.prefill_chunk_group_size },
        );
    } else {
        try out.print(
            ctx.request_gpa,
            "{{\"object\":\"prefix.status\",\"busy\":false,\"cached_tokens\":{d},\"max_cache_tokens\":{d},\"resident_entries\":{d},\"resident_tokens\":{d},\"pinned_entries\":{d},\"cached_logits\":{},\"hits\":{d},\"misses\":{d},\"hit_tokens\":{d},\"evictions\":{d},\"prefill_chunk_tokens\":{d},\"prefill_chunk_group_size\":{d}}}",
            .{ status.cached_tokens, status.max_cache_tokens, status.resident_entries, status.resident_tokens, status.pinned_entries, status.cached_logits, status.hits, status.misses, status.hit_tokens, status.evictions, status.prefill_chunk_tokens, status.prefill_chunk_group_size },
        );
    }
    try req.respond(out.items, .{ .keep_alive = true, .extra_headers = &api_response.json_ct });
}

const StartupPrewarmProgress = struct {
    label: []const u8,
    last_reported: usize = 0,

    fn callback(self: *StartupPrewarmProgress) DeviceModel.GeneratedTokenCallback {
        return .{
            .context = self,
            .emit = emitNoop,
            .progress = progressOpaque,
        };
    }

    fn emitNoop(context: *anyopaque, tokens: []const u32) anyerror!void {
        _ = context;
        _ = tokens;
    }

    fn progressOpaque(context: *anyopaque, done: usize, total: usize) anyerror!void {
        const self: *StartupPrewarmProgress = @ptrCast(@alignCast(context));
        if (done != total and done < self.last_reported + prewarm_progress_token_interval) return;
        self.last_reported = done;
        log.info("prewarming {s} ({d}/{d} prompt tokens)", .{ self.label, done, total });
    }
};

const prewarm_progress_token_interval = 4096;

const StartupPrewarmFileKind = enum {
    prompt,
    request,
};

fn startupPrewarmFile(io: Io, ctx: *Context, path: []const u8, kind: StartupPrewarmFileKind) !usize {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, ctx.request_gpa, .limited(MAX_PREWARM_PROMPT_BYTES));
    defer ctx.request_gpa.free(bytes);
    const prompt = switch (kind) {
        .prompt => bytes,
        .request => try api_request.parseWarmupPromptWithDefaultMode(ctx.request_gpa, bytes, .before_last_user_content),
    };
    defer if (kind == .request) ctx.request_gpa.free(prompt);

    var progress: StartupPrewarmProgress = .{ .label = path };
    const result = try prewarmPromptCacheWithCallback(ctx, prompt, progress.callback());
    log.info(
        "prewarmed prefix cache from {s}{s} (prompt_tokens={d}, cached_tokens={d}, reused_tokens={d}, computed_tokens={d}, cached_logits={})",
        .{
            if (kind == .request) "request " else "",
            path,
            result.prompt_tokens,
            result.cached_tokens,
            result.reused_tokens,
            result.computed_tokens,
            result.cached_logits,
        },
    );
    return result.computed_tokens;
}

fn finishChatSseWithError(stream: *http.BodyWriter, ended: *bool, e: anyerror) !void {
    try api_response.writeSseError(stream, e);
    try api_response.writeSseDone(stream);
    try stream.end();
    ended.* = true;
}

fn respondChatSseStreaming(
    req: *http.Server.Request,
    ctx: *Context,
    params: *const api_request.ChatParams,
) !void {
    const gpa = ctx.request_gpa;
    const ids = try ctx.tok.encode(gpa, params.prompt);
    defer gpa.free(ids);
    if (ids.len == 0) return error.EmptyPrompt;
    const effective_max_new = try generation_limits.effectiveMaxNewTokens(ids.len, params.max_new_tokens, ctx.max_total);

    const gen = try gpa.alloc(u32, effective_max_new);
    defer gpa.free(gen);
    const stop_tokens = try stop_sequences.encode(gpa, ctx.tok, params.stop);
    defer stop_sequences.freeTokenSequences(gpa, stop_tokens);

    var body_buf: [16 * 1024]u8 = undefined;
    var stream = try req.respondStreaming(&body_buf, .{
        .respond_options = .{
            .keep_alive = false,
            .extra_headers = &api_response.sse_ct,
        },
    });
    var ended = false;
    defer if (!ended) {
        stream.end() catch |e| log.warn("failed to end unfinished chat SSE stream: {}", .{e});
    };

    var metadata = try api_response.initChatSseMetadata(gpa);
    defer metadata.deinit(gpa);
    try api_response.writeSseRole(gpa, &stream, metadata);
    try api_response.flushSse(&stream);

    const expose_reasoning = !params.thinking_disabled;
    var sink: api_response.ChatTokenSink = .{ .gpa = gpa, .tok = ctx.tok, .stream = &stream, .metadata = metadata, .stop_sequences = params.stop, .expose_reasoning = expose_reasoning };
    defer sink.deinit();
    const n = generateWithStaticPrefixCache(ctx, ids, params.cache_prefix_prompt, gen, .{
        .context = &sink,
        .emit = api_response.ChatTokenSink.emitOpaque,
        .progress = api_response.ChatTokenSink.progressOpaque,
    }, stop_tokens) catch |e| {
        try finishChatSseWithError(&stream, &ended, e);
        return;
    };
    if (ctx.trace_prefill) {
        log.info(
            "trace: chat stream generated tokens={d} emitted_len={d} pending_bytes={d} assistant_bytes={d}",
            .{ n, sink.emitted_len, sink.pending_bytes.items.len, sink.assistant_bytes.items.len },
        );
    }

    finishChatSse(
        gpa,
        ctx,
        &stream,
        &ended,
        metadata,
        &sink,
        gen[0..n],
        ids.len,
        effective_max_new,
        params.stream_include_usage,
        expose_reasoning,
        params.stop,
        stop_tokens,
    ) catch |e| {
        try finishChatSseWithError(&stream, &ended, e);
        return;
    };
}

fn finishChatSse(
    gpa: std.mem.Allocator,
    ctx: *Context,
    stream: *http.BodyWriter,
    ended: *bool,
    metadata: api_response.ChatSseMetadata,
    sink: *api_response.ChatTokenSink,
    generated: []const u32,
    prompt_tokens: usize,
    effective_max_new: usize,
    include_usage: bool,
    expose_reasoning: bool,
    stop: []const []const u8,
    stop_tokens: []const []const u32,
) !void {
    const text = try ctx.tok.decode(gpa, generated);
    defer gpa.free(text);
    if (ctx.trace_prefill) {
        log.info("trace: chat stream decoded_text_len={d}", .{text.len});
    }
    const stopped = stop_sequences.apply(text, stop);
    const stopped_by_token_sequence = stop_sequences.generatedEndsWith(generated, stop_tokens);

    const c: Completion = .{
        .text = text,
        .prompt_tokens = prompt_tokens,
        .completion_tokens = generated.len,
        .hit_max = generated.len == effective_max_new and !stopped.stopped and !stopped_by_token_sequence,
    };
    var assistant = try api_response.parseAssistantOutputOrContent(gpa, stopped.text, expose_reasoning);
    defer assistant.deinit(gpa);
    try sink.flushFinal();
    if (ctx.trace_prefill) {
        log.info(
            "trace: chat stream flush final emitted_len={d} pending_bytes={d} assistant_bytes={d}",
            .{ sink.emitted_len, sink.pending_bytes.items.len, sink.assistant_bytes.items.len },
        );
    }
    if (assistant.reasoning_content) |reasoning| {
        if (sink.reasoning_emitted_len < reasoning.len) {
            try api_response.writeSseReasoning(gpa, stream, metadata, reasoning[sink.reasoning_emitted_len..]);
        }
    }
    if (assistant.tool_call) |call| {
        if (!sink.emitted_tool_call) try api_response.writeSseToolCall(gpa, stream, metadata, call, 0);
        try api_response.writeSseFinish(gpa, stream, metadata, "tool_calls");
    } else {
        if (!sink.stop_seen and sink.emitted_len < assistant.content.len) {
            try api_response.writeSseContent(gpa, stream, metadata, assistant.content[sink.emitted_len..]);
        }
        try api_response.writeSseFinish(gpa, stream, metadata, api_response.finishReason(c));
    }
    if (include_usage) try api_response.writeSseUsage(gpa, stream, metadata, c);
    try api_response.writeSseDone(stream);
    try stream.end();
    if (ctx.trace_prefill) {
        log.info("trace: chat stream ended", .{});
    }
    ended.* = true;
}

fn generateWithStaticPrefixCache(
    ctx: *Context,
    prompt_ids: []const u32,
    static_prefix_prompt: ?[]const u8,
    out: []u32,
    callback: ?DeviceModel.GeneratedTokenCallback,
    stop_token_sequences: []const []const u32,
) !usize {
    ctx.decode_lock.lockUncancelable(ctx.io);
    defer ctx.decode_lock.unlock(ctx.io);

    const cache_store_limit = try prewarmStaticPromptTokenPrefixLocked(ctx, prompt_ids, static_prefix_prompt, callback);
    return generateWithPrefixCacheLocked(ctx, prompt_ids, out, callback, cache_store_limit, stop_token_sequences);
}

const PrefixReuse = struct {
    match: ?prefix_state_cache.Match,
    reuse_len: usize,
    direct_fill_empty_cache: bool,
};

const InitialNextToken = struct {
    prefix_match: ?prefix_state_cache.Match,
    token: u32,
};

fn generateWithPrefixCacheLocked(
    ctx: *Context,
    prompt_ids: []const u32,
    out: []u32,
    callback: ?DeviceModel.GeneratedTokenCallback,
    cache_store_limit: ?usize,
    stop_token_sequences: []const []const u32,
) !usize {
    if (prompt_ids.len == 0 or out.len == 0) return error.EmptyInput;
    _ = try generation_limits.effectiveMaxNewTokens(prompt_ids.len, @intCast(out.len), ctx.max_total);
    const requested = std.math.add(usize, prompt_ids.len, out.len) catch return error.SequenceTooLong;

    const prefix_reuse = try resolveReusablePrefix(ctx, prompt_ids, callback, cache_store_limit);
    const local_capacity = requested - prefix_reuse.reuse_len;
    if (ctx.trace_prefill) {
        log.info(
            "trace: generate prompt_tokens={d} out_tokens={d} reuse_tokens={d} local_capacity={d} direct_fill_empty_cache={}",
            .{ prompt_ids.len, out.len, prefix_reuse.reuse_len, local_capacity, prefix_reuse.direct_fill_empty_cache },
        );
    }

    var work_state = try ModelState.initBf16FullCaches(ctx.device, local_capacity);
    defer work_state.deinit();
    var work_next_token = try ctx.device.createSharedBuffer(@sizeOf(u32));
    defer work_next_token.destroy();

    const effective_cache_store_limit = effectivePromptCacheStoreLimit(
        cache_store_limit,
        prefix_reuse.direct_fill_empty_cache,
        prefix_reuse.reuse_len,
        ctx.prefix_cache.entryCount(),
    );
    try copyReusablePrefixState(ctx, prefix_reuse, &work_state);
    if (!prefix_reuse.direct_fill_empty_cache) {
        try reportPrefillProgress(callback, prefix_reuse.reuse_len, prompt_ids.len);
    }

    const initial_next_token = try initialNextTokenFromPrefix(
        ctx,
        prompt_ids,
        prefix_reuse.match,
        prefix_reuse.reuse_len,
        &work_state,
        work_next_token,
        callback,
        effective_cache_store_limit,
        false,
    );
    return decode_loop.decodeFromPrefixCache(.{
        .device = ctx.device,
        .queue = ctx.queue,
        .model = ctx.model,
        .eos_id = ctx.tok.eos_id,
        .prompt_len = prompt_ids.len,
        .prefix_len = prefix_reuse.reuse_len,
        .state = &work_state,
        .prefix = ctx.prefix_cache.prefixState(initial_next_token.prefix_match),
        .initial_next_token = initial_next_token.token,
        .next_token = work_next_token,
        .out = out,
        .callback = callback,
        .stop_token_sequences = stop_token_sequences,
        .trace = ctx.trace_prefill,
    });
}

fn resolveReusablePrefix(
    ctx: *Context,
    prompt_ids: []const u32,
    callback: ?DeviceModel.GeneratedTokenCallback,
    cache_store_limit: ?usize,
) !PrefixReuse {
    const cache = &ctx.prefix_cache;
    var prefix_match = cache.findPrefix(prompt_ids);
    if (prefix_match) |match| {
        if (match.len == prompt_ids.len and !cache.cachedNextTokenValid(match)) {
            cache.removeEntry(ctx.request_gpa, match.index);
            prefix_match = cache.findPrefix(prompt_ids);
        }
    }
    const reuse_len = if (prefix_match) |match| match.len else 0;
    const direct_fill_empty_cache = shouldDirectFillEmptyPromptCache(
        cache_store_limit,
        cache.entryCount(),
        reuse_len,
        prompt_ids.len,
        cache.max_tokens,
    );
    if (direct_fill_empty_cache) {
        cache.recordMiss();
        try reportPrefillProgress(callback, 0, prompt_ids.len);
        prefix_match = try prefillStaticPrefixCacheDirect(ctx, prompt_ids, callback, false);
        return .{
            .match = prefix_match,
            .reuse_len = prompt_ids.len,
            .direct_fill_empty_cache = true,
        };
    }
    return .{
        .match = prefix_match,
        .reuse_len = reuse_len,
        .direct_fill_empty_cache = false,
    };
}

fn effectivePromptCacheStoreLimit(
    cache_store_limit: ?usize,
    direct_fill_empty_cache: bool,
    reuse_len: usize,
    cached_entry_count: usize,
) ?usize {
    const preserve_unmatched_cache = !direct_fill_empty_cache and
        reuse_len == 0 and
        shouldPreserveUnmatchedCache(cache_store_limit, cached_entry_count);
    return if (preserve_unmatched_cache) 0 else cache_store_limit;
}

fn copyReusablePrefixState(ctx: *Context, prefix_reuse: PrefixReuse, work_state: *ModelState) !void {
    const cache = &ctx.prefix_cache;
    if (prefix_reuse.direct_fill_empty_cache) {
        try work_state.copyLinearStateFrom(cache.entryState(prefix_reuse.match.?));
    } else if (prefix_reuse.match) |match| {
        try work_state.copyLinearStateFrom(cache.entryState(match));
        cache.touch(match);
        cache.recordHit(match.len);
    } else {
        cache.recordMiss();
    }
}

fn initialNextTokenFromPrefix(
    ctx: *Context,
    prompt_ids: []const u32,
    prefix_match: ?prefix_state_cache.Match,
    reuse_len: usize,
    work_state: *ModelState,
    work_next_token: metal.Buffer,
    callback: ?DeviceModel.GeneratedTokenCallback,
    cache_store_limit: ?usize,
    pinned: bool,
) !InitialNextToken {
    const cache = &ctx.prefix_cache;
    if (prefix_match) |match| {
        if (reuse_len == prompt_ids.len and cache.cachedNextTokenValid(match)) {
            return .{ .prefix_match = match, .token = cache.cachedNextToken(match) };
        }
    }
    const decode_prefix_match = try prefillPromptWithPrefixCache(ctx, prompt_ids, prefix_match, work_state, work_next_token, callback, cache_store_limit, pinned);
    return .{ .prefix_match = decode_prefix_match, .token = work_next_token.slice(u32)[0] };
}

fn reportPrefillProgress(callback: ?DeviceModel.GeneratedTokenCallback, completed: usize, total: usize) !void {
    if (callback) |cb| {
        if (cb.progress) |progress| try progress(cb.context, completed, total);
    }
}

fn prewarmStaticPromptTokenPrefixLocked(
    ctx: *Context,
    prompt_ids: []const u32,
    static_prefix_prompt: ?[]const u8,
    callback: ?DeviceModel.GeneratedTokenCallback,
) !?usize {
    const prefix_prompt = static_prefix_prompt orelse return null;
    const gpa = ctx.request_gpa;
    const static_ids = try ctx.tok.encode(gpa, prefix_prompt);
    defer gpa.free(static_ids);
    const prefix_len = prefix_matching.boundedStaticReuseLen(static_ids, prompt_ids, ctx.prefix_cache.max_tokens, ctx.max_total);
    if (prefix_len == 0) return null;
    if (ctx.prefix_cache.containsPromptPrefix(prompt_ids, prefix_len)) return prefix_len;
    _ = try prewarmPromptCacheLockedWithCallback(ctx, prompt_ids[0..prefix_len], prompt_ids.len, callback, true);
    return prefix_len;
}

fn shouldPreserveUnmatchedCache(cache_store_limit: ?usize, cached_entry_count: usize) bool {
    return cache_store_limit == null and cached_entry_count != 0;
}

fn shouldDirectFillEmptyPromptCache(
    cache_store_limit: ?usize,
    cached_entry_count: usize,
    reuse_len: usize,
    prompt_len: usize,
    max_cache_tokens: usize,
) bool {
    return cache_store_limit == null and
        cached_entry_count == 0 and
        reuse_len == 0 and
        prompt_len != 0 and
        prompt_len <= max_cache_tokens;
}

fn prefillPromptWithPrefixCache(
    ctx: *Context,
    prompt_ids: []const u32,
    prefix_match: ?prefix_state_cache.Match,
    work_state: *ModelState,
    work_next_token: metal.Buffer,
    callback: ?DeviceModel.GeneratedTokenCallback,
    cache_store_limit: ?usize,
    pinned: bool,
) !?prefix_state_cache.Match {
    const cache = &ctx.prefix_cache;
    const prefix_len = if (prefix_match) |match| match.len else 0;
    const max_cache_target = @min(prompt_ids.len, cache.max_tokens);
    const cache_target = if (cache_store_limit) |limit|
        @max(prefix_len, @min(max_cache_target, limit))
    else
        max_cache_target;
    var decode_prefix_match = prefix_match;
    var prefix = cache.prefixState(decode_prefix_match);
    var pos = prefix_len;

    if (pos < cache_target and prefixShouldStore(cache, prompt_ids[0..cache_target], prefix_match)) {
        try prefillPromptSegment(ctx, prompt_ids, work_state, prefix, work_next_token, prefix_len, pos, cache_target, callback);
        pos = cache_target;
        const stored = try storePromptPrefixCache(ctx, prompt_ids[0..cache_target], prefix_match, work_state, work_next_token, pinned);
        if (prefix_len != 0) {
            decode_prefix_match = .{ .index = stored.index, .len = prefix_len };
            prefix = cache.prefixState(decode_prefix_match);
        }
    }

    if (pos < prompt_ids.len) {
        try prefillPromptSegment(ctx, prompt_ids, work_state, prefix, work_next_token, prefix_len, pos, prompt_ids.len, callback);
    }
    return decode_prefix_match;
}

fn prefillPromptSegment(
    ctx: *Context,
    prompt_ids: []const u32,
    work_state: *ModelState,
    prefix: ?model_mod.DeviceModel.PrefixState,
    work_next_token: metal.Buffer,
    prefix_len: usize,
    start_pos: usize,
    end_pos: usize,
    callback: ?DeviceModel.GeneratedTokenCallback,
) !void {
    try prefill_mod.prefillSegment(ctx.model, ctx.device, ctx.queue, prompt_ids, work_state, prefix, work_next_token, .{
        .prefix_len = prefix_len,
        .start_pos = start_pos,
        .end_pos = end_pos,
        .chunk_tokens = ctx.prefill_chunk_tokens,
        .chunk_group_size = ctx.prefill_chunk_group_size,
        .trace = ctx.trace_prefill,
        .callback = callback,
    });
}

fn prefixShouldStore(cache: *const prefix_state_cache.Cache, desired_prefix: []const u32, prefix_match: ?prefix_state_cache.Match) bool {
    if (desired_prefix.len == 0) return false;
    const matched_len = if (prefix_match) |match| match.len else 0;
    if (prefix_match) |match| {
        const entry = &cache.entries.items[match.index];
        return entry.tokens.items.len != desired_prefix.len or matched_len < desired_prefix.len;
    }
    return true;
}

fn storePromptPrefixCache(
    ctx: *Context,
    prefix_ids: []const u32,
    prefix_match: ?prefix_state_cache.Match,
    work_state: *const ModelState,
    next_token: ?metal.Buffer,
    pinned: bool,
) !prefix_state_cache.Match {
    if (prefix_ids.len == 0) return error.EmptyInput;
    return ctx.prefix_cache.storePrefixFromWork(ctx.request_gpa, ctx.device, prefix_ids, prefix_match, work_state, next_token, pinned);
}

const WarmupResult = struct {
    prompt_tokens: usize,
    cached_tokens: usize,
    reused_tokens: usize,
    computed_tokens: usize,
    cached_logits: bool,

    fn init(prompt_tokens: usize, prefix_len: usize, reused_tokens: usize, cached_logits: bool) WarmupResult {
        return .{
            .prompt_tokens = prompt_tokens,
            .cached_tokens = prefix_len,
            .reused_tokens = reused_tokens,
            .computed_tokens = prefix_len - reused_tokens,
            .cached_logits = cached_logits,
        };
    }
};

fn prewarmPromptCache(ctx: *Context, prompt: []const u8) !WarmupResult {
    return prewarmPromptCacheWithCallback(ctx, prompt, null);
}

fn prewarmPromptCacheWithCallback(ctx: *Context, prompt: []const u8, callback: ?DeviceModel.GeneratedTokenCallback) !WarmupResult {
    const gpa = ctx.request_gpa;
    const ids = try ctx.tok.encode(gpa, prompt);
    defer gpa.free(ids);
    if (ids.len == 0) return error.EmptyPrompt;

    const cache_len = @min(ids.len, ctx.prefix_cache.max_tokens, ctx.max_total);
    if (cache_len == 0) return error.EmptyPrompt;

    ctx.decode_lock.lockUncancelable(ctx.io);
    defer ctx.decode_lock.unlock(ctx.io);
    return prewarmPromptCacheLockedWithCallback(ctx, ids[0..cache_len], ids.len, callback, true);
}

fn prewarmPromptCacheLockedWithCallback(
    ctx: *Context,
    prefix_ids: []const u32,
    prompt_tokens: usize,
    callback: ?DeviceModel.GeneratedTokenCallback,
    pinned: bool,
) !WarmupResult {
    const cache = &ctx.prefix_cache;
    const prefix_match = cache.findPrefix(prefix_ids);
    const reuse_len = if (prefix_match) |match| match.len else 0;
    if (prefix_match) |match| {
        if (reuse_len == prefix_ids.len) {
            cache.touch(match);
            cache.recordHit(reuse_len);
            return WarmupResult.init(prompt_tokens, prefix_ids.len, reuse_len, cache.cachedNextTokenValid(match));
        }
    }

    if (prefix_match == null) {
        cache.recordMiss();
        const stored = try prefillStaticPrefixCacheDirect(ctx, prefix_ids, callback, pinned);
        return WarmupResult.init(prompt_tokens, prefix_ids.len, reuse_len, cache.cachedNextTokenValid(stored));
    }

    const local_capacity = prefix_ids.len - reuse_len;
    var work_state = try ModelState.initBf16FullCaches(ctx.device, local_capacity);
    defer work_state.deinit();
    var work_next_token = try ctx.device.createSharedBuffer(@sizeOf(u32));
    defer work_next_token.destroy();

    std.debug.assert(reuse_len > 0);
    const match = prefix_match.?;
    try work_state.copyLinearStateFrom(cache.entryState(match));
    cache.touch(match);
    cache.recordHit(reuse_len);
    const stored = try prefillStaticPrefixCacheOnly(ctx, prefix_ids, match, &work_state, work_next_token, callback, pinned);
    return WarmupResult.init(prompt_tokens, prefix_ids.len, reuse_len, cache.cachedNextTokenValid(stored));
}

fn prefillStaticPrefixCacheDirect(
    ctx: *Context,
    prefix_ids: []const u32,
    callback: ?DeviceModel.GeneratedTokenCallback,
    pinned: bool,
) !prefix_state_cache.Match {
    const cache = &ctx.prefix_cache;
    const match = try cache.startDirectEntry(ctx.request_gpa, ctx.device, prefix_ids, pinned);
    var committed = false;
    errdefer if (!committed) cache.removeEntry(ctx.request_gpa, match.index);
    var next_token = try ctx.device.createSharedBuffer(@sizeOf(u32));
    defer next_token.destroy();
    try prefillPromptSegment(ctx, prefix_ids, cache.entryStateMut(match), null, next_token, 0, 0, prefix_ids.len, callback);
    try cache.finishDirectEntry(match, next_token);
    committed = true;
    return match;
}

fn prefillStaticPrefixCacheOnly(
    ctx: *Context,
    prefix_ids: []const u32,
    prefix_match: prefix_state_cache.Match,
    work_state: *ModelState,
    work_next_token: metal.Buffer,
    callback: ?DeviceModel.GeneratedTokenCallback,
    pinned: bool,
) !prefix_state_cache.Match {
    const prefix_len = prefix_match.len;
    const prefix = ctx.prefix_cache.prefixState(prefix_match);
    if (prefix_len < prefix_ids.len) {
        try prefillPromptSegment(ctx, prefix_ids, work_state, prefix, work_next_token, prefix_len, prefix_len, prefix_ids.len, callback);
    }
    return storePromptPrefixCache(ctx, prefix_ids, prefix_match, work_state, work_next_token, pinned);
}

fn complete(ctx: *Context, prompt: []const u8, cache_prefix_prompt: ?[]const u8, max_new: u32, stop: []const []const u8) !Completion {
    const gpa = ctx.request_gpa;
    const ids = try ctx.tok.encode(gpa, prompt);
    defer gpa.free(ids);
    if (ids.len == 0) return error.EmptyPrompt;
    const effective_max_new = try generation_limits.effectiveMaxNewTokens(ids.len, max_new, ctx.max_total);

    const gen = try gpa.alloc(u32, effective_max_new);
    defer gpa.free(gen);
    const stop_tokens = try stop_sequences.encode(gpa, ctx.tok, stop);
    defer stop_sequences.freeTokenSequences(gpa, stop_tokens);

    const n = try generateWithStaticPrefixCache(ctx, ids, cache_prefix_prompt, gen, null, stop_tokens);
    const decoded = try ctx.tok.decode(gpa, gen[0..n]);
    errdefer gpa.free(decoded);
    const stopped = stop_sequences.apply(decoded, stop);
    const stopped_by_token_sequence = stop_sequences.generatedEndsWith(gen[0..n], stop_tokens);
    const text = if (stopped.text.len == decoded.len) decoded else blk: {
        const trimmed = try gpa.dupe(u8, stopped.text);
        gpa.free(decoded);
        break :blk trimmed;
    };
    return .{
        .text = text,
        .prompt_tokens = ids.len,
        .completion_tokens = n,
        .hit_max = n == effective_max_new and !stopped.stopped and !stopped_by_token_sequence,
    };
}

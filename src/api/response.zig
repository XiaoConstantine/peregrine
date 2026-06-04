//! OpenAI-compatible chat response envelopes and SSE helpers for Peregrine serving.

const std = @import("std");
const http = std.http;

const runtime_time = @import("../runtime/time.zig");
const Tokenizer = @import("../model/tokenizer.zig").Tokenizer;
const api_error = @import("error.zig");
const response_format = @import("../response_format.zig");
const stop_sequences = @import("../stop_sequences.zig");
const utf8_prefix = @import("../utf8_prefix.zig");
const monotonicNowNs = runtime_time.monotonicNowNs;

pub const MODEL_ID = "qwen3.5-9b-4bit";
const me_response =
    "{\"object\":\"user\",\"id\":\"user_peregrine_local\",\"email\":\"local@peregrine.invalid\",\"name\":\"Peregrine Local User\",\"orgs\":{\"object\":\"list\",\"data\":[],\"has_more\":false}}";
const DEFAULT_SERVICE_TIER = "default";

pub const json_ct = api_error.json_ct;
pub const sse_ct = [_]http.Header{
    .{ .name = "content-type", .value = "text/event-stream" },
    .{ .name = "cache-control", .value = "no-cache" },
} ++ api_error.cors_headers;

pub fn isServedModelId(id: []const u8) bool {
    const aliases = [_][]const u8{
        MODEL_ID,
        "qwen3.5-9b-q4",
        "mlx-community/Qwen3.5-9B-4bit",
    };
    for (aliases) |alias| {
        if (std.mem.eql(u8, id, alias)) return true;
        if (percentDecodedEql(id, alias)) return true;
    }
    return false;
}

fn percentDecodedEql(encoded: []const u8, raw: []const u8) bool {
    var encoded_index: usize = 0;
    var raw_index: usize = 0;
    while (encoded_index < encoded.len) {
        if (raw_index >= raw.len) return false;
        const decoded = decodedPathByte(encoded, &encoded_index);
        if (decoded != raw[raw_index]) return false;
        raw_index += 1;
    }
    return raw_index == raw.len;
}

fn decodedPathByte(encoded: []const u8, index: *usize) u8 {
    const start = index.*;
    if (encoded[start] == '%' and start + 2 < encoded.len) {
        if (std.fmt.parseInt(u8, encoded[start + 1 .. start + 3], 16)) |value| {
            index.* = start + 3;
            return value;
        } else |_| {}
    }
    index.* = start + 1;
    return encoded[start];
}

pub const ChatSseMetadata = struct {
    id: []u8,
    created: u64,

    pub fn deinit(self: *ChatSseMetadata, gpa: std.mem.Allocator) void {
        gpa.free(self.id);
        self.* = undefined;
    }
};

pub fn initChatSseMetadata(gpa: std.mem.Allocator) !ChatSseMetadata {
    const created = unixTimestampSeconds();
    return .{
        .id = try allocChatCompletionId(gpa, created),
        .created = created,
    };
}

pub fn buildModelListResponse(gpa: std.mem.Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "{\"object\":\"list\",\"data\":[");
    try appendModelObject(gpa, &out);
    try out.appendSlice(gpa, "]}");
    return out.toOwnedSlice(gpa);
}

pub fn buildModelResponse(gpa: std.mem.Allocator, id: []const u8) ![]u8 {
    if (!isServedModelId(id)) return error.ModelNotFound;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try appendModelObject(gpa, &out);
    return out.toOwnedSlice(gpa);
}

pub fn buildMeResponse(gpa: std.mem.Allocator) ![]u8 {
    return gpa.dupe(u8, me_response);
}

pub const Completion = struct {
    text: []const u8,
    prompt_tokens: usize,
    completion_tokens: usize,
    hit_max: bool,
};

pub const ToolCall = struct {
    id: []u8,
    name: []u8,
    arguments_json: []u8,

    pub fn deinit(self: *ToolCall, gpa: std.mem.Allocator) void {
        gpa.free(self.arguments_json);
        gpa.free(self.name);
        gpa.free(self.id);
        self.* = undefined;
    }
};

/// Parsed assistant output. `content` and `reasoning_content` borrow from the
/// generated text; `tool_call` owns its id/name/arguments and must be deinitialized.
pub const AssistantOutput = struct {
    content: []const u8,
    reasoning_content: ?[]const u8 = null,
    tool_call: ?ToolCall = null,

    pub fn deinit(self: *AssistantOutput, gpa: std.mem.Allocator) void {
        if (self.tool_call) |*call| call.deinit(gpa);
        self.* = undefined;
    }
};

pub const ChatTokenSink = struct {
    gpa: std.mem.Allocator,
    tok: *const Tokenizer,
    stream: *http.BodyWriter,
    metadata: ChatSseMetadata,
    emitted_len: usize = 0,
    reasoning_emitted_len: usize = 0,
    stop_sequences: []const []const u8 = &.{},
    stop_seen: bool = false,
    expose_reasoning: bool = false,
    consumed_token_count: usize = 0,
    emitted_tool_call: bool = false,
    assistant_state: AssistantStreamState = .visible,
    pending_bytes: std.ArrayList(u8) = .empty,
    assistant_bytes: std.ArrayList(u8) = .empty,
    tool_call_bytes: std.ArrayList(u8) = .empty,
    tool_call_count: usize = 0,

    pub fn deinit(self: *ChatTokenSink) void {
        self.tool_call_bytes.deinit(self.gpa);
        self.assistant_bytes.deinit(self.gpa);
        self.pending_bytes.deinit(self.gpa);
        self.* = undefined;
    }

    pub fn emitOpaque(context: *anyopaque, generated_ids: []const u32) anyerror!void {
        const self: *ChatTokenSink = @ptrCast(@alignCast(context));
        if (self.stop_seen) return;
        if (generated_ids.len < self.consumed_token_count) return error.InvalidGeneratedPrefix;
        const decoded = try self.tok.decode(self.gpa, generated_ids[self.consumed_token_count..]);
        defer self.gpa.free(decoded);
        try self.pending_bytes.appendSlice(self.gpa, decoded);
        self.consumed_token_count = generated_ids.len;

        const decision = stop_sequences.streamEmitDecision(self.pending_bytes.items, self.stop_sequences);
        self.stop_seen = decision.stop_seen;
        if (decision.emit_len == 0) return;

        try self.emitDelta(self.pending_bytes.items[0..decision.emit_len]);
        self.discardPendingPrefix(decision.emit_len);
    }

    pub fn progressOpaque(context: *anyopaque, done: usize, total: usize) anyerror!void {
        const self: *ChatTokenSink = @ptrCast(@alignCast(context));
        _ = done;
        _ = total;
        try writeSseKeepalive(self.gpa, self.stream, self.metadata);
        try flushSse(self.stream);
    }

    pub fn flushFinal(self: *ChatTokenSink) !void {
        if (!self.stop_seen) {
            const emit_len = utf8_prefix.prefixLenIncludingFirstInvalidByte(self.pending_bytes.items);
            if (emit_len > 0) {
                try self.emitDelta(self.pending_bytes.items[0..emit_len]);
                self.discardPendingPrefix(emit_len);
            }
        }
        try self.processAssistantBytes(true);
    }

    fn emitDelta(self: *ChatTokenSink, delta: []const u8) !void {
        try self.assistant_bytes.appendSlice(self.gpa, delta);
        try self.processAssistantBytes(false);
    }

    fn emitVisibleDelta(self: *ChatTokenSink, delta: []const u8) !void {
        if (delta.len == 0) return;
        try writeSseContent(self.gpa, self.stream, self.metadata, delta);
        try flushSse(self.stream);
        self.emitted_len += delta.len;
    }

    fn emitReasoningDelta(self: *ChatTokenSink, delta: []const u8) !void {
        if (delta.len == 0) return;
        if (self.expose_reasoning) {
            try writeSseReasoning(self.gpa, self.stream, self.metadata, delta);
            try flushSse(self.stream);
            self.reasoning_emitted_len += delta.len;
        }
    }

    fn emitToolCall(self: *ChatTokenSink, body: []const u8) !void {
        const index = self.tool_call_count;
        var call = try parseToolCallBlock(self.gpa, body, index);
        defer call.deinit(self.gpa);
        try writeSseToolCall(self.gpa, self.stream, self.metadata, call, index);
        try flushSse(self.stream);
        self.tool_call_count += 1;
        self.emitted_tool_call = true;
    }

    fn processAssistantBytes(self: *ChatTokenSink, final: bool) !void {
        while (true) {
            switch (self.assistant_state) {
                .visible => {
                    const marker = nextAssistantMarker(self.assistant_bytes.items) orelse {
                        const hold_len = if (final) 0 else controlMarkerSuffixLen(self.assistant_bytes.items);
                        const emit_len = self.assistant_bytes.items.len - hold_len;
                        if (emit_len > 0) {
                            try self.emitVisibleDelta(self.assistant_bytes.items[0..emit_len]);
                            self.discardAssistantPrefix(emit_len);
                        }
                        return;
                    };
                    if (marker.index > 0) {
                        try self.emitVisibleDelta(self.assistant_bytes.items[0..marker.index]);
                    }
                    self.discardAssistantPrefix(marker.index + marker.text.len);
                    self.assistant_state = marker.state;
                },
                .reasoning => {
                    const close = "</think>";
                    if (std.mem.indexOf(u8, self.assistant_bytes.items, close)) |index| {
                        try self.emitReasoningDelta(self.assistant_bytes.items[0..index]);
                        self.discardAssistantPrefix(index + close.len);
                        self.assistant_state = .visible;
                        continue;
                    }
                    const hold_len = if (final) 0 else suffixPrefixLen(self.assistant_bytes.items, close);
                    const emit_len = self.assistant_bytes.items.len - hold_len;
                    if (emit_len > 0) {
                        try self.emitReasoningDelta(self.assistant_bytes.items[0..emit_len]);
                        self.discardAssistantPrefix(emit_len);
                    }
                    return;
                },
                .tool_call => {
                    try self.tool_call_bytes.appendSlice(self.gpa, self.assistant_bytes.items);
                    self.assistant_bytes.clearRetainingCapacity();

                    const close = "</tool_call>";
                    const index = std.mem.indexOf(u8, self.tool_call_bytes.items, close) orelse {
                        if (final) self.tool_call_bytes.clearRetainingCapacity();
                        return;
                    };
                    const remainder_start = index + close.len;
                    if (remainder_start < self.tool_call_bytes.items.len) {
                        try self.assistant_bytes.appendSlice(self.gpa, self.tool_call_bytes.items[remainder_start..]);
                    }
                    try self.emitToolCall(self.tool_call_bytes.items[0..index]);
                    self.tool_call_bytes.clearRetainingCapacity();
                    self.assistant_state = .visible;
                },
            }
        }
    }

    fn discardPendingPrefix(self: *ChatTokenSink, len: usize) void {
        const remaining_len = self.pending_bytes.items.len - len;
        std.mem.copyForwards(u8, self.pending_bytes.items[0..remaining_len], self.pending_bytes.items[len..]);
        self.pending_bytes.items.len = remaining_len;
    }

    fn discardAssistantPrefix(self: *ChatTokenSink, len: usize) void {
        const remaining_len = self.assistant_bytes.items.len - len;
        std.mem.copyForwards(u8, self.assistant_bytes.items[0..remaining_len], self.assistant_bytes.items[len..]);
        self.assistant_bytes.items.len = remaining_len;
    }
};

pub fn respondChatJson(
    gpa: std.mem.Allocator,
    req: *http.Server.Request,
    c: Completion,
    expose_reasoning: bool,
    response_format_validation: response_format.Validation,
) !void {
    const body = buildChatJson(gpa, c, expose_reasoning, response_format_validation) catch |err| switch (err) {
        error.InvalidModelOutput => {
            try api_error.respondJson(
                gpa,
                req,
                .unprocessable_entity,
                "invalid_model_output",
                "model output did not satisfy requested JSON response format; non-streaming JSON mode is post-validated after generation",
            );
            return;
        },
        else => return err,
    };
    defer gpa.free(body);

    try req.respond(body, .{ .keep_alive = true, .extra_headers = &json_ct });
}

fn buildChatJson(
    gpa: std.mem.Allocator,
    c: Completion,
    expose_reasoning: bool,
    response_format_validation: response_format.Validation,
) ![]u8 {
    const output_text = try response_format.validateOutput(gpa, c.text, response_format_validation);
    var assistant = try parseAssistantOutputOrContent(gpa, output_text, expose_reasoning);
    defer assistant.deinit(gpa);
    var metadata = try initChatSseMetadata(gpa);
    defer metadata.deinit(gpa);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "{\"id\":\"");
    try appendJsonEscaped(gpa, &out, metadata.id);
    try out.appendSlice(gpa, "\",\"object\":\"chat.completion\",\"created\":");
    try out.print(gpa, "{d}", .{metadata.created});
    try out.appendSlice(gpa, ",\"model\":\"" ++ MODEL_ID ++ "\",\"service_tier\":\"" ++ DEFAULT_SERVICE_TIER ++ "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",");
    if (assistant.tool_call) |call| {
        try out.appendSlice(gpa, "\"content\":null");
        if (assistant.reasoning_content) |reasoning| {
            try out.appendSlice(gpa, ",\"reasoning_content\":\"");
            try appendJsonEscaped(gpa, &out, reasoning);
            try out.append(gpa, '"');
        }
        try out.appendSlice(gpa, ",\"refusal\":null");
        try out.appendSlice(gpa, ",\"tool_calls\":[{\"id\":\"");
        try appendJsonEscaped(gpa, &out, call.id);
        try out.appendSlice(gpa, "\",\"type\":\"function\",\"function\":{\"name\":\"");
        try appendJsonEscaped(gpa, &out, call.name);
        try out.appendSlice(gpa, "\",\"arguments\":\"");
        try appendJsonEscaped(gpa, &out, call.arguments_json);
        try out.appendSlice(gpa, "\"}}]},\"finish_reason\":\"tool_calls\"}],");
    } else {
        try out.appendSlice(gpa, "\"content\":\"");
        try appendJsonEscaped(gpa, &out, assistant.content);
        try out.append(gpa, '"');
        if (assistant.reasoning_content) |reasoning| {
            try out.appendSlice(gpa, ",\"reasoning_content\":\"");
            try appendJsonEscaped(gpa, &out, reasoning);
            try out.append(gpa, '"');
        }
        try out.appendSlice(gpa, ",\"refusal\":null,\"tool_calls\":[]");
        try out.print(gpa, "}},\"finish_reason\":\"{s}\"}}],", .{finishReason(c)});
    }
    try out.appendSlice(gpa, "\"usage\":");
    try appendUsageValue(gpa, &out, c);
    try out.appendSlice(gpa, ",\"metadata\":{}}");
    return out.toOwnedSlice(gpa);
}

pub fn writeSseRole(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata) !void {
    try writeSseChunk(gpa, stream, metadata, "{\"role\":\"assistant\"}", null);
}

pub fn flushSse(stream: *http.BodyWriter) !void {
    try stream.writer.flush();
    try stream.flush();
}

fn writeSseKeepalive(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata) !void {
    try writeSseChunk(gpa, stream, metadata, "{}", null);
}

pub fn writeSseReasoning(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata, text: []const u8) !void {
    try writeSseStringDelta(gpa, stream, metadata, "reasoning_content", text);
}

pub fn writeSseContent(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata, text: []const u8) !void {
    try writeSseStringDelta(gpa, stream, metadata, "content", text);
}

fn writeSseStringDelta(
    gpa: std.mem.Allocator,
    stream: *http.BodyWriter,
    metadata: ChatSseMetadata,
    field: []const u8,
    text: []const u8,
) !void {
    var delta: std.ArrayList(u8) = .empty;
    defer delta.deinit(gpa);
    try delta.appendSlice(gpa, "{\"");
    try delta.appendSlice(gpa, field);
    try delta.appendSlice(gpa, "\":\"");
    try appendJsonEscaped(gpa, &delta, text);
    try delta.appendSlice(gpa, "\"}");
    try writeSseChunk(gpa, stream, metadata, delta.items, null);
}

pub fn writeSseToolCall(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata, call: ToolCall, index: usize) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try appendSseToolCallChunk(gpa, &out, metadata, call, index);
    try stream.writer.writeAll(out.items);
}

fn appendSseToolCallChunk(gpa: std.mem.Allocator, out: *std.ArrayList(u8), metadata: ChatSseMetadata, call: ToolCall, index: usize) !void {
    var delta: std.ArrayList(u8) = .empty;
    defer delta.deinit(gpa);
    try delta.appendSlice(gpa, "{\"tool_calls\":[{\"index\":");
    try delta.print(gpa, "{d}", .{index});
    try delta.appendSlice(gpa, ",\"id\":\"");
    try appendJsonEscaped(gpa, &delta, call.id);
    try delta.appendSlice(gpa, "\",\"type\":\"function\",\"function\":{\"name\":\"");
    try appendJsonEscaped(gpa, &delta, call.name);
    try delta.appendSlice(gpa, "\",\"arguments\":\"");
    try appendJsonEscaped(gpa, &delta, call.arguments_json);
    try delta.appendSlice(gpa, "\"}}]}");
    try appendSseChunk(gpa, out, metadata, delta.items, null);
}

pub fn writeSseFinish(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata, reason: []const u8) !void {
    try writeSseChunk(gpa, stream, metadata, "{}", reason);
}

pub fn writeSseUsage(gpa: std.mem.Allocator, stream: *http.BodyWriter, metadata: ChatSseMetadata, c: Completion) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try appendSseUsage(gpa, &out, metadata, c);
    try stream.writer.writeAll(out.items);
}

pub fn writeSseError(stream: *http.BodyWriter, e: anyerror) !void {
    const payload: api_error.Payload = switch (e) {
        error.InvalidModelOutput, error.MalformedModelOutput, error.InvalidToolCall => .{
            .code = "invalid_model_output",
            .message = "model output could not be converted to an OpenAI streaming response",
        },
        else => .{
            .code = "internal_error",
            .message = "request failed",
        },
    };
    try api_error.writeSse(stream, payload.code, payload.message);
}

pub fn writeSseDone(stream: *http.BodyWriter) !void {
    try stream.writer.writeAll("data: [DONE]\n\n");
}

pub fn finishReason(c: Completion) []const u8 {
    return if (c.hit_max) "length" else "stop";
}

fn appendModelObject(gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
    try out.appendSlice(gpa, "{\"id\":\"" ++ MODEL_ID ++ "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"peregrine\"}");
}

fn appendJsonEscaped(gpa: std.mem.Allocator, list: *std.ArrayList(u8), s: []const u8) !void {
    const REPLACEMENT = "\u{FFFD}"; // for invalid/truncated UTF-8 (a completion cut mid-char)
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c < 0x80) {
            switch (c) {
                '"' => try list.appendSlice(gpa, "\\\""),
                '\\' => try list.appendSlice(gpa, "\\\\"),
                '\n' => try list.appendSlice(gpa, "\\n"),
                '\r' => try list.appendSlice(gpa, "\\r"),
                '\t' => try list.appendSlice(gpa, "\\t"),
                else => if (c < 0x20) {
                    var b: [6]u8 = undefined;
                    const escaped = try std.fmt.bufPrint(&b, "\\u{x:0>4}", .{c});
                    try list.appendSlice(gpa, escaped);
                } else try list.append(gpa, c),
            }
            i += 1;
            continue;
        }
        // Multibyte: pass through only well-formed UTF-8 so the JSON body stays
        // valid even when generation was truncated mid-character.
        const len = std.unicode.utf8ByteSequenceLength(c) catch {
            try list.appendSlice(gpa, REPLACEMENT);
            i += 1;
            continue;
        };
        if (i + len > s.len or (std.unicode.utf8Decode(s[i .. i + len]) catch null) == null) {
            try list.appendSlice(gpa, REPLACEMENT);
            i += 1;
            continue;
        }
        try list.appendSlice(gpa, s[i .. i + len]);
        i += len;
    }
}

fn appendChatSsePrefix(gpa: std.mem.Allocator, out: *std.ArrayList(u8), metadata: ChatSseMetadata) !void {
    try out.appendSlice(gpa, "data: {\"id\":\"");
    try appendJsonEscaped(gpa, out, metadata.id);
    try out.appendSlice(gpa, "\",\"object\":\"chat.completion.chunk\",\"created\":");
    try out.print(gpa, "{d}", .{metadata.created});
}

fn writeSseChunk(
    gpa: std.mem.Allocator,
    stream: *http.BodyWriter,
    metadata: ChatSseMetadata,
    delta_json: []const u8,
    finish_reason: ?[]const u8,
) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try appendSseChunk(gpa, &out, metadata, delta_json, finish_reason);
    try stream.writer.writeAll(out.items);
}

fn appendSseChunk(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(u8),
    metadata: ChatSseMetadata,
    delta_json: []const u8,
    finish_reason: ?[]const u8,
) !void {
    try appendChatSsePrefix(gpa, out, metadata);
    try out.appendSlice(gpa, ",\"model\":\"" ++ MODEL_ID ++ "\",\"service_tier\":\"" ++ DEFAULT_SERVICE_TIER ++ "\",\"choices\":[{\"index\":0,\"delta\":");
    try out.appendSlice(gpa, delta_json);
    try out.appendSlice(gpa, ",\"finish_reason\":");
    if (finish_reason) |reason| {
        try out.append(gpa, '"');
        try appendJsonEscaped(gpa, out, reason);
        try out.append(gpa, '"');
    } else {
        try out.appendSlice(gpa, "null");
    }
    try out.appendSlice(gpa, "}]}\n\n");
}

fn appendSseUsage(gpa: std.mem.Allocator, out: *std.ArrayList(u8), metadata: ChatSseMetadata, c: Completion) !void {
    try appendChatSsePrefix(gpa, out, metadata);
    try out.appendSlice(gpa, ",\"model\":\"" ++ MODEL_ID ++ "\",\"service_tier\":\"" ++ DEFAULT_SERVICE_TIER ++ "\",\"choices\":[],\"usage\":");
    try appendUsageValue(gpa, out, c);
    try out.appendSlice(gpa, "}\n\n");
}

fn appendUsageValue(gpa: std.mem.Allocator, out: *std.ArrayList(u8), c: Completion) !void {
    try out.print(
        gpa,
        "{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":0}},\"completion_tokens_details\":{{\"reasoning_tokens\":0,\"audio_tokens\":0,\"accepted_prediction_tokens\":0,\"rejected_prediction_tokens\":0}}}}",
        .{ c.prompt_tokens, c.completion_tokens, c.prompt_tokens + c.completion_tokens },
    );
}

fn allocChatCompletionId(gpa: std.mem.Allocator, created: u64) ![]u8 {
    return std.fmt.allocPrint(
        gpa,
        "chatcmpl-peregrine-qwen35-q4-{d}-{d}",
        .{ created, monotonicNowNs() },
    );
}

fn unixTimestampSeconds() u64 {
    const posix = std.posix;
    var ts: posix.timespec = undefined;
    return switch (posix.errno(posix.system.clock_gettime(posix.CLOCK.REALTIME, &ts))) {
        .SUCCESS => blk: {
            std.debug.assert(ts.sec >= 0);
            break :blk @intCast(ts.sec);
        },
        else => @panic("clock_gettime(CLOCK_REALTIME) failed"),
    };
}

pub fn parseAssistantOutputOrContent(gpa: std.mem.Allocator, text: []const u8, expose_reasoning: bool) !AssistantOutput {
    var output = parseAssistantOutput(gpa, text) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => fallbackAssistantOutput(text, expose_reasoning),
    };
    if (!expose_reasoning) output.reasoning_content = null;
    return output;
}

fn parseAssistantOutput(gpa: std.mem.Allocator, raw: []const u8) !AssistantOutput {
    var content = raw;
    var reasoning: ?[]const u8 = null;

    if (std.mem.indexOf(u8, content, "<think>")) |think_start| {
        if (std.mem.trim(u8, content[0..think_start], " \t\r\n").len == 0) {
            const body_start = think_start + "<think>".len;
            const close_start = std.mem.indexOfPos(u8, content, body_start, "</think>") orelse return error.MalformedModelOutput;
            reasoning = std.mem.trim(u8, content[body_start..close_start], "\r\n");
            content = trimLeadingAscii(content[close_start + "</think>".len ..]);
        }
    }

    if (std.mem.indexOf(u8, content, "<tool_call>")) |tool_start| {
        const body_start = tool_start + "<tool_call>".len;
        const close_start = std.mem.indexOfPos(u8, content, body_start, "</tool_call>") orelse return error.MalformedModelOutput;
        const call = try parseToolCallBlock(gpa, content[body_start..close_start], 0);
        return .{
            .content = std.mem.trim(u8, content[0..tool_start], " \t\r\n"),
            .reasoning_content = reasoning,
            .tool_call = call,
        };
    }

    return .{
        .content = content,
        .reasoning_content = reasoning,
    };
}

const AssistantStreamState = enum {
    visible,
    reasoning,
    tool_call,
};

const AssistantMarker = struct {
    index: usize,
    text: []const u8,
    state: AssistantStreamState,
};

fn nextAssistantMarker(text: []const u8) ?AssistantMarker {
    const think = std.mem.indexOf(u8, text, "<think>");
    const tool = std.mem.indexOf(u8, text, "<tool_call>");
    if (think == null and tool == null) return null;
    if (tool == null or (think != null and think.? <= tool.?)) {
        return .{ .index = think.?, .text = "<think>", .state = .reasoning };
    }
    return .{ .index = tool.?, .text = "<tool_call>", .state = .tool_call };
}

fn fallbackAssistantOutput(raw: []const u8, expose_reasoning: bool) AssistantOutput {
    if (firstControlMarkerIndex(raw)) |marker_start| {
        const content = std.mem.trim(u8, raw[0..marker_start], " \t\r\n");
        const marker = raw[marker_start..];
        const reasoning = if (expose_reasoning and std.mem.startsWith(u8, marker, "<think>"))
            std.mem.trim(u8, marker["<think>".len..], " \t\r\n")
        else
            null;
        return .{
            .content = content,
            .reasoning_content = reasoning,
        };
    }
    return .{ .content = raw };
}

fn firstControlMarkerIndex(text: []const u8) ?usize {
    return if (nextAssistantMarker(text)) |marker| marker.index else null;
}

fn controlMarkerSuffixLen(text: []const u8) usize {
    return @max(
        suffixPrefixLen(text, "<think>"),
        suffixPrefixLen(text, "<tool_call>"),
    );
}

fn suffixPrefixLen(text: []const u8, marker: []const u8) usize {
    const max_len = @min(text.len, marker.len - 1);
    var len = max_len;
    while (len > 0) : (len -= 1) {
        if (std.mem.eql(u8, text[text.len - len ..], marker[0..len])) return len;
    }
    return 0;
}

fn parseToolCallBlock(gpa: std.mem.Allocator, body: []const u8, index: usize) !ToolCall {
    const function_prefix = "<function=";
    const function_start = std.mem.indexOf(u8, body, function_prefix) orelse return error.InvalidToolCall;
    const name_start = function_start + function_prefix.len;
    const name_end = std.mem.indexOfPos(u8, body, name_start, ">") orelse return error.InvalidToolCall;
    const name = std.mem.trim(u8, body[name_start..name_end], " \t\r\n");
    if (name.len == 0) return error.InvalidToolCall;

    const function_close = "</function>";
    const args_start = name_end + 1;
    const args_end = std.mem.lastIndexOf(u8, body, function_close) orelse return error.InvalidToolCall;
    if (args_end < args_start) return error.InvalidToolCall;

    const arguments_json = try parseToolArgumentsJson(gpa, body[args_start..args_end]);
    errdefer gpa.free(arguments_json);
    const owned_name = try gpa.dupe(u8, name);
    errdefer gpa.free(owned_name);
    const id = try std.fmt.allocPrint(gpa, "call_peregrine_{d}", .{index});

    return .{
        .id = id,
        .name = owned_name,
        .arguments_json = arguments_json,
    };
}

fn parseToolArgumentsJson(gpa: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();

    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();

    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, body, cursor, "<parameter=")) |tag_start| {
        const name_start = tag_start + "<parameter=".len;
        const name_end = std.mem.indexOfPos(u8, body, name_start, ">") orelse return error.InvalidToolCall;
        const name = std.mem.trim(u8, body[name_start..name_end], " \t\r\n");
        if (name.len == 0) return error.InvalidToolCall;

        const value_start = name_end + 1;
        const close_start = std.mem.indexOfPos(u8, body, value_start, "</parameter>") orelse return error.InvalidToolCall;
        const value = std.mem.trim(u8, body[value_start..close_start], "\r\n");

        try jw.objectField(name);
        try writeToolArgumentValue(gpa, &jw, value);

        cursor = close_start + "</parameter>".len;
    }

    try jw.endObject();
    return out.toOwnedSlice();
}

fn writeToolArgumentValue(
    gpa: std.mem.Allocator,
    jw: *std.json.Stringify,
    value: []const u8,
) !void {
    const trimmed = std.mem.trim(u8, value, " \t\r\n");
    if (isJsonLikeValue(trimmed)) {
        var parsed = std.json.parseFromSlice(std.json.Value, gpa, trimmed, .{}) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => {
                try jw.write(value);
                return;
            },
        };
        defer parsed.deinit();
        try jw.write(parsed.value);
        return;
    }
    try jw.write(value);
}

fn isJsonLikeValue(value: []const u8) bool {
    if (value.len == 0) return false;
    return switch (value[0]) {
        '{', '[', '"', '-', '0'...'9' => true,
        't' => std.mem.eql(u8, value, "true"),
        'f' => std.mem.eql(u8, value, "false"),
        'n' => std.mem.eql(u8, value, "null"),
        else => false,
    };
}

fn trimLeadingAscii(value: []const u8) []const u8 {
    var start: usize = 0;
    while (start < value.len and std.mem.indexOfScalar(u8, " \t\r\n", value[start]) != null) {
        start += 1;
    }
    return value[start..];
}

test "buildChatJson emits valid tool call response JSON" {
    const gpa = std.testing.allocator;
    const body = try buildChatJson(gpa, .{
        .text =
        \\<tool_call>
        \\<function=lookup>
        \\<parameter=query>
        \\cats
        \\</parameter>
        \\</function>
        \\</tool_call>
        ,
        .prompt_tokens = 3,
        .completion_tokens = 9,
        .hit_max = false,
    }, false, .none);
    defer gpa.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const choices = root.get("choices").?.array;
    const choice = choices.items[0].object;
    try std.testing.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
}

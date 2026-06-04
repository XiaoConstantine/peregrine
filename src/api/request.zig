//! OpenAI-compatible request parsing for Peregrine's minimal serving surface.

const std = @import("std");

const response = @import("response.zig");
const qwen_chat = @import("../qwen_chat.zig");
const response_format = @import("../response_format.zig");
const stop_sequences = @import("../stop_sequences.zig");

pub const WarmupMode = qwen_chat.RenderMode;

pub const ChatParams = struct {
    prompt: []u8,
    cache_prefix_prompt: ?[]u8 = null,
    max_new_tokens: u32,
    stream: bool,
    stream_include_usage: bool,
    thinking_disabled: bool,
    response_format_validation: response_format.Validation,
    stop: [][]u8,

    pub fn deinit(self: *const ChatParams, gpa: std.mem.Allocator) void {
        if (self.cache_prefix_prompt) |prefix| gpa.free(prefix);
        gpa.free(self.prompt);
        stop_sequences.free(gpa, self.stop);
    }
};

pub fn parseChatParams(gpa: std.mem.Allocator, body: []const u8, default_new_tokens: u32, max_new_tokens: u32) !ChatParams {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, body, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.MalformedChatRequest,
    };
    try validateRequestModel(&root);
    try validateGreedyGenerationOptions(&root);
    try validateChatTextOnlyOptions(&root);
    const stop = try stop_sequences.parse(gpa, &root);
    errdefer stop_sequences.free(gpa, stop);

    const rendered = try qwen_chat.renderRequestPrompt(gpa, &root);
    errdefer rendered.deinit(gpa);

    return .{
        .prompt = rendered.text,
        .cache_prefix_prompt = rendered.cache_prefix,
        .max_new_tokens = try requestedChatMaxTokens(&root, default_new_tokens, max_new_tokens),
        .stream = try jsonBool(&root, "stream"),
        .stream_include_usage = try jsonStreamIncludeUsage(&root),
        .thinking_disabled = rendered.thinking_disabled,
        .response_format_validation = rendered.response_format_validation,
        .stop = stop,
    };
}

pub fn parseWarmupPrompt(gpa: std.mem.Allocator, body: []const u8) ![]u8 {
    return parseWarmupPromptWithDefaultMode(gpa, body, .full);
}

pub fn parseWarmupPromptWithDefaultMode(
    gpa: std.mem.Allocator,
    body: []const u8,
    default_mode: WarmupMode,
) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, body, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.MalformedRequest,
    };
    try validateRequestModel(&root);
    if (root.get("prompt") != null or root.get("input") != null) {
        const prompt = try rawWarmupPrompt(&root);
        return try gpa.dupe(u8, prompt);
    }
    if (root.get("messages") != null) {
        const mode = try warmupRenderMode(&root, default_mode);
        return qwen_chat.renderPrompt(gpa, &root, mode);
    }
    return error.MalformedRequest;
}

fn warmupRenderMode(root: *const std.json.ObjectMap, default_mode: WarmupMode) !WarmupMode {
    const mode = jsonStringField(root, "warmup_mode") orelse return default_mode;
    if (std.mem.eql(u8, mode, "full")) return .full;
    if (std.mem.eql(u8, mode, "before_last_user_content")) return .before_last_user_content;
    if (std.mem.eql(u8, mode, "static_prefix")) return .before_last_user_content;
    if (std.mem.eql(u8, mode, "pi_static_prefix")) return .before_last_user_content;
    return error.MalformedRequest;
}

fn jsonStringField(root: *const std.json.ObjectMap, field: []const u8) ?[]const u8 {
    const value = root.get(field) orelse return null;
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

fn validateRequestModel(root: *const std.json.ObjectMap) !void {
    const value = root.get("model") orelse return;
    const requested = switch (value) {
        .string => |s| s,
        else => return error.InvalidModel,
    };
    if (response.isServedModelId(requested)) return;
    return error.UnsupportedModel;
}

fn validateChatTextOnlyOptions(root: *const std.json.ObjectMap) !void {
    try validateOptionalMetadata(root);
    try validateOptionalString(root, "user");
    try validateOpenAiRoutingNoopOptions(root, error.UnsupportedChatParameter);
    try validateOptionalNoneOrAutoString(root, "tool_choice");
    try validateOptionalNoopBool(root, "parallel_tool_calls");
    try validateOptionalNoneOrAutoString(root, "function_call");
    try validateOptionalModalities(root);
    try validateOptionalNullOnly(root, "audio");
    try validateOptionalNullOnly(root, "prediction");
    try validateOptionalEmptyObject(root, "logit_bias");
}

fn validateGreedyGenerationOptions(root: *const std.json.ObjectMap) !void {
    try validateOptionalNoopPenalty(root, "presence_penalty");
    try validateOptionalNoopPenalty(root, "frequency_penalty");
    try validateOptionalChoiceCount(root);
    try validateOptionalLogprobs(root, "logprobs", true);
    try validateOptionalLogprobs(root, "top_logprobs", false);
    try validateOptionalGreedySampling(root);
    try validateOptionalSeed(root);
    try validateOptionalFalseBool(root, "store");
}

fn validateOptionalMetadata(root: *const std.json.ObjectMap) !void {
    const value = root.get("metadata") orelse return;
    if (value == .null) return;
    const metadata = switch (value) {
        .object => |object| object,
        else => return error.InvalidRequestJson,
    };
    if (metadata.count() > 16) return error.InvalidRequestJson;
    var it = metadata.iterator();
    while (it.next()) |entry| {
        if (entry.key_ptr.*.len > 64) return error.InvalidRequestJson;
        const metadata_value = switch (entry.value_ptr.*) {
            .string => |string| string,
            else => return error.InvalidRequestJson,
        };
        if (metadata_value.len > 512) return error.InvalidRequestJson;
    }
}

fn validateOpenAiRoutingNoopOptions(root: *const std.json.ObjectMap, unsupported_error: anyerror) !void {
    try validateOptionalStringChoice(root, "service_tier", &.{ "auto", "default", "flex", "priority" }, unsupported_error);
    try validateOptionalString(root, "prompt_cache_key");
    try validateOptionalStringChoice(root, "prompt_cache_retention", &.{ "in-memory", "in_memory", "24h" }, unsupported_error);
    try validateOptionalString(root, "safety_identifier");
}

fn validateOptionalStringChoice(
    root: *const std.json.ObjectMap,
    field: []const u8,
    supported: []const []const u8,
    unsupported_error: anyerror,
) !void {
    const value = root.get(field) orelse return;
    const choice = switch (value) {
        .null => return,
        .string => |string| string,
        else => return error.InvalidRequestJson,
    };
    for (supported) |candidate| {
        if (std.mem.eql(u8, choice, candidate)) return;
    }
    return unsupported_error;
}

fn validateOptionalString(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    switch (value) {
        .null, .string => {},
        else => return error.InvalidRequestJson,
    }
}

fn validateOptionalNoopBool(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    switch (value) {
        .null, .bool => {},
        else => return error.InvalidRequestJson,
    }
}

fn validateOptionalNoneOrAutoString(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    const choice = switch (value) {
        .null => return,
        .string => |s| s,
        else => return error.UnsupportedChatParameter,
    };
    if (std.mem.eql(u8, choice, "none") or std.mem.eql(u8, choice, "auto")) return;
    return error.UnsupportedChatParameter;
}

fn validateOptionalNullOnly(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    if (value != .null) return error.UnsupportedChatParameter;
}

fn validateOptionalEmptyObject(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    switch (value) {
        .null => {},
        .object => |object| if (object.count() != 0) return error.UnsupportedChatParameter,
        else => return error.InvalidRequestJson,
    }
}

fn validateOptionalFalseBool(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    switch (value) {
        .null => {},
        .bool => |enabled| if (enabled) return error.UnsupportedSamplingParameter,
        else => return error.InvalidRequestJson,
    }
}

fn validateOptionalChoiceCount(root: *const std.json.ObjectMap) !void {
    const value = root.get("n") orelse return;
    if (value == .null) return;
    const count = try jsonUsize(value);
    if (count != 1) return error.UnsupportedSamplingParameter;
}

fn validateOptionalNoopPenalty(root: *const std.json.ObjectMap, field: []const u8) !void {
    const value = root.get(field) orelse return;
    if (value == .null) return;
    if (try jsonF64(value) != 0) return error.UnsupportedSamplingParameter;
}

fn validateOptionalLogprobs(root: *const std.json.ObjectMap, field: []const u8, allow_bool: bool) !void {
    const value = root.get(field) orelse return;
    switch (value) {
        .null => {},
        .bool => |enabled| {
            if (!allow_bool or enabled) return error.UnsupportedSamplingParameter;
        },
        .integer, .float => if (try jsonF64(value) != 0) return error.UnsupportedSamplingParameter,
        else => return error.UnsupportedSamplingParameter,
    }
}

fn validateOptionalGreedySampling(root: *const std.json.ObjectMap) !void {
    const temperature = try jsonOptionalF64(root, "temperature", 0);
    const top_p = try jsonOptionalF64(root, "top_p", 1);
    if (temperature != 0) return error.UnsupportedSamplingParameter;
    if (top_p != 1) return error.UnsupportedSamplingParameter;
}

fn validateOptionalSeed(root: *const std.json.ObjectMap) !void {
    const value = root.get("seed") orelse return;
    if (value == .null) return;
    _ = try jsonUsize(value);
}

fn validateOptionalModalities(root: *const std.json.ObjectMap) !void {
    const value = root.get("modalities") orelse return;
    if (value == .null) return;
    const modalities = switch (value) {
        .array => |array| array.items,
        else => return error.InvalidRequestJson,
    };
    if (modalities.len == 0) return error.UnsupportedChatParameter;
    for (modalities) |modality_value| {
        const modality = switch (modality_value) {
            .string => |s| s,
            else => return error.InvalidRequestJson,
        };
        if (!std.mem.eql(u8, modality, "text")) return error.UnsupportedChatParameter;
    }
}

fn jsonPositiveU32(value: std.json.Value) !?u32 {
    return switch (value) {
        .null => null,
        .integer => |n| blk: {
            if (n <= 0) return error.InvalidMaxTokens;
            if (n > std.math.maxInt(u32)) return error.MaxNewTokensTooLarge;
            break :blk @as(u32, @intCast(n));
        },
        else => error.InvalidMaxTokens,
    };
}

fn jsonUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |n| blk: {
            if (n < 0) return error.InvalidRequestJson;
            break :blk @as(usize, @intCast(n));
        },
        .float => |n| blk: {
            if (n < 0 or @floor(n) != n) return error.InvalidRequestJson;
            break :blk @as(usize, @intFromFloat(n));
        },
        else => error.InvalidRequestJson,
    };
}

fn jsonF64(value: std.json.Value) !f64 {
    return switch (value) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => error.InvalidRequestJson,
    };
}

fn jsonOptionalF64(root: *const std.json.ObjectMap, field: []const u8, default_value: f64) !f64 {
    const value = root.get(field) orelse return default_value;
    if (value == .null) return default_value;
    return jsonF64(value);
}

fn rawWarmupPrompt(root: *const std.json.ObjectMap) ![]const u8 {
    const value = root.get("prompt") orelse root.get("input") orelse return error.MissingPrompt;
    return switch (value) {
        .string => |prompt| if (prompt.len == 0) error.EmptyPrompt else prompt,
        .array => |array| blk: {
            if (array.items.len == 0) return error.EmptyPrompt;
            if (array.items.len != 1) return error.MalformedRequest;
            const prompt = switch (array.items[0]) {
                .string => |prompt| prompt,
                else => return error.MalformedRequest,
            };
            if (prompt.len == 0) return error.EmptyPrompt;
            break :blk prompt;
        },
        else => error.InvalidRequestJson,
    };
}

fn requestedMaxTokens(
    root: *const std.json.ObjectMap,
    default_new_tokens: u32,
    max_new_tokens: u32,
    fields: []const []const u8,
) !u32 {
    for (fields) |field| {
        if (root.get(field)) |value| {
            const requested = (try jsonPositiveU32(value)) orelse return default_new_tokens;
            if (requested > max_new_tokens) return error.MaxNewTokensTooLarge;
            return requested;
        }
    }
    return default_new_tokens;
}

fn requestedChatMaxTokens(root: *const std.json.ObjectMap, default_new_tokens: u32, max_new_tokens: u32) !u32 {
    return requestedMaxTokens(
        root,
        default_new_tokens,
        max_new_tokens,
        &.{ "max_new_tokens", "max_completion_tokens", "max_tokens", "max_output_tokens" },
    );
}

fn jsonBool(root: *const std.json.ObjectMap, field: []const u8) !bool {
    const value = root.get(field) orelse return false;
    return switch (value) {
        .null => false,
        .bool => |b| b,
        else => error.InvalidRequestJson,
    };
}

fn jsonStreamIncludeUsage(root: *const std.json.ObjectMap) !bool {
    const value = root.get("stream_options") orelse return false;
    if (value == .null) return false;
    const options = switch (value) {
        .object => |object| object,
        else => return error.InvalidRequestJson,
    };
    return switch (options.get("include_usage") orelse return false) {
        .null => false,
        .bool => |b| b,
        else => error.InvalidRequestJson,
    };
}

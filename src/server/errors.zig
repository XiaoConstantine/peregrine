//! Peregrine server error mapping for OpenAI-compatible clients.

const std = @import("std");
const http = std.http;

const api_error = @import("../api/error.zig");

pub const TokenCaps = struct {
    default_new: u32,
    max_new_cap: u32,
    max_total: usize,
};

const Response = struct {
    status: http.Status,
    code: []const u8,
    message: []const u8,
};

pub fn respondRequestError(
    gpa: std.mem.Allocator,
    request: *http.Server.Request,
    caps: TokenCaps,
    err: anyerror,
) !void {
    var msg: [256]u8 = undefined;
    const response = requestErrorResponse(caps, err, &msg);
    try api_error.respondJson(gpa, request, response.status, response.code, response.message);
}

pub fn respondGenError(
    gpa: std.mem.Allocator,
    request: *http.Server.Request,
    caps: TokenCaps,
    err: anyerror,
) !void {
    var msg: [256]u8 = undefined;
    const response = generationErrorResponse(caps, err, &msg);
    try api_error.respondJson(gpa, request, response.status, response.code, response.message);
}

pub fn respondRequestBodyTooLarge(gpa: std.mem.Allocator, request: *http.Server.Request) !void {
    try api_error.respondJsonAndClose(gpa, request, .payload_too_large, "request_body_too_large", "request body exceeds the server request-body limit");
}

pub fn respondNotFound(gpa: std.mem.Allocator, request: *http.Server.Request) !void {
    try api_error.respondJson(gpa, request, .not_found, "not_found", "endpoint not found; see /health capabilities");
}

pub fn respondMethodNotAllowed(gpa: std.mem.Allocator, request: *http.Server.Request) !void {
    try api_error.respondJson(gpa, request, .method_not_allowed, "method_not_allowed", "unsupported method for this endpoint; see /health capabilities");
}

pub fn respondModelNotFound(gpa: std.mem.Allocator, request: *http.Server.Request) !void {
    try api_error.respondJson(gpa, request, .not_found, "model_not_found", "requested model is not served by this Peregrine instance");
}

pub fn respondUnsupportedOpenAIEndpoint(gpa: std.mem.Allocator, request: *http.Server.Request) !void {
    try api_error.respondJson(gpa, request, .not_implemented, "unsupported_endpoint", "this OpenAI endpoint is recognized but not implemented by the Qwen3.5 q4 server");
}

pub fn isKnownUnsupportedOpenAIPath(path: []const u8) bool {
    if (isChatCompletionStoragePath(path)) return true;

    const prefixes = [_][]const u8{
        "/v1/assistants",
        "/v1/audio",
        "/v1/batches",
        "/v1/chatkit",
        "/v1/containers",
        "/v1/conversations",
        "/v1/embeddings",
        "/v1/evals",
        "/v1/files",
        "/v1/images",
        "/v1/moderations",
        "/v1/organization",
        "/v1/projects",
        "/v1/realtime",
        "/v1/responses",
        "/v1/skills",
        "/v1/threads",
        "/v1/uploads",
        "/v1/vector_stores",
        "/v1/videos",
    };
    for (prefixes) |prefix| {
        if (std.mem.eql(u8, path, prefix)) return true;
        if (path.len > prefix.len and
            path[prefix.len] == '/' and
            std.mem.startsWith(u8, path, prefix))
        {
            return true;
        }
    }
    return false;
}

fn isChatCompletionStoragePath(path: []const u8) bool {
    const prefix = "/v1/chat/completions/";
    if (!std.mem.startsWith(u8, path, prefix)) return false;
    const suffix = path[prefix.len..];
    if (suffix.len == 0) return false;

    if (std.mem.endsWith(u8, suffix, "/messages")) {
        const id = suffix[0 .. suffix.len - "/messages".len];
        return id.len != 0 and std.mem.indexOfScalar(u8, id, '/') == null;
    }

    return std.mem.indexOfScalar(u8, suffix, '/') == null;
}

fn requestErrorResponse(caps: TokenCaps, err: anyerror, msg: []u8) Response {
    return switch (err) {
        error.MaxNewTokensTooLarge => .{
            .status = .bad_request,
            .code = "max_new_tokens_too_large",
            .message = std.fmt.bufPrint(
                msg,
                "requested max_new_tokens exceeds server limit (default_new_tokens={d}, max_new_tokens={d})",
                .{ caps.default_new, caps.max_new_cap },
            ) catch "requested max_new_tokens exceeds the server limit",
        },
        error.InvalidMaxTokens => .{
            .status = .bad_request,
            .code = "invalid_max_tokens",
            .message = "max output tokens must be a positive integer",
        },
        error.InvalidModel => .{
            .status = .bad_request,
            .code = "invalid_model",
            .message = "model must be a string",
        },
        error.UnsupportedModel => .{
            .status = .bad_request,
            .code = "unsupported_model",
            .message = "requested model is not served by this Peregrine instance",
        },
        error.MissingPrompt => .{
            .status = .bad_request,
            .code = "missing_prompt",
            .message = "request body must include a single text prompt or input",
        },
        error.EmptyPrompt => .{
            .status = .bad_request,
            .code = "empty_prompt",
            .message = "prompt must be non-empty",
        },
        error.MissingMessages => .{
            .status = .bad_request,
            .code = "missing_messages",
            .message = "chat request body must include a messages array",
        },
        error.EmptyMessages => .{
            .status = .bad_request,
            .code = "empty_messages",
            .message = "messages must be non-empty",
        },
        error.EmptyChatContent => .{
            .status = .bad_request,
            .code = "empty_chat_content",
            .message = "messages contained no content",
        },
        error.MissingMessageRole => .{
            .status = .bad_request,
            .code = "missing_message_role",
            .message = "each chat message must include a role",
        },
        error.UnsupportedMessageRole => .{
            .status = .bad_request,
            .code = "unsupported_message_role",
            .message = "supported chat roles: system, developer, user, assistant",
        },
        error.MissingMessageContent => .{
            .status = .bad_request,
            .code = "missing_message_content",
            .message = "each chat message must include content",
        },
        error.UnsupportedMessageContent => .{
            .status = .bad_request,
            .code = "unsupported_message_content",
            .message = "chat content must be a string or text parts",
        },
        error.UnsupportedChatParameter => .{
            .status = .bad_request,
            .code = "unsupported_chat_parameter",
            .message = "Chat completions currently support text-only generation, post-validated non-streaming JSON response formatting, custom function tool definitions as prompt context, and no forced tool calls, audio, prediction, or non-empty logit bias",
        },
        error.UnsupportedSamplingParameter => .{
            .status = .bad_request,
            .code = "unsupported_sampling_parameter",
            .message = "supported sampling controls are greedy temperature 0, top_p 1, n 1, and OpenAI stop sequences; omit logprobs and penalties",
        },
        error.UnsafeChatContent => .{
            .status = .bad_request,
            .code = "unsafe_chat_content",
            .message = "chat content could not be rendered safely",
        },
        error.InvalidRequestJson,
        error.MalformedChatRequest,
        error.MalformedRequest,
        error.SyntaxError,
        error.UnexpectedEndOfInput,
        => .{
            .status = .bad_request,
            .code = "invalid_json",
            .message = "request body must be valid JSON with supported OpenAI chat fields",
        },
        else => .{
            .status = .bad_request,
            .code = "invalid_request",
            .message = "invalid request",
        },
    };
}

fn generationErrorResponse(caps: TokenCaps, err: anyerror, msg: []u8) Response {
    return switch (err) {
        error.EmptyPrompt => .{
            .status = .bad_request,
            .code = "empty_prompt",
            .message = "prompt must be non-empty",
        },
        error.EmptyInput => .{
            .status = .bad_request,
            .code = "invalid_prompt",
            .message = "prompt produced no tokens",
        },
        error.SequenceTooLong => .{
            .status = .bad_request,
            .code = "context_too_long",
            .message = std.fmt.bufPrint(
                msg,
                "prompt plus max_new_tokens exceeds available sequence capacity (max_total_tokens={d}, max_new_tokens={d})",
                .{ caps.max_total, caps.max_new_cap },
            ) catch "prompt plus max_new_tokens exceeds available sequence capacity",
        },
        error.InvalidModelOutput, error.MalformedModelOutput, error.InvalidToolCall => .{
            .status = .unprocessable_entity,
            .code = "invalid_model_output",
            .message = "model output could not be converted to an OpenAI response",
        },
        else => .{
            .status = .internal_server_error,
            .code = "internal_error",
            .message = "request failed",
        },
    };
}

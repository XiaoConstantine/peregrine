//! Minimal OpenAI response_format support shared by chat rendering and responses.

const std = @import("std");
const qwen_escape = @import("qwen_escape.zig");

pub const Validation = enum {
    none,
    json_value,
    json_object,
};

pub const Format = union(enum) {
    text,
    json_object,
    json_schema: std.json.Value,
};

pub fn validationFor(format: Format) Validation {
    return switch (format) {
        .text => .none,
        .json_object => .json_object,
        .json_schema => .json_value,
    };
}

pub fn appendQwenSystemInstruction(
    gpa: std.mem.Allocator,
    prompt: *std.ArrayList(u8),
    format: Format,
) !bool {
    switch (format) {
        .text => return false,
        .json_object => {
            try appendQwenSystemMessage(
                gpa,
                prompt,
                "Respond with a single valid JSON object. Do not include markdown, code fences, or explanatory text.",
            );
            return true;
        },
        .json_schema => |json_schema| {
            const schema_json = try serializeJsonValue(gpa, json_schema);
            defer gpa.free(schema_json);

            try prompt.appendSlice(gpa, "<|im_start|>system\n");
            try prompt.appendSlice(
                gpa,
                "Respond with a JSON value that matches this JSON schema. Do not include markdown, code fences, or explanatory text.\nJSON schema:\n",
            );
            try appendSafeInstructionContent(gpa, prompt, schema_json);
            try prompt.appendSlice(gpa, "<|im_end|>\n");
            return true;
        },
    }
}

pub fn validateOutput(gpa: std.mem.Allocator, text: []const u8, response_validation: Validation) ![]const u8 {
    switch (response_validation) {
        .none => return text,
        .json_value, .json_object => {},
    }

    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, trimmed, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidModelOutput,
    };
    defer parsed.deinit();

    if (response_validation == .json_object and parsed.value != .object) {
        return error.InvalidModelOutput;
    }
    return trimmed;
}

pub fn parse(maybe_value: ?std.json.Value) !Format {
    const value = maybe_value orelse return .text;
    const object = switch (value) {
        .null => return .text,
        .object => |object| object,
        else => return error.InvalidRequestJson,
    };
    const type_value = object.get("type") orelse return error.InvalidRequestJson;
    const format_type = switch (type_value) {
        .string => |string| string,
        else => return error.InvalidRequestJson,
    };
    if (std.mem.eql(u8, format_type, "text")) return .text;
    if (std.mem.eql(u8, format_type, "json_object")) return .json_object;
    if (std.mem.eql(u8, format_type, "json_schema")) {
        const json_schema_value = object.get("json_schema") orelse return error.InvalidRequestJson;
        if (json_schema_value != .object) return error.InvalidRequestJson;
        return .{ .json_schema = json_schema_value };
    }
    return error.UnsupportedChatParameter;
}

fn appendQwenSystemMessage(gpa: std.mem.Allocator, prompt: *std.ArrayList(u8), content: []const u8) !void {
    try prompt.appendSlice(gpa, "<|im_start|>system\n");
    try prompt.appendSlice(gpa, content);
    try prompt.appendSlice(gpa, "<|im_end|>\n");
}

fn serializeJsonValue(gpa: std.mem.Allocator, value: std.json.Value) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();

    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.write(value);
    return out.toOwnedSlice();
}

fn appendSafeInstructionContent(gpa: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    try qwen_escape.appendSafeContent(gpa, out, text);
}

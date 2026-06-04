//! OpenAI-compatible error response serialization.

const std = @import("std");
const http = std.http;

pub const cors_allow_headers =
    "authorization, content-type, openai-beta, openai-organization, openai-project, " ++
    "x-request-id, x-stainless-lang, x-stainless-package-version, x-stainless-os, " ++
    "x-stainless-arch, x-stainless-runtime, x-stainless-runtime-version, " ++
    "x-stainless-async, x-stainless-retry-count, x-stainless-timeout";

pub const cors_headers = [_]http.Header{
    .{ .name = "access-control-allow-origin", .value = "*" },
    .{ .name = "access-control-expose-headers", .value = "content-type" },
};

pub const json_ct = cors_headers ++ [_]http.Header{
    .{ .name = "content-type", .value = "application/json" },
};

pub const cors_preflight_headers = cors_headers ++ [_]http.Header{
    .{ .name = "access-control-allow-methods", .value = "GET, POST, DELETE, OPTIONS" },
    .{ .name = "access-control-allow-headers", .value = cors_allow_headers },
    .{ .name = "access-control-max-age", .value = "600" },
};

pub const Payload = struct {
    code: []const u8,
    message: []const u8,
};

pub fn buildJson(gpa: std.mem.Allocator, code: []const u8, message: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();

    var jw: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("error");
    try jw.beginObject();
    try jw.objectField("message");
    try jw.write(message);
    try jw.objectField("type");
    try jw.write(code);
    try jw.objectField("param");
    try jw.write(null);
    try jw.objectField("code");
    try jw.write(code);
    try jw.endObject();
    try jw.endObject();
    return out.toOwnedSlice();
}

pub fn respondJson(
    gpa: std.mem.Allocator,
    request: *http.Server.Request,
    status: http.Status,
    code: []const u8,
    message: []const u8,
) !void {
    try respondJsonWithConnection(gpa, request, status, code, message, true);
}

pub fn respondJsonAndClose(
    gpa: std.mem.Allocator,
    request: *http.Server.Request,
    status: http.Status,
    code: []const u8,
    message: []const u8,
) !void {
    try respondJsonWithConnection(gpa, request, status, code, message, false);
}

fn respondJsonWithConnection(
    gpa: std.mem.Allocator,
    request: *http.Server.Request,
    status: http.Status,
    code: []const u8,
    message: []const u8,
    keep_alive: bool,
) !void {
    const body = try buildJson(gpa, code, message);
    defer gpa.free(body);
    try request.respond(body, .{
        .status = status,
        .keep_alive = keep_alive,
        .extra_headers = &json_ct,
    });
}

pub fn respondOptions(request: *http.Server.Request) !void {
    try request.respond("", .{
        .status = .no_content,
        .keep_alive = true,
        .extra_headers = &cors_preflight_headers,
    });
}

fn writeSseEvent(writer: *std.Io.Writer, code: []const u8, message: []const u8) !void {
    try writer.writeAll("event: error\n");
    try writer.writeAll("data: ");

    var jw: std.json.Stringify = .{ .writer = writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("error");
    try jw.beginObject();
    try jw.objectField("code");
    try jw.write(code);
    try jw.objectField("message");
    try jw.write(message);
    try jw.endObject();
    try jw.endObject();

    try writer.writeAll("\n\n");
}

pub fn writeSse(stream: *http.BodyWriter, code: []const u8, message: []const u8) !void {
    try writeSseEvent(&stream.writer, code, message);
    try stream.writer.flush();
}

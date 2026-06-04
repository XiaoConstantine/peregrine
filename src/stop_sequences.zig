//! OpenAI stop-sequence parsing and generation helpers.

const std = @import("std");

const Tokenizer = @import("model/tokenizer.zig").Tokenizer;
const utf8_prefix = @import("utf8_prefix.zig");

pub const max_count: usize = 4;
pub const default_chat_stop = "<|im_end|>";

pub const AppliedText = struct {
    text: []const u8,
    stopped: bool,
};

pub const StreamDecision = struct {
    emit_len: usize,
    stop_seen: bool,
};

pub fn parse(gpa: std.mem.Allocator, root: *const std.json.ObjectMap) ![][]u8 {
    const value = root.get("stop") orelse return defaultStop(gpa);
    if (value == .null) return defaultStop(gpa);

    switch (value) {
        .string => |text| {
            if (text.len == 0) return error.UnsupportedSamplingParameter;
            if (std.mem.eql(u8, text, default_chat_stop)) return defaultStop(gpa);
            const sequences = try gpa.alloc([]u8, 2);
            errdefer gpa.free(sequences);
            sequences[0] = try gpa.dupe(u8, text);
            errdefer gpa.free(sequences[0]);
            sequences[1] = try gpa.dupe(u8, default_chat_stop);
            return sequences;
        },
        .array => |array| {
            if (array.items.len > max_count) return error.UnsupportedSamplingParameter;
            var has_default = false;
            for (array.items) |item| {
                const text = try stopString(item);
                if (text.len == 0) return error.UnsupportedSamplingParameter;
                if (std.mem.eql(u8, text, default_chat_stop)) has_default = true;
            }
            const sequence_count = array.items.len + @intFromBool(!has_default);
            const sequences = try gpa.alloc([]u8, sequence_count);
            var initialized: usize = 0;
            errdefer {
                for (sequences[0..initialized]) |sequence| gpa.free(sequence);
                gpa.free(sequences);
            }
            for (array.items) |item| {
                const text = try stopString(item);
                sequences[initialized] = try gpa.dupe(u8, text);
                initialized += 1;
            }
            if (!has_default) {
                sequences[initialized] = try gpa.dupe(u8, default_chat_stop);
                initialized += 1;
            }
            return sequences;
        },
        else => return error.InvalidRequestJson,
    }
}

fn stopString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |string| string,
        else => error.InvalidRequestJson,
    };
}

fn defaultStop(gpa: std.mem.Allocator) ![][]u8 {
    const sequences = try gpa.alloc([]u8, 1);
    errdefer gpa.free(sequences);
    sequences[0] = try gpa.dupe(u8, default_chat_stop);
    return sequences;
}

pub fn free(gpa: std.mem.Allocator, sequences: [][]u8) void {
    for (sequences) |sequence| gpa.free(sequence);
    gpa.free(sequences);
}

pub fn encode(gpa: std.mem.Allocator, tokenizer: *const Tokenizer, sequences: []const []const u8) ![][]u32 {
    const token_sequences = try gpa.alloc([]u32, sequences.len);
    var initialized: usize = 0;
    errdefer {
        for (token_sequences[0..initialized]) |sequence| gpa.free(sequence);
        gpa.free(token_sequences);
    }
    for (sequences, token_sequences) |sequence, *token_slot| {
        const token_ids = try tokenizer.encode(gpa, sequence);
        if (token_ids.len == 0) return error.EmptyStopSequence;
        token_slot.* = token_ids;
        initialized += 1;
    }
    return token_sequences;
}

pub fn freeTokenSequences(gpa: std.mem.Allocator, sequences: [][]u32) void {
    for (sequences) |sequence| gpa.free(sequence);
    gpa.free(sequences);
}

pub fn apply(text: []const u8, sequences: []const []const u8) AppliedText {
    const stop_index = earliestStopIndex(text, sequences) orelse return .{ .text = text, .stopped = false };
    return .{ .text = text[0..stop_index], .stopped = true };
}

pub fn generatedEndsWith(generated_ids: []const u32, token_sequences: []const []const u32) bool {
    for (token_sequences) |sequence| {
        if (sequence.len == 0 or sequence.len > generated_ids.len) continue;
        if (std.mem.eql(u32, generated_ids[generated_ids.len - sequence.len ..], sequence)) return true;
    }
    return false;
}

pub fn streamEmitDecision(bytes: []const u8, sequences: []const []const u8) StreamDecision {
    const valid_len = utf8_prefix.validPrefixLen(bytes);
    if (valid_len == 0) return .{ .emit_len = 0, .stop_seen = false };
    const valid = bytes[0..valid_len];
    if (earliestStopIndex(valid, sequences)) |stop_index| {
        return .{ .emit_len = stop_index, .stop_seen = true };
    }
    if (sequences.len == 0) return .{ .emit_len = valid_len, .stop_seen = false };
    return .{
        .emit_len = valid_len - longestStopPrefixSuffixLen(valid, sequences),
        .stop_seen = false,
    };
}

fn earliestStopIndex(bytes: []const u8, sequences: []const []const u8) ?usize {
    var earliest: ?usize = null;
    for (sequences) |sequence| {
        if (sequence.len == 0) continue;
        const index = std.mem.indexOf(u8, bytes, sequence) orelse continue;
        if (earliest == null or index < earliest.?) earliest = index;
    }
    return earliest;
}

fn longestStopPrefixSuffixLen(bytes: []const u8, sequences: []const []const u8) usize {
    var longest: usize = 0;
    for (sequences) |sequence| {
        if (sequence.len == 0) continue;
        const max_len = @min(bytes.len, sequence.len - 1);
        var len: usize = 1;
        while (len <= max_len) : (len += 1) {
            if (std.mem.eql(u8, bytes[bytes.len - len ..], sequence[0..len])) longest = @max(longest, len);
        }
    }
    return longest;
}

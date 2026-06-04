//! Shared escaping for text embedded in Qwen chat-template control-token space.

const std = @import("std");

pub fn appendSafeContent(gpa: std.mem.Allocator, out: *std.ArrayList(u8), text: []const u8) !void {
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, text, start, "<|im_")) |index| {
        try out.appendSlice(gpa, text[start..index]);
        try out.appendSlice(gpa, "<\\|im_");
        start = index + "<|im_".len;
    }
    try out.appendSlice(gpa, text[start..]);
}

const std = @import("std");

pub fn validPrefixLen(bytes: []const u8) usize {
    var index: usize = 0;
    while (index < bytes.len) {
        const sequence_len = std.unicode.utf8ByteSequenceLength(bytes[index]) catch return index;
        if (index + sequence_len > bytes.len) return index;
        _ = std.unicode.utf8Decode(bytes[index..][0..sequence_len]) catch return index;
        index += sequence_len;
    }
    return bytes.len;
}

pub fn prefixLenIncludingFirstInvalidByte(bytes: []const u8) usize {
    var index: usize = 0;
    while (index < bytes.len) {
        const sequence_len = std.unicode.utf8ByteSequenceLength(bytes[index]) catch return index + 1;
        if (index + sequence_len > bytes.len) return index;
        _ = std.unicode.utf8Decode(bytes[index..][0..sequence_len]) catch return index + 1;
        index += sequence_len;
    }
    return bytes.len;
}

test "validPrefixLen stops before incomplete or invalid utf8" {
    try std.testing.expectEqual(@as(usize, 5), validPrefixLen("hello"));
    try std.testing.expectEqual(@as(usize, 0), validPrefixLen(&.{0xC3}));
    try std.testing.expectEqual(@as(usize, 2), validPrefixLen(&.{ 'o', 'k', 0xE2, 0x82 }));
    try std.testing.expectEqual(@as(usize, 1), validPrefixLen(&.{ 'a', 0x80 }));
}

test "prefixLenIncludingFirstInvalidByte preserves final lossy flush behavior" {
    try std.testing.expectEqual(@as(usize, 5), prefixLenIncludingFirstInvalidByte("hello"));
    try std.testing.expectEqual(@as(usize, 0), prefixLenIncludingFirstInvalidByte(&.{0xC3}));
    try std.testing.expectEqual(@as(usize, 2), prefixLenIncludingFirstInvalidByte(&.{ 'o', 'k', 0xE2, 0x82 }));
    try std.testing.expectEqual(@as(usize, 2), prefixLenIncludingFirstInvalidByte(&.{ 'a', 0x80 }));
}

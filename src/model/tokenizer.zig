//! Byte-level BPE tokenizer for Qwen3.5 (GPT-2/cl100k family). Loads the vocab +
//! merges from tokenizer.json. This file holds the loader, the byte-level maps,
//! and DECODE (id -> text); ENCODE (pre-tokenizer + BPE) is added on top.
//! No BOS; the eos is <|im_end|>. Validated token-exact against the HF tokenizer
//! on ASCII prompts — full-unicode pre-tokenization is a documented limitation.

const std = @import("std");

const MAX_TOKEN_ID = 1_000_000;

const SpecialToken = struct {
    content: []const u8,
    id: u32,
};

/// GPT-2 byte<->unicode map: every byte gets a printable codepoint so the BPE
/// operates on text. Printable ASCII/Latin bytes map to themselves; the rest are
/// shifted into 256.. . Deterministic — computed, not stored.
const ByteMap = struct {
    to_cp: [256]u21, // byte -> codepoint
    from_cp: [324]i16, // codepoint -> byte (-1 if none); max cp is 323

    fn init() ByteMap {
        var m: ByteMap = .{ .to_cp = undefined, .from_cp = [_]i16{-1} ** 324 };
        var used = [_]bool{false} ** 256;
        // The "printable" ranges that map to themselves.
        const ranges = [_][2]u16{ .{ '!', '~' }, .{ 0xA1, 0xAC }, .{ 0xAE, 0xFF } };
        for (ranges) |r| {
            var b: u16 = r[0];
            while (b <= r[1]) : (b += 1) {
                m.to_cp[b] = @intCast(b);
                used[b] = true;
            }
        }
        var n: u21 = 0;
        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            if (!used[b]) {
                m.to_cp[b] = 256 + n;
                n += 1;
            }
        }
        for (0..256) |i| m.from_cp[@intCast(m.to_cp[i])] = @intCast(i);
        return m;
    }
};

fn encodeByteCodepoint(codepoint: u21, out: *[4]u8) !usize {
    std.debug.assert(codepoint < 324);
    return std.unicode.utf8Encode(codepoint, out) catch return error.InvalidTokenizer;
}

fn encodedSymbolByteLen(first_byte: u8) !usize {
    return std.unicode.utf8ByteSequenceLength(first_byte) catch return error.InvalidTokenizer;
}

fn jsonObject(v: std.json.Value) !std.json.ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => error.InvalidTokenizer,
    };
}

fn jsonArray(v: std.json.Value) !std.json.Array {
    return switch (v) {
        .array => |a| a,
        else => error.InvalidTokenizer,
    };
}

fn jsonString(v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        else => error.InvalidTokenizer,
    };
}

fn jsonU32Value(v: std.json.Value) !u32 {
    return switch (v) {
        .integer => |n| blk: {
            if (n < 0) return error.InvalidTokenizer;
            const id = std.math.cast(u32, n) orelse return error.InvalidTokenizer;
            if (id > MAX_TOKEN_ID) return error.InvalidTokenizer;
            break :blk id;
        },
        else => error.InvalidTokenizer,
    };
}

fn jsonFieldObject(o: *const std.json.ObjectMap, field: []const u8) !std.json.ObjectMap {
    return jsonObject(o.get(field) orelse return error.InvalidTokenizer);
}

fn jsonFieldArray(o: *const std.json.ObjectMap, field: []const u8) !std.json.Array {
    return jsonArray(o.get(field) orelse return error.InvalidTokenizer);
}

fn jsonFieldString(o: *const std.json.ObjectMap, field: []const u8) ![]const u8 {
    return jsonString(o.get(field) orelse return error.InvalidTokenizer);
}

fn jsonFieldU32(o: *const std.json.ObjectMap, field: []const u8) !u32 {
    return jsonU32Value(o.get(field) orelse return error.InvalidTokenizer);
}

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    bytes: ByteMap,
    /// token string (byte-level encoded) -> id
    vocab: std.StringHashMapUnmanaged(u32),
    /// id -> token (byte-level string for normal tokens; literal content for
    /// special/added tokens, flagged so decode emits them verbatim)
    id_to_token: [][]const u8,
    id_is_special: []bool,
    /// Added tokens that must be matched before byte-level pre-tokenization.
    special_tokens: []SpecialToken,
    /// merge priority: key "left\x20right" -> rank (lower = merged first). Spaces
    /// are safe separators (byte-level strings never contain a literal 0x20).
    merge_rank: std.StringHashMapUnmanaged(u32),
    eos_id: u32,

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Load tokenizer.json from `dir_path`.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
        defer dir.close(io);
        const json_bytes = try dir.readFileAlloc(io, "tokenizer.json", gpa, .limited(128 * 1024 * 1024));
        defer gpa.free(json_bytes);

        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, json_bytes, .{});
        defer parsed.deinit();
        const root = try jsonObject(parsed.value);
        const model = try jsonFieldObject(&root, "model");

        // vocab: { token: id }
        const vocab_obj = try jsonFieldObject(&model, "vocab");
        var vocab: std.StringHashMapUnmanaged(u32) = .empty;
        var seen_ids: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer seen_ids.deinit(gpa);
        var max_id: u32 = 0;
        {
            var it = vocab_obj.iterator();
            while (it.next()) |e| {
                const id = try jsonU32Value(e.value_ptr.*);
                if ((try seen_ids.getOrPut(gpa, id)).found_existing) return error.InvalidTokenizer;
                try vocab.put(a, try a.dupe(u8, e.key_ptr.*), id);
                if (id > max_id) max_id = id;
            }
        }

        // added/special tokens: { id, content }
        var added: std.AutoHashMapUnmanaged(u32, []const u8) = .empty;
        defer added.deinit(gpa);
        var special_tokens: std.ArrayList(SpecialToken) = .empty;
        var eos_id: u32 = 0;
        var saw_eos = false;
        if (root.get("added_tokens")) |at| {
            const added_tokens = switch (at) {
                .null => null,
                .array => |arr| arr,
                else => return error.InvalidTokenizer,
            };
            if (added_tokens) |arr| {
                for (arr.items) |tok| {
                    const o = try jsonObject(tok);
                    const id = try jsonFieldU32(&o, "id");
                    if ((try seen_ids.getOrPut(gpa, id)).found_existing) return error.InvalidTokenizer;
                    const content = try a.dupe(u8, try jsonFieldString(&o, "content"));
                    try added.put(gpa, id, content);
                    try special_tokens.append(a, .{ .content = content, .id = id });
                    if (id > max_id) max_id = id;
                    if (std.mem.eql(u8, content, "<|im_end|>")) {
                        eos_id = id;
                        saw_eos = true;
                    }
                }
            }
        }
        if (!saw_eos) return error.InvalidTokenizer;
        std.mem.sort(SpecialToken, special_tokens.items, {}, struct {
            fn lt(_: void, lhs: SpecialToken, rhs: SpecialToken) bool {
                return lhs.content.len > rhs.content.len;
            }
        }.lt);

        // id -> token table (normal byte-level strings + special contents)
        const token_count = std.math.add(usize, @as(usize, max_id), 1) catch return error.InvalidTokenizer;
        const id_to_token = try a.alloc([]const u8, token_count);
        const id_is_special = try a.alloc(bool, token_count);
        @memset(id_to_token, "");
        @memset(id_is_special, false);
        {
            var it = vocab.iterator();
            while (it.next()) |e| {
                id_to_token[e.value_ptr.*] = e.key_ptr.*;
            }
        }
        {
            var it = added.iterator();
            while (it.next()) |e| {
                id_to_token[e.key_ptr.*] = e.value_ptr.*;
                id_is_special[e.key_ptr.*] = true;
            }
        }

        // merges: [[left, right], ...], rank = index
        var merge_rank: std.StringHashMapUnmanaged(u32) = .empty;
        const merges = try jsonFieldArray(&model, "merges");
        for (merges.items, 0..) |m, rank| {
            const pair = try jsonArray(m);
            if (pair.items.len != 2) return error.InvalidTokenizer;
            const left = try jsonString(pair.items[0]);
            const right = try jsonString(pair.items[1]);
            const key = try std.fmt.allocPrint(a, "{s} {s}", .{ left, right });
            const entry = try merge_rank.getOrPut(a, key);
            if (entry.found_existing) return error.InvalidTokenizer;
            entry.value_ptr.* = std.math.cast(u32, rank) orelse return error.InvalidTokenizer;
        }

        return .{
            .arena = arena,
            .bytes = ByteMap.init(),
            .vocab = vocab,
            .id_to_token = id_to_token,
            .id_is_special = id_is_special,
            .special_tokens = try special_tokens.toOwnedSlice(a),
            .merge_rank = merge_rank,
            .eos_id = eos_id,
        };
    }

    /// Decode ids to UTF-8 text. Normal tokens are byte-level decoded; special
    /// tokens are emitted as their literal content. Caller owns the result.
    pub fn decode(self: *const Tokenizer, gpa: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (ids) |id| {
            if (id >= self.id_to_token.len) return error.InvalidTokenId;
            const tok = self.id_to_token[id];
            if (self.id_is_special[id]) {
                try out.appendSlice(gpa, tok);
                continue;
            }
            // byte-level decode: each codepoint of `tok` maps back to one byte.
            var i: usize = 0;
            while (i < tok.len) {
                const cp_len = std.unicode.utf8ByteSequenceLength(tok[i]) catch return error.InvalidUtf8;
                if (i + cp_len > tok.len) return error.InvalidUtf8; // truncated multibyte
                const cp = std.unicode.utf8Decode(tok[i .. i + cp_len]) catch return error.InvalidUtf8;
                if (cp >= self.bytes.from_cp.len or self.bytes.from_cp[cp] < 0) return error.InvalidByteMap;
                try out.append(gpa, @intCast(self.bytes.from_cp[cp]));
                i += cp_len;
            }
        }
        return out.toOwnedSlice(gpa);
    }

    /// Encode UTF-8 text to ids: pre-tokenize, byte-level encode each pre-token,
    /// then BPE-merge. Caller owns the result. Token-exact vs HF on ASCII input;
    /// the pre-tokenizer approximates the unicode \p{L}/\p{N} classes with ASCII.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        var enc: std.ArrayList(u8) = .empty; // byte-level encoded pre-token
        defer enc.deinit(gpa);
        var syms: std.ArrayList([]const u8) = .empty;
        defer syms.deinit(gpa);
        var merge_key: std.ArrayList(u8) = .empty;
        defer merge_key.deinit(gpa);

        var i: usize = 0;
        while (i < text.len) {
            if (self.matchSpecial(text[i..])) |special| {
                try out.append(gpa, special.id);
                i += special.content.len;
                continue;
            }
            const end = nextPreToken(text, i);
            std.debug.assert(end > i);
            enc.clearRetainingCapacity();
            for (text[i..end]) |b| {
                var u: [4]u8 = undefined;
                const l = try encodeByteCodepoint(self.bytes.to_cp[b], &u);
                try enc.appendSlice(gpa, u[0..l]);
            }
            try self.bpe(gpa, enc.items, &syms, &merge_key, &out);
            i = end;
        }
        return out.toOwnedSlice(gpa);
    }

    fn matchSpecial(self: *const Tokenizer, text: []const u8) ?SpecialToken {
        for (self.special_tokens) |special| {
            if (special.content.len > 0 and std.mem.startsWith(u8, text, special.content)) {
                return special;
            }
        }
        return null;
    }

    /// BPE on one byte-level-encoded pre-token. `syms` is reusable scratch.
    fn bpe(
        self: *const Tokenizer,
        gpa: std.mem.Allocator,
        word: []const u8,
        syms: *std.ArrayList([]const u8),
        merge_key: *std.ArrayList(u8),
        out: *std.ArrayList(u32),
    ) !void {
        // Fast path: a pre-token that is itself a vocab entry is emitted directly.
        // This is an optimization, not the `ignore_merges` flag (this model has
        // ignore_merges=false): it is correct because every vocab token in this
        // checkpoint BPE-reconstructs to its own id, so the shortcut and the full
        // merge loop yield the same result. (Verified by fuzzing vs HF.)
        if (self.vocab.get(word)) |id| {
            try out.append(gpa, id);
            return;
        }
        syms.clearRetainingCapacity();
        var i: usize = 0;
        while (i < word.len) {
            const l = try encodedSymbolByteLen(word[i]);
            std.debug.assert(i + l <= word.len);
            try syms.append(gpa, word[i .. i + l]);
            i += l;
        }
        while (syms.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_k: ?usize = null;
            for (0..syms.items.len - 1) |k| {
                merge_key.clearRetainingCapacity();
                try merge_key.appendSlice(gpa, syms.items[k]);
                try merge_key.append(gpa, ' ');
                try merge_key.appendSlice(gpa, syms.items[k + 1]);
                if (self.merge_rank.get(merge_key.items)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_k = k;
                    }
                }
            }
            const k = best_k orelse break;
            const left = syms.items[k];
            const right = syms.items[k + 1];
            std.debug.assert(@intFromPtr(left.ptr) + left.len == @intFromPtr(right.ptr));
            syms.items[k] = left.ptr[0 .. left.len + right.len];
            _ = syms.orderedRemove(k + 1);
        }
        for (syms.items) |s| {
            try out.append(gpa, self.vocab.get(s) orelse return error.UnknownToken);
        }
    }
};

fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
}
fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}
fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}
fn lower(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

/// The pre-tokenizer split, one match at a time. Implements (ASCII subset of) the
/// cl100k regex alternation IN ORDER: contractions, optional-prefix word, single
/// digit, optional-space punctuation, whitespace-ending-in-newline, then the
/// `\s+(?!\S)` / `\s+` whitespace rules (the last whitespace of a run before a
/// word attaches to the word). Returns the end index of the pre-token at `i`.
fn nextPreToken(t: []const u8, i: usize) usize {
    const n = t.len;
    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (t[i] == '\'' and i + 1 < n) {
        const c1 = lower(t[i + 1]);
        if (i + 2 < n) {
            const c2 = lower(t[i + 2]);
            if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) return i + 3;
        }
        if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return i + 2;
    }
    // 2. [^\r\n\p{L}\p{N}]?[\p{L}]+  (optional non-letter/digit lead, then letters)
    {
        const opt = t[i] != '\r' and t[i] != '\n' and !isAlpha(t[i]) and !isDigit(t[i]);
        const ls = if (opt) i + 1 else i;
        if (ls < n and isAlpha(t[ls])) {
            var j = ls + 1;
            while (j < n and isAlpha(t[j])) j += 1;
            return j;
        }
    }
    // 3. \p{N}  (a single digit)
    if (isDigit(t[i])) return i + 1;
    // 4. " ?[^\s\p{L}\p{N}]+[\r\n]*"  (optional one space, punctuation run, trailing CRLF)
    {
        var j = i;
        if (t[j] == ' ' and j + 1 < n and !isSpace(t[j + 1]) and !isAlpha(t[j + 1]) and !isDigit(t[j + 1])) j += 1;
        if (j < n and !isSpace(t[j]) and !isAlpha(t[j]) and !isDigit(t[j])) {
            while (j < n and !isSpace(t[j]) and !isAlpha(t[j]) and !isDigit(t[j])) j += 1;
            while (j < n and (t[j] == '\r' or t[j] == '\n')) j += 1;
            return j;
        }
    }
    // run of whitespace [i, j)
    var j = i;
    while (j < n and isSpace(t[j])) j += 1;
    // 5. \s*[\r\n]+  (match up to and including the last CRLF in the run)
    {
        var k = j;
        while (k > i and !(t[k - 1] == '\r' or t[k - 1] == '\n')) k -= 1;
        if (k > i) return k;
    }
    // 6. \s+(?!\S) and 7. \s+
    if (j == n) return j; // trailing whitespace -> all of it
    if (j - i >= 2) return j - 1; // leave the last ws char for the following word
    return i + 1; // a lone ws char before a word
}

test "jsonU32Value validates tokenizer id ranges" {
    try std.testing.expectEqual(@as(u32, 7), try jsonU32Value(.{ .integer = 7 }));
    try std.testing.expectError(error.InvalidTokenizer, jsonU32Value(.{ .integer = -1 }));
    try std.testing.expectError(error.InvalidTokenizer, jsonU32Value(.{ .integer = std.math.maxInt(u32) }));
    try std.testing.expectError(error.InvalidTokenizer, jsonU32Value(.{ .integer = @as(i64, std.math.maxInt(u32)) + 1 }));
    try std.testing.expectError(error.InvalidTokenizer, jsonU32Value(.{ .string = "not an id" }));
}

test "load accepts minimal tokenizer JSON" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "tokenizer.json",
        .data =
        \\{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},"added_tokens":[{"id":3,"content":"<|im_end|>"}]}
        ,
    });
    var path_buf: [4096]u8 = undefined;
    const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);

    var tok = try Tokenizer.load(gpa, std.testing.io, path_buf[0..path_len]);
    defer tok.deinit();
    try std.testing.expectEqual(@as(u32, 3), tok.eos_id);
    try std.testing.expectEqualStrings("<|im_end|>", tok.id_to_token[tok.eos_id]);
}

test "load rejects malformed tokenizer JSON shape" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "tokenizer.json",
        .data =
        \\{"model":{"vocab":{"a":-1},"merges":[["a","b"]]},"added_tokens":[{"id":3,"content":"<|im_end|>"}]}
        ,
    });
    var path_buf: [4096]u8 = undefined;
    const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);

    try std.testing.expectError(error.InvalidTokenizer, Tokenizer.load(gpa, std.testing.io, path_buf[0..path_len]));
}

//! Prompt-prefix cache owner for the minimal HTTP server.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const model_mod = @import("../model/model.zig");
const dims = @import("../model/dims.zig");

const ModelState = model_mod.ModelState;
const hiddenRowsBytes = dims.hiddenRowsBytes;
const log = std.log.scoped(.peregrine_prefix_cache);

/// Stable only until the next cache mutation. Public mutators may evict entries
/// and shift `index`; callers must discard old matches after mutation and keep
/// the `Match` returned by `startDirectEntry`, `storePrefixFromWork`, or
/// `loadEntrySlot`.
pub const Match = struct {
    index: usize,
    len: usize,
};

const PrefixIndex = struct {
    nodes: std.ArrayList(Node) = .empty,
    free_nodes: std.ArrayList(usize) = .empty,

    const Child = struct {
        token: u32,
        node: usize,
    };

    const Node = struct {
        parent: ?usize = null,
        parent_token: u32 = 0,
        children: std.ArrayList(Child) = .empty,
        entry_index: ?usize = null,
        alive: bool = false,

        fn root() Node {
            return .{ .alive = true };
        }

        fn child(parent: usize, token: u32) Node {
            return .{
                .parent = parent,
                .parent_token = token,
                .alive = true,
            };
        }

        fn deinit(self: *Node, gpa: std.mem.Allocator) void {
            self.children.deinit(gpa);
            self.* = .{};
        }
    };

    fn deinit(self: *PrefixIndex, gpa: std.mem.Allocator) void {
        for (self.nodes.items) |*node| node.deinit(gpa);
        self.nodes.deinit(gpa);
        self.free_nodes.deinit(gpa);
        self.* = .{};
    }

    fn clear(self: *PrefixIndex, gpa: std.mem.Allocator) void {
        for (self.nodes.items) |*node| node.deinit(gpa);
        self.nodes.clearRetainingCapacity();
        self.free_nodes.clearRetainingCapacity();
    }

    fn insert(self: *PrefixIndex, gpa: std.mem.Allocator, tokens: []const u32, entry_index: usize) !void {
        if (tokens.len == 0) return;
        try self.ensureRoot(gpa);
        var node_index: usize = 0;
        for (tokens) |token| {
            node_index = if (self.findChild(node_index, token)) |child_index|
                child_index
            else blk: {
                const child_index = try self.allocNode(gpa, node_index, token);
                errdefer self.freeDetachedNode(gpa, child_index);
                try self.nodes.items[node_index].children.append(gpa, .{
                    .token = token,
                    .node = child_index,
                });
                break :blk child_index;
            };
        }
        self.nodes.items[node_index].entry_index = entry_index;
    }

    fn remove(self: *PrefixIndex, gpa: std.mem.Allocator, tokens: []const u32) void {
        const node_index = self.findNode(tokens) orelse return;
        self.nodes.items[node_index].entry_index = null;
        self.pruneFrom(gpa, node_index);
    }

    fn findLongest(self: *const PrefixIndex, prompt: []const u32) ?Match {
        if (self.nodes.items.len == 0) return null;
        var node_index: usize = 0;
        var best: ?Match = null;
        for (prompt, 0..) |token, i| {
            node_index = self.findChild(node_index, token) orelse break;
            if (self.nodes.items[node_index].entry_index) |entry_index| {
                best = .{ .index = entry_index, .len = i + 1 };
            }
        }
        return best;
    }

    fn findExact(self: *const PrefixIndex, tokens: []const u32) ?usize {
        const node_index = self.findNode(tokens) orelse return null;
        return self.nodes.items[node_index].entry_index;
    }

    fn containsPromptPrefix(self: *const PrefixIndex, prompt: []const u32, prefix_len: usize) bool {
        if (prefix_len == 0) return true;
        if (prefix_len > prompt.len or self.nodes.items.len == 0) return false;
        var node_index: usize = 0;
        for (prompt[0..prefix_len]) |token| {
            node_index = self.findChild(node_index, token) orelse return false;
        }
        return self.nodes.items[node_index].entry_index != null;
    }

    fn decrementEntryIndexesAfter(self: *PrefixIndex, removed_index: usize) void {
        for (self.nodes.items) |*node| {
            if (!node.alive) continue;
            if (node.entry_index) |entry_index| {
                if (entry_index == removed_index) {
                    node.entry_index = null;
                } else if (entry_index > removed_index) {
                    node.entry_index = entry_index - 1;
                }
            }
        }
    }

    fn ensureRoot(self: *PrefixIndex, gpa: std.mem.Allocator) !void {
        if (self.nodes.items.len != 0) return;
        try self.nodes.append(gpa, Node.root());
    }

    fn allocNode(self: *PrefixIndex, gpa: std.mem.Allocator, parent: usize, token: u32) !usize {
        if (self.free_nodes.items.len != 0) {
            const index = self.free_nodes.items[self.free_nodes.items.len - 1];
            self.free_nodes.items.len -= 1;
            self.nodes.items[index] = Node.child(parent, token);
            return index;
        }
        const index = self.nodes.items.len;
        try self.nodes.append(gpa, Node.child(parent, token));
        return index;
    }

    fn freeDetachedNode(self: *PrefixIndex, gpa: std.mem.Allocator, node_index: usize) void {
        std.debug.assert(node_index != 0);
        std.debug.assert(self.nodes.items[node_index].children.items.len == 0);
        std.debug.assert(self.nodes.items[node_index].entry_index == null);
        self.nodes.items[node_index].deinit(gpa);
        self.free_nodes.append(gpa, node_index) catch |err| {
            log.warn("prefix index free-list append failed while freeing node {d} ({any})", .{ node_index, err });
        };
    }

    fn pruneFrom(self: *PrefixIndex, gpa: std.mem.Allocator, start_index: usize) void {
        var node_index = start_index;
        while (node_index != 0) {
            const node = &self.nodes.items[node_index];
            if (node.entry_index != null or node.children.items.len != 0) return;
            const parent_index = node.parent.?;
            const parent_token = node.parent_token;
            self.removeChild(parent_index, parent_token, node_index);
            self.freeDetachedNode(gpa, node_index);
            node_index = parent_index;
        }
    }

    fn removeChild(self: *PrefixIndex, parent_index: usize, token: u32, child_index: usize) void {
        const children = &self.nodes.items[parent_index].children;
        for (children.items, 0..) |child, i| {
            if (child.token == token and child.node == child_index) {
                _ = children.orderedRemove(i);
                return;
            }
        }
        std.debug.assert(false);
    }

    fn findNode(self: *const PrefixIndex, tokens: []const u32) ?usize {
        if (self.nodes.items.len == 0) return null;
        var node_index: usize = 0;
        for (tokens) |token| {
            node_index = self.findChild(node_index, token) orelse return null;
        }
        return node_index;
    }

    fn findChild(self: *const PrefixIndex, parent_index: usize, token: u32) ?usize {
        const parent = &self.nodes.items[parent_index];
        if (!parent.alive) return null;
        for (parent.children.items) |child| {
            if (child.token == token) return child.node;
        }
        return null;
    }

    fn hasEntryInSubtree(self: *const PrefixIndex, node_index: usize) bool {
        const node = &self.nodes.items[node_index];
        if (!node.alive) return false;
        if (node.entry_index != null) return true;
        for (node.children.items) |child| {
            if (self.hasEntryInSubtree(child.node)) return true;
        }
        return false;
    }
};

pub const Entry = struct {
    cached_state: ModelState,
    /// Normalized target hidden states for `tokens.items.len` prompt rows,
    /// `[max_seq, HIDDEN]` bf16. Only allocated when hidden tracking is enabled
    /// (MTP sidecar loaded); null otherwise.
    hidden: ?metal.Buffer = null,
    /// Whether `hidden` holds real captured rows for `tokens.items.len` rows.
    /// False when the buffer is allocated-but-zero (e.g. a v3 file written by a
    /// non-MTP run, or capacity allocated before any prefill captured rows).
    /// Used to gate MTP drafter seeding so all-zero hiddens don't silently
    /// produce ~0% acceptance.
    hidden_valid: bool = false,
    cached_next_token: u32 = 0,
    cached_next_token_valid: bool = false,
    tokens: std.ArrayList(u32) = .empty,
    pinned: bool = false,
    last_access: u64 = 0,
    hits: usize = 0,

    fn init(device: *metal.Device, token_capacity: usize, pinned: bool) !Entry {
        var cached_state = try ModelState.initBf16FullCaches(device, token_capacity);
        errdefer cached_state.deinit();
        return .{
            .cached_state = cached_state,
            .pinned = pinned,
        };
    }

    fn deinit(self: *Entry, gpa: std.mem.Allocator) void {
        self.tokens.deinit(gpa);
        if (self.hidden) |*buf| buf.destroy();
        self.cached_state.deinit();
        self.* = undefined;
    }

    fn ensureStateCapacity(self: *Entry, device: *metal.Device, token_count: usize) !void {
        if (token_count == 0) return;
        if (token_count <= self.cached_state.max_seq) return;

        var next = try ModelState.initBf16FullCaches(device, token_count);
        errdefer next.deinit();
        if (self.tokens.items.len > 0) {
            try next.copyPrefixFrom(&self.cached_state, self.tokens.items.len);
        }
        self.cached_state.deinit();
        self.cached_state = next;
    }

    /// Allocate or grow the hidden buffer to at least `token_count` rows.
    /// Existing rows are preserved up to the smaller of old and new capacity.
    fn ensureHiddenCapacity(self: *Entry, device: *metal.Device, token_count: usize) !void {
        const needed = try hiddenRowsBytes(@max(token_count, self.cached_state.max_seq));
        if (self.hidden) |*buf| {
            if (buf.length >= needed) return;
            var next = try device.createSharedBuffer(needed);
            errdefer next.destroy();
            @memset(next.slice(u8), 0);
            // Preserve the old buffer's valid rows. `tokens` may already have
            // been grown past the old buffer's row capacity (replaceEntryPrefix
            // resizes before this runs), so clamp to the old buffer's actual
            // byte size to avoid an over-read.
            const want_bytes = try hiddenRowsBytes(self.tokens.items.len);
            const copy_bytes = @min(want_bytes, buf.length);
            if (copy_bytes != 0) {
                @memcpy(next.slice(u8)[0..copy_bytes], buf.slice(u8)[0..copy_bytes]);
            }
            buf.destroy();
            self.hidden = next;
            return;
        }
        var buf = try device.createSharedBuffer(needed);
        errdefer buf.destroy();
        @memset(buf.slice(u8), 0);
        self.hidden = buf;
    }
};

pub const Cache = struct {
    entries: std.ArrayList(Entry) = .empty,
    index: PrefixIndex = .{},
    max_tokens: usize,
    initial_state_tokens: usize,
    resident_tokens: usize = 0,
    reserved_tokens: usize = 0,
    active_index: ?usize = null,
    access_clock: u64 = 0,
    hits: usize = 0,
    misses: usize = 0,
    hit_tokens: usize = 0,
    computed_tokens: usize = 0,
    evictions: usize = 0,
    /// When true, entries allocate and maintain a normalized-hidden buffer for
    /// MTP drafter seeding. Flipped on by the server once the MTP sidecar is
    /// loaded; stays off for the normal decode-only path so non-MTP serving
    /// pays no extra device memory.
    track_hidden: bool = false,

    pub fn init(
        max_cache_tokens: usize,
        prefill_chunk_tokens: usize,
    ) !Cache {
        return .{
            .max_tokens = max_cache_tokens,
            .initial_state_tokens = @min(max_cache_tokens, prefill_chunk_tokens),
        };
    }

    pub fn deinit(self: *Cache, gpa: std.mem.Allocator) void {
        for (self.entries.items) |*entry| entry.deinit(gpa);
        self.entries.deinit(gpa);
        self.index.deinit(gpa);
        self.* = undefined;
    }

    pub fn clearAll(self: *Cache, gpa: std.mem.Allocator) void {
        for (self.entries.items) |*entry| entry.deinit(gpa);
        self.entries.clearRetainingCapacity();
        self.index.clear(gpa);
        self.resident_tokens = 0;
        self.reserved_tokens = 0;
        self.active_index = null;
    }

    pub fn entryCount(self: *const Cache) usize {
        return self.entries.items.len;
    }

    pub fn pinnedEntryCount(self: *const Cache) usize {
        var n: usize = 0;
        for (self.entries.items) |entry| {
            if (entry.pinned) n += 1;
        }
        return n;
    }

    pub fn activeTokenCount(self: *const Cache) usize {
        const entry = self.activeEntryConst() orelse return 0;
        return entry.tokens.items.len;
    }

    pub fn activeCachedNextTokenValid(self: *const Cache) bool {
        const entry = self.activeEntryConst() orelse return false;
        return entry.cached_next_token_valid;
    }

    pub fn activeEntry(self: *Cache) ?*Entry {
        const index = self.active_index orelse return null;
        if (index >= self.entries.items.len) return null;
        return &self.entries.items[index];
    }

    pub fn activeEntryConst(self: *const Cache) ?*const Entry {
        const index = self.active_index orelse return null;
        if (index >= self.entries.items.len) return null;
        return &self.entries.items[index];
    }

    pub fn persistedEntryConst(self: *const Cache) ?*const Entry {
        if (self.activeEntryConst()) |entry| return entry;
        var best: ?usize = null;
        for (self.entries.items, 0..) |entry, i| {
            if (entry.tokens.items.len == 0) continue;
            if (best == null or entry.last_access > self.entries.items[best.?].last_access) {
                best = i;
            }
        }
        return if (best) |i| &self.entries.items[i] else null;
    }

    pub fn findPrefix(self: *const Cache, prompt: []const u32) ?Match {
        return self.index.findLongest(prompt);
    }

    pub fn containsPromptPrefix(self: *const Cache, prompt: []const u32, prefix_len: usize) bool {
        return self.index.containsPromptPrefix(prompt, prefix_len);
    }

    pub fn touch(self: *Cache, match: Match) void {
        if (match.index >= self.entries.items.len) return;
        self.access_clock +%= 1;
        const entry = &self.entries.items[match.index];
        entry.last_access = self.access_clock;
        entry.hits += 1;
        self.active_index = match.index;
    }

    pub fn recordHit(self: *Cache, reused_tokens: usize) void {
        self.hits += 1;
        self.hit_tokens += reused_tokens;
    }

    pub fn recordMiss(self: *Cache) void {
        self.misses += 1;
    }

    /// Record tokens computed fresh (not served from cache) for this request.
    /// Paired with recordHit's hit_tokens to give a token-level cache hit
    /// ratio: hit_tokens / (hit_tokens + computed_tokens).
    pub fn recordComputedTokens(self: *Cache, token_count: usize) void {
        self.computed_tokens += token_count;
    }

    /// Cumulative cache statistics for observability. `hits` counts requests
    /// that reused a cached prefix; `hit_tokens` is the total reused token
    /// count across those hits; `computed_tokens` is the total tokens
    /// computed fresh (prompt_tokens - reuse_tokens summed over requests).
    /// The token-level hit ratio hit_tokens / (hit_tokens + computed_tokens)
    /// is the most meaningful efficiency metric: it says what fraction of
    /// prompt tokens were served from the cache vs computed fresh.
    pub const Stats = struct {
        hits: usize,
        misses: usize,
        hit_tokens: usize,
        computed_tokens: usize,
        reserved_tokens: usize,
        max_tokens: usize,
        entry_count: usize,
        evictions: usize,
    };

    pub fn stats(self: *const Cache) Stats {
        return .{
            .hits = self.hits,
            .misses = self.misses,
            .hit_tokens = self.hit_tokens,
            .computed_tokens = self.computed_tokens,
            .reserved_tokens = self.reserved_tokens,
            .max_tokens = self.max_tokens,
            .entry_count = self.entries.items.len,
            .evictions = self.evictions,
        };
    }

    /// Enable hidden-state tracking for subsequent stores and loads. Must be
    /// called before any entry that should carry a hidden buffer is created.
    pub fn enableHiddenTracking(self: *Cache) void {
        self.track_hidden = true;
    }

    pub fn hiddenTrackingEnabled(self: *const Cache) bool {
        return self.track_hidden;
    }

    /// Returns the entry's normalized-hidden buffer for `match`, or null when
    /// hidden tracking is off or the entry has no hidden buffer.
    pub fn entryHidden(self: *const Cache, match: Match) ?metal.Buffer {
        if (match.index >= self.entries.items.len) return null;
        return self.entries.items[match.index].hidden;
    }

    /// Whether `entryHidden`'s buffer holds real captured rows for `match.len`
    /// rows. False for an allocated-but-zero buffer (e.g. a v3 file written
    /// without hiddens). Callers that gate on real hiddens (MTP drafter
    /// seeding) should check this rather than nullness.
    pub fn entryHiddenValid(self: *const Cache, match: Match) bool {
        if (match.index >= self.entries.items.len) return false;
        return self.entries.items[match.index].hidden_valid;
    }

    /// Copy `src`'s first `match.len` hidden rows into the entry's hidden
    /// buffer, allocating it if needed. Used by the direct-entry path which
    /// bypasses `storePrefixFromWork`.
    pub fn setEntryHidden(self: *Cache, device: *metal.Device, match: Match, src: metal.Buffer) !void {
        if (match.index >= self.entries.items.len) return error.EmptyPrefix;
        const entry = &self.entries.items[match.index];
        try entry.ensureHiddenCapacity(device, match.len);
        if (entry.hidden) |dst| {
            const copy_bytes = try hiddenRowsBytes(match.len);
            if (src.length < copy_bytes) return error.InputBufferTooSmall;
            @memcpy(dst.slice(u8)[0..copy_bytes], src.slice(u8)[0..copy_bytes]);
            entry.hidden_valid = true;
        }
    }

    pub fn cachedNextTokenValid(self: *const Cache, match: Match) bool {
        return self.entries.items[match.index].cached_next_token_valid;
    }

    pub fn cachedNextToken(self: *const Cache, match: Match) u32 {
        return self.entries.items[match.index].cached_next_token;
    }

    pub fn prefixState(self: *const Cache, match: ?Match) ?model_mod.DeviceModel.PrefixState {
        const m = match orelse return null;
        if (m.len == 0) return null;
        const entry = &self.entries.items[m.index];
        return .{ .state = &entry.cached_state, .len = @intCast(m.len) };
    }

    pub fn entryState(self: *const Cache, match: Match) *const ModelState {
        return &self.entries.items[match.index].cached_state;
    }

    pub fn entryStateMut(self: *Cache, match: Match) *ModelState {
        return &self.entries.items[match.index].cached_state;
    }

    /// Remove an entry by current index. Invalidates all existing `Match` values
    /// because `orderedRemove` may shift later entry indexes.
    pub fn removeEntry(self: *Cache, gpa: std.mem.Allocator, index: usize) void {
        if (index >= self.entries.items.len) return;
        self.index.remove(gpa, self.entries.items[index].tokens.items);
        self.resident_tokens -|= self.entries.items[index].tokens.items.len;
        self.reserved_tokens -|= self.entries.items[index].cached_state.max_seq;
        var removed = self.entries.orderedRemove(index);
        removed.deinit(gpa);
        self.index.decrementEntryIndexesAfter(index);
        if (self.active_index) |active| {
            if (active == index) {
                self.active_index = null;
            } else if (active > index) {
                self.active_index = active - 1;
            }
        }
    }

    /// Reserve or reuse an entry for a full direct prefill. Invalidates all
    /// existing `Match` values; callers must use the returned match for the
    /// matching `finishDirectEntry` or rollback.
    pub fn startDirectEntry(
        self: *Cache,
        gpa: std.mem.Allocator,
        device: *metal.Device,
        prefix_ids: []const u32,
        pinned: bool,
    ) !Match {
        if (prefix_ids.len == 0 or prefix_ids.len > self.max_tokens) return error.SequenceTooLong;
        const match = try self.prepareEntry(gpa, device, prefix_ids, null, pinned);
        const entry = &self.entries.items[match.index];
        entry.cached_state.resetForNewSequence();
        entry.cached_next_token_valid = false;
        return match;
    }

    pub fn finishDirectEntry(self: *Cache, match: Match, next_token: metal.Buffer) !void {
        if (next_token.length < @sizeOf(u32)) return error.InputBufferTooSmall;
        const entry = &self.entries.items[match.index];
        entry.cached_next_token = next_token.slice(u32)[0];
        entry.cached_next_token_valid = true;
        self.touch(match);
    }

    /// Store a computed prefix from a work state. This may evict entries and
    /// shift indexes, including `reuse_match`; callers must discard old matches
    /// and use the returned `Match` for later cache access. When hidden
    /// tracking is enabled and `work_hidden` is provided, its rows starting at
    /// `work_hidden_base_rows` (length `prefix_ids.len - reuse_len`) are copied
    /// into the entry's hidden buffer after the preserved reused prefix rows.
    pub fn storePrefixFromWork(
        self: *Cache,
        gpa: std.mem.Allocator,
        device: *metal.Device,
        prefix_ids: []const u32,
        reuse_match: ?Match,
        work_state: *const ModelState,
        work_hidden: ?metal.Buffer,
        work_hidden_base_rows: usize,
        next_token: ?metal.Buffer,
        pinned: bool,
    ) !Match {
        if (prefix_ids.len == 0 or prefix_ids.len > self.max_tokens) return error.SequenceTooLong;
        const had_exact_entry = self.findExact(prefix_ids) != null;
        const match = try self.prepareEntry(gpa, device, prefix_ids, reuse_match, pinned);
        const entry = &self.entries.items[match.index];
        const reuse_len = if (reuse_match) |reuse| @min(reuse.len, prefix_ids.len) else 0;

        if (!had_exact_entry) {
            if (reuse_match != null) {
                if (prefix_ids.len > reuse_len) {
                    try entry.cached_state.copyFullAttentionRangeFrom(reuse_len, work_state, 0, prefix_ids.len - reuse_len);
                    try entry.cached_state.copyLinearStateFrom(work_state);
                }
            } else if (reuse_len == 0) {
                try entry.cached_state.copyPrefixFrom(work_state, prefix_ids.len);
            }
        }

        if (self.track_hidden) {
            // Whether the reused prefix rows in the entry are already valid.
            // ensureHiddenCapacity preserves the old buffer's rows in-place
            // (clamped to the old buffer size), so we only need to track
            // validity, not re-copy here.
            //
            // Read validity from `entry` (match.index, adjusted) rather than
            // reuse_match.index: ensureReservedTokenBudget can evict entries
            // with smaller indices, shifting the reuse entry down, so the
            // caller's reuse_match.index may be stale and point at a different
            // entry. On the no-exact-match path (the only one current callers
            // reach), `entry` IS the reused entry, so its flag is correct.
            const reuse_hidden_valid = if (reuse_len != 0) entry.hidden_valid else false;
            try entry.ensureHiddenCapacity(device, prefix_ids.len);
            if (entry.hidden) |dst| {
                const suffix_rows = prefix_ids.len - reuse_len;
                if (suffix_rows != 0) {
                    const suffix_bytes = try hiddenRowsBytes(suffix_rows);
                    const dst_off = try hiddenRowsBytes(reuse_len);
                    if (work_hidden) |src| {
                        const src_off = try hiddenRowsBytes(work_hidden_base_rows);
                        if (src.length < src_off + suffix_bytes) return error.InputBufferTooSmall;
                        @memcpy(dst.slice(u8)[dst_off..][0..suffix_bytes], src.slice(u8)[src_off..][0..suffix_bytes]);
                    } else {
                        @memset(dst.slice(u8)[dst_off..][0..suffix_bytes], 0);
                    }
                }
                // The buffer is valid when both halves are: the reused prefix
                // rows (0..reuse_len) are valid iff the reused entry's hiddens
                // were valid, and the suffix rows (reuse_len..prefix_ids.len)
                // are valid iff they were filled from a real captured buffer
                // (or there is no suffix). When reuse_len == 0 (fresh entry),
                // only the suffix half matters.
                const prefix_half_valid = (reuse_len == 0) or reuse_hidden_valid;
                const suffix_half_valid = (suffix_rows == 0) or (work_hidden != null);
                entry.hidden_valid = prefix_half_valid and suffix_half_valid;
            }
        }

        if (next_token) |buf| {
            if (buf.length < @sizeOf(u32)) return error.InputBufferTooSmall;
            entry.cached_next_token = buf.slice(u32)[0];
            entry.cached_next_token_valid = true;
        } else {
            entry.cached_next_token_valid = false;
        }
        self.touch(match);
        return match;
    }

    /// Clear the cache and allocate a pinned entry for persisted-state loading.
    /// Invalidates all existing `Match` values; callers must use the returned
    /// match and call `finishLoadedEntry` after token ids are read.
    pub fn loadEntrySlot(
        self: *Cache,
        gpa: std.mem.Allocator,
        device: *metal.Device,
        token_count: usize,
    ) !Match {
        self.clearAll(gpa);
        if (token_count == 0 or token_count > self.max_tokens) return error.SequenceTooLong;
        var entry = try Entry.init(device, @max(token_count, self.initial_state_tokens), true);
        errdefer entry.deinit(gpa);
        if (self.track_hidden) {
            try entry.ensureHiddenCapacity(device, token_count);
        }
        try entry.tokens.resize(gpa, token_count);
        entry.last_access = self.nextAccess();
        try self.entries.append(gpa, entry);
        self.resident_tokens = token_count;
        self.reserved_tokens = self.entries.items[self.entries.items.len - 1].cached_state.max_seq;
        self.active_index = self.entries.items.len - 1;
        return .{ .index = self.entries.items.len - 1, .len = token_count };
    }

    pub fn finishLoadedEntry(self: *Cache, gpa: std.mem.Allocator, match: Match) !void {
        if (match.index >= self.entries.items.len) return error.EmptyPrefix;
        const entry = &self.entries.items[match.index];
        if (entry.tokens.items.len != match.len) return error.SequenceTooLong;
        try self.index.insert(gpa, entry.tokens.items, match.index);
    }

    fn prepareEntry(
        self: *Cache,
        gpa: std.mem.Allocator,
        device: *metal.Device,
        prefix_ids: []const u32,
        reuse_match: ?Match,
        pinned: bool,
    ) !Match {
        if (prefix_ids.len == 0 or prefix_ids.len > self.max_tokens) return error.SequenceTooLong;
        if (self.findExact(prefix_ids)) |index| {
            var preserved_index: ?usize = index;
            try self.replaceEntryPrefix(gpa, device, &preserved_index, prefix_ids, pinned);
            return .{ .index = preserved_index.?, .len = prefix_ids.len };
        }

        const reuse_index = if (reuse_match) |reuse| reuse.index else null;
        if (reuse_index) |index| {
            var preserved_index: ?usize = index;
            const entry = &self.entries.items[preserved_index.?];
            const old_tokens = try gpa.dupe(u32, entry.tokens.items);
            defer gpa.free(old_tokens);
            try self.index.insert(gpa, prefix_ids, preserved_index.?);
            var new_index_registered = true;
            errdefer if (new_index_registered) self.index.remove(gpa, prefix_ids);
            try self.replaceEntryPrefix(gpa, device, &preserved_index, prefix_ids, pinned);
            self.index.remove(gpa, old_tokens);
            new_index_registered = false;
            return .{ .index = preserved_index.?, .len = prefix_ids.len };
        }

        var no_preserve: ?usize = null;
        const state_capacity = self.entryStateCapacity(prefix_ids.len);
        try self.ensureReservedTokenBudget(gpa, &no_preserve, state_capacity);
        var entry = try Entry.init(device, state_capacity, pinned);
        errdefer entry.deinit(gpa);
        try entry.tokens.appendSlice(gpa, prefix_ids);
        entry.last_access = self.nextAccess();
        const new_index = self.entries.items.len;
        try self.index.insert(gpa, entry.tokens.items, new_index);
        var registered = true;
        errdefer if (registered) self.index.remove(gpa, entry.tokens.items);
        try self.entries.append(gpa, entry);
        registered = false;
        self.resident_tokens += prefix_ids.len;
        self.reserved_tokens += state_capacity;
        return .{ .index = self.entries.items.len - 1, .len = prefix_ids.len };
    }

    fn replaceEntryPrefix(
        self: *Cache,
        gpa: std.mem.Allocator,
        device: *metal.Device,
        preserved_index: *?usize,
        prefix_ids: []const u32,
        pinned: bool,
    ) !void {
        var entry = &self.entries.items[preserved_index.*.?];
        const old_len = entry.tokens.items.len;
        const old_capacity = entry.cached_state.max_seq;
        if (prefix_ids.len > old_capacity) {
            try self.ensureReservedTokenBudget(gpa, preserved_index, prefix_ids.len - old_capacity);
        }
        entry = &self.entries.items[preserved_index.*.?];
        try entry.ensureStateCapacity(device, prefix_ids.len);
        self.reserved_tokens = self.reserved_tokens - old_capacity + entry.cached_state.max_seq;
        try entry.tokens.resize(gpa, prefix_ids.len);
        @memcpy(entry.tokens.items, prefix_ids);
        entry.pinned = entry.pinned or pinned;
        self.resident_tokens = self.resident_tokens - old_len + prefix_ids.len;
    }

    fn findExact(self: *const Cache, prefix_ids: []const u32) ?usize {
        return self.index.findExact(prefix_ids);
    }

    fn ensureReservedTokenBudget(
        self: *Cache,
        gpa: std.mem.Allocator,
        preserve_index: *?usize,
        additional_reserved_tokens: usize,
    ) !void {
        if (additional_reserved_tokens > self.max_tokens) return error.SequenceTooLong;
        while (self.reserved_tokens + additional_reserved_tokens > self.max_tokens) {
            const victim = self.evictableIndex(preserve_index.*) orelse return error.SequenceTooLong;
            self.removeEntry(gpa, victim);
            if (preserve_index.*) |preserved| {
                if (victim < preserved) preserve_index.* = preserved - 1;
            }
            self.evictions += 1;
        }
    }

    fn entryStateCapacity(self: *const Cache, token_count: usize) usize {
        return @max(token_count, self.initial_state_tokens);
    }

    fn evictableIndex(self: *const Cache, preserve_index: ?usize) ?usize {
        var victim: ?usize = null;
        for (self.entries.items, 0..) |entry, i| {
            if (preserve_index != null and i == preserve_index.?) continue;
            if (entry.pinned) continue;
            if (victim == null or entry.last_access < self.entries.items[victim.?].last_access) {
                victim = i;
            }
        }
        return victim;
    }

    fn nextAccess(self: *Cache) u64 {
        self.access_clock +%= 1;
        return self.access_clock;
    }
};

fn testDevice() !metal.Device {
    return metal.Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return error.SkipZigTest,
        else => return err,
    };
}

fn addTestEntry(cache: *Cache, device: *metal.Device, tokens: []const u32, pinned: bool, next_token_id: u32) !Match {
    const match = try cache.startDirectEntry(std.testing.allocator, device, tokens, pinned);
    var committed = false;
    errdefer if (!committed) cache.removeEntry(std.testing.allocator, match.index);
    var next_token = try device.createSharedBuffer(@sizeOf(u32));
    defer next_token.destroy();
    next_token.slice(u32)[0] = next_token_id;
    try cache.finishDirectEntry(match, next_token);
    committed = true;
    return match;
}

test "resident prefix cache keeps multiple exact-prefix entries" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(12, 4);
    defer cache.deinit(std.testing.allocator);

    _ = try addTestEntry(&cache, &device, &.{ 1, 2, 3 }, false, 10);
    _ = try addTestEntry(&cache, &device, &.{ 1, 2, 3, 4 }, false, 11);
    _ = try addTestEntry(&cache, &device, &.{ 9, 8 }, false, 12);

    try std.testing.expectEqual(@as(usize, 3), cache.entryCount());
    try std.testing.expectEqual(@as(usize, 9), cache.resident_tokens);
    try std.testing.expectEqual(@as(usize, 12), cache.reserved_tokens);

    const longest = cache.findPrefix(&.{ 1, 2, 3, 4, 5 }).?;
    try std.testing.expectEqual(@as(usize, 4), longest.len);
    try std.testing.expectEqual(@as(u32, 11), cache.cachedNextToken(longest));
    try std.testing.expect(cache.findPrefix(&.{ 1, 2, 9 }) == null);
    try std.testing.expect(cache.containsPromptPrefix(&.{ 1, 2, 3, 4, 5 }, 3));
}

test "resident prefix cache evicts least recently used unpinned entries" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(8, 4);
    defer cache.deinit(std.testing.allocator);

    const first = try addTestEntry(&cache, &device, &.{ 1, 2, 3 }, false, 10);
    _ = try addTestEntry(&cache, &device, &.{ 4, 5, 6 }, false, 11);
    cache.touch(first);

    _ = try addTestEntry(&cache, &device, &.{ 7, 8, 9 }, false, 12);

    try std.testing.expectEqual(@as(usize, 2), cache.entryCount());
    try std.testing.expectEqual(@as(usize, 6), cache.resident_tokens);
    try std.testing.expectEqual(@as(usize, 8), cache.reserved_tokens);
    try std.testing.expectEqual(@as(usize, 1), cache.evictions);
    try std.testing.expect(cache.findPrefix(&.{ 1, 2, 3, 0 }) != null);
    try std.testing.expect(cache.findPrefix(&.{ 4, 5, 6, 0 }) == null);
    try std.testing.expect(cache.findPrefix(&.{ 7, 8, 9, 0 }) != null);
}

test "resident prefix index preserves descendant entries after branch removal" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(8, 4);
    defer cache.deinit(std.testing.allocator);

    const shorter = try addTestEntry(&cache, &device, &.{ 1, 2, 3 }, false, 10);
    _ = try addTestEntry(&cache, &device, &.{ 1, 2, 3, 4 }, false, 11);

    cache.removeEntry(std.testing.allocator, shorter.index);

    try std.testing.expectEqual(@as(usize, 1), cache.entryCount());
    try std.testing.expectEqual(@as(usize, 4), cache.resident_tokens);
    try std.testing.expect(cache.findPrefix(&.{ 1, 2, 3, 0 }) == null);
    try std.testing.expect(!cache.containsPromptPrefix(&.{ 1, 2, 3, 4, 5 }, 3));
    try std.testing.expect(cache.containsPromptPrefix(&.{ 1, 2, 3, 4, 5 }, 4));

    const longest = cache.findPrefix(&.{ 1, 2, 3, 4, 5 }).?;
    try std.testing.expectEqual(@as(usize, 4), longest.len);
    try std.testing.expectEqual(@as(u32, 11), cache.cachedNextToken(longest));
}

test "resident prefix cache returns shifted match after extending and evicting earlier entry" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(5, 2);
    defer cache.deinit(std.testing.allocator);

    _ = try addTestEntry(&cache, &device, &.{ 9, 9 }, false, 10);
    const reuse = try addTestEntry(&cache, &device, &.{ 1, 2, 3 }, false, 11);
    try std.testing.expectEqual(@as(usize, 1), reuse.index);

    var work_state = try ModelState.initBf16FullCaches(&device, 2);
    defer work_state.deinit();
    var next_token = try device.createSharedBuffer(@sizeOf(u32));
    defer next_token.destroy();
    next_token.slice(u32)[0] = 12;

    const stored = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &.{ 1, 2, 3, 4, 5 },
        reuse,
        &work_state,
        null,
        0,
        next_token,
        false,
    );

    try std.testing.expectEqual(@as(usize, 0), stored.index);
    try std.testing.expectEqual(@as(usize, 5), stored.len);
    try std.testing.expectEqual(@as(usize, 1), cache.entryCount());
    try std.testing.expectEqual(@as(usize, 1), cache.evictions);
    try std.testing.expect(cache.findPrefix(&.{ 9, 9, 0 }) == null);

    const matched = cache.findPrefix(&.{ 1, 2, 3, 4, 5, 6 }).?;
    try std.testing.expectEqual(stored, matched);
    const decode_prefix = cache.prefixState(.{ .index = stored.index, .len = reuse.len }).?;
    try std.testing.expectEqual(@as(u32, @intCast(reuse.len)), decode_prefix.len);
}

test "resident prefix cache preserves pinned entries across eviction" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(8, 4);
    defer cache.deinit(std.testing.allocator);

    _ = try addTestEntry(&cache, &device, &.{ 1, 2, 3 }, true, 10);
    _ = try addTestEntry(&cache, &device, &.{ 4, 5, 6 }, false, 11);
    _ = try addTestEntry(&cache, &device, &.{ 7, 8, 9 }, false, 12);

    try std.testing.expectEqual(@as(usize, 2), cache.entryCount());
    try std.testing.expectEqual(@as(usize, 6), cache.resident_tokens);
    try std.testing.expectEqual(@as(usize, 8), cache.reserved_tokens);
    try std.testing.expectEqual(@as(usize, 1), cache.pinnedEntryCount());
    try std.testing.expect(cache.findPrefix(&.{ 1, 2, 3, 0 }) != null);
    try std.testing.expect(cache.findPrefix(&.{ 4, 5, 6, 0 }) == null);
    try std.testing.expect(cache.findPrefix(&.{ 7, 8, 9, 0 }) != null);
}

test "storePrefixFromWork marks hidden_valid true for fresh entry with real hiddens" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(8, 4);
    defer cache.deinit(std.testing.allocator);
    cache.enableHiddenTracking();

    var work_state = try ModelState.initBf16FullCaches(&device, 4);
    defer work_state.deinit();

    const tokens = [_]u32{ 1, 2, 3 };
    var work_hidden = try device.createSharedBuffer(try hiddenRowsBytes(tokens.len));
    defer work_hidden.destroy();
    @memset(work_hidden.slice(u8), 0xAB);

    const match = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &tokens,
        null,
        &work_state,
        work_hidden,
        0,
        null,
        false,
    );
    try std.testing.expect(cache.entryHiddenValid(match));
    const hidden = cache.entryHidden(match).?;
    for (hidden.slice(u8)[0..(try hiddenRowsBytes(tokens.len))]) |b| try std.testing.expectEqual(@as(u8, 0xAB), b);
}

test "storePrefixFromWork preserves reused hidden rows and extends validity" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(16, 4);
    defer cache.deinit(std.testing.allocator);
    cache.enableHiddenTracking();

    var work_state = try ModelState.initBf16FullCaches(&device, 8);
    defer work_state.deinit();

    // Seed a 3-row prefix with valid hiddens.
    const prefix_a = [_]u32{ 1, 2, 3 };
    var hidden_a = try device.createSharedBuffer(try hiddenRowsBytes(prefix_a.len));
    defer hidden_a.destroy();
    @memset(hidden_a.slice(u8), 0x11);
    const match_a = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_a,
        null,
        &work_state,
        hidden_a,
        0,
        null,
        false,
    );
    try std.testing.expect(cache.entryHiddenValid(match_a));

    // Extend to a 5-row prefix reusing the 3-row entry; suffix rows come from
    // a real work_hidden buffer (rows 3..5 at base offset 3).
    const prefix_b = [_]u32{ 1, 2, 3, 4, 5 };
    var hidden_b = try device.createSharedBuffer(try hiddenRowsBytes(prefix_b.len));
    defer hidden_b.destroy();
    @memset(hidden_b.slice(u8), 0);
    // Suffix rows 3..5 in the full-prompt buffer.
    @memset(hidden_b.slice(u8)[try hiddenRowsBytes(3)..try hiddenRowsBytes(5)], 0x22);
    const match_b = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_b,
        match_a,
        &work_state,
        hidden_b,
        3,
        null,
        false,
    );
    try std.testing.expect(cache.entryHiddenValid(match_b));
    const hidden = cache.entryHidden(match_b).?;
    // Reused rows 0..3 keep 0x11; suffix rows 3..5 are 0x22.
    const row_bytes = try hiddenRowsBytes(1);
    try std.testing.expectEqual(@as(u8, 0x11), hidden.slice(u8)[0]);
    try std.testing.expectEqual(@as(u8, 0x11), hidden.slice(u8)[row_bytes * 2]);
    try std.testing.expectEqual(@as(u8, 0x22), hidden.slice(u8)[row_bytes * 3]);
    try std.testing.expectEqual(@as(u8, 0x22), hidden.slice(u8)[row_bytes * 4]);
}

test "storePrefixFromWork clears hidden_valid when suffix hiddens are absent" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(16, 4);
    defer cache.deinit(std.testing.allocator);
    cache.enableHiddenTracking();

    var work_state = try ModelState.initBf16FullCaches(&device, 8);
    defer work_state.deinit();

    const prefix_a = [_]u32{ 1, 2, 3 };
    var hidden_a = try device.createSharedBuffer(try hiddenRowsBytes(prefix_a.len));
    defer hidden_a.destroy();
    @memset(hidden_a.slice(u8), 0x11);
    const match_a = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_a,
        null,
        &work_state,
        hidden_a,
        0,
        null,
        false,
    );
    try std.testing.expect(cache.entryHiddenValid(match_a));

    // Extend with no work_hidden: suffix rows are zero-filled and the entry
    // must not claim validity (so MTP falls back instead of using zeros).
    const prefix_b = [_]u32{ 1, 2, 3, 4, 5 };
    const match_b = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_b,
        match_a,
        &work_state,
        null,
        0,
        null,
        false,
    );
    try std.testing.expect(!cache.entryHiddenValid(match_b));
}

test "storePrefixFromWork reads reuse validity from the shifted entry, not stale reuse_match.index" {
    // Regression: ensureReservedTokenBudget can evict a lower-index entry,
    // shifting the reused entry down. The caller's reuse_match.index is the
    // pre-shift value, so reading entries[reuse_match.index].hidden_valid can
    // hit a different entry (or run off the end). The fix reads
    // entry.hidden_valid (match.index, adjusted) instead.
    var device = try testDevice();
    defer device.destroy();

    // max_tokens=8, initial_state_tokens=4 -> each entry reserves 4; two
    // entries fill the budget, a capacity-growing reuse-store must evict.
    var cache = try Cache.init(8, 4);
    defer cache.deinit(std.testing.allocator);
    cache.enableHiddenTracking();

    var work_state = try ModelState.initBf16FullCaches(&device, 8);
    defer work_state.deinit();

    // Entry A at index 0 (3 tokens, reserves 4).
    const prefix_a = [_]u32{ 1, 2, 3 };
    var hidden_a = try device.createSharedBuffer(try hiddenRowsBytes(prefix_a.len));
    defer hidden_a.destroy();
    @memset(hidden_a.slice(u8), 0x11);
    const match_a = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_a,
        null,
        &work_state,
        hidden_a,
        0,
        null,
        false,
    );
    try std.testing.expectEqual(@as(usize, 0), match_a.index);
    try std.testing.expect(cache.entryHiddenValid(match_a));

    // Entry B at index 1 (3 tokens, reserves 4) with valid hiddens.
    const prefix_b = [_]u32{ 4, 5, 6 };
    var hidden_b = try device.createSharedBuffer(try hiddenRowsBytes(prefix_b.len));
    defer hidden_b.destroy();
    @memset(hidden_b.slice(u8), 0x22);
    const match_b = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_b,
        null,
        &work_state,
        hidden_b,
        0,
        null,
        false,
    );
    try std.testing.expectEqual(@as(usize, 1), match_b.index);
    try std.testing.expect(cache.entryHiddenValid(match_b));
    try std.testing.expectEqual(@as(usize, 8), cache.reserved_tokens);

    // Reuse B (match_b.index = 1) and grow B from 3 to 5 tokens. This needs +1
    // reserved token (capacity 4 -> 5), forcing eviction of A (index 0). B
    // shifts from index 1 to 0; match_b.index (1) is now stale.
    const prefix_c = [_]u32{ 4, 5, 6, 7, 8 };
    var hidden_c = try device.createSharedBuffer(try hiddenRowsBytes(prefix_c.len));
    defer hidden_c.destroy();
    // Suffix rows 3..5 carry real data so the result is valid.
    @memset(hidden_c.slice(u8), 0);
    @memset(hidden_c.slice(u8)[try hiddenRowsBytes(3)..try hiddenRowsBytes(5)], 0x33);
    const match_c = try cache.storePrefixFromWork(
        std.testing.allocator,
        &device,
        &prefix_c,
        match_b,
        &work_state,
        hidden_c,
        3,
        null,
        false,
    );

    // B shifted to index 0; A was evicted; only one entry remains.
    try std.testing.expectEqual(@as(usize, 1), cache.entryCount());
    try std.testing.expectEqual(@as(usize, 0), match_c.index);
    // The fix: validity is read from the shifted reuse entry (B), whose
    // hiddens were valid, and the suffix is real, so the result is valid.
    // The old code read entries[1] (out of range -> guard -> false) and would
    // have reported the entry invalid.
    try std.testing.expect(cache.entryHiddenValid(match_c));
    // And the reused prefix rows (0..3) still carry B's 0x22 pattern.
    const hidden = cache.entryHidden(match_c).?;
    const row_bytes = try hiddenRowsBytes(1);
    try std.testing.expectEqual(@as(u8, 0x22), hidden.slice(u8)[0]);
    try std.testing.expectEqual(@as(u8, 0x22), hidden.slice(u8)[row_bytes * 2]);
    // Suffix rows 3..5 are 0x33.
    try std.testing.expectEqual(@as(u8, 0x33), hidden.slice(u8)[row_bytes * 3]);
    try std.testing.expectEqual(@as(u8, 0x33), hidden.slice(u8)[row_bytes * 4]);
}

test "stats tracks token-level hit ratio across hits, misses, and computed tokens" {
    var device = try testDevice();
    defer device.destroy();

    var cache = try Cache.init(4096, 1600);
    defer cache.deinit(std.testing.allocator);

    // Miss: no entries yet, nothing reused. Caller records computed_tokens.
    cache.recordMiss();
    cache.recordComputedTokens(1000);
    // Add an entry (1000 tokens) so subsequent requests can hit.
    var work_state = try ModelState.initBf16FullCaches(&device, 1600);
    defer work_state.deinit();
    const prefix_a = [_]u32{1} ** 1000;
    const match_a = try cache.startDirectEntry(std.testing.allocator, &device, &prefix_a, false);
    try cache.finishLoadedEntry(std.testing.allocator, match_a);

    // Partial hit: reuses 800 of a 1200-token prompt; 400 computed.
    cache.recordHit(800);
    cache.recordComputedTokens(400);

    // Full hit: reuses all 1000 of a 1000-token prompt; 0 computed.
    cache.recordHit(1000);
    cache.recordComputedTokens(0);

    const s = cache.stats();
    try std.testing.expectEqual(@as(usize, 2), s.hits);
    try std.testing.expectEqual(@as(usize, 1), s.misses);
    try std.testing.expectEqual(@as(usize, 1800), s.hit_tokens); // 800 + 1000
    try std.testing.expectEqual(@as(usize, 1400), s.computed_tokens); // 1000 + 400 + 0
    // Token hit ratio = 1800 / (1800 + 1400) = 0.5625
    const served = s.hit_tokens + s.computed_tokens;
    const ratio: f64 = @as(f64, @floatFromInt(s.hit_tokens)) / @as(f64, @floatFromInt(served));
    const expected: f64 = 0.5625;
    try std.testing.expect(@abs(ratio - expected) < 1e-9);
}

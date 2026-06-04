//! Minimal safetensors reader for the MLX-quantized Qwen3.5-9B checkpoint.
//! Each shard is memory-mapped (zero-copy), its header parsed into a tensor
//! table (name -> shape/byte range), and tensor bytes are copied directly from
//! the backing files into caller-provided buffers.

const std = @import("std");
const posix = std.posix;

pub const TensorInfo = struct {
    shape: []const u64,
    shard: usize,
    /// Byte range within the shard's data blob (after the 8-byte length + header).
    begin: u64,
    end: u64,
};

pub const Repository = struct {
    arena: std.heap.ArenaAllocator,
    io: std.Io, // must outlive deinit() — used to pread (readInto) and close the shards
    shard_maps: [][]u8,
    shard_files: []std.Io.File,
    shard_data_start: []u64,
    tensors: std.StringHashMapUnmanaged(TensorInfo),

    /// Open a checkpoint directory: discover shards from the safetensors index,
    /// mmap each, and build the tensor table from their headers.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8) !Repository {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
        defer dir.close(io);

        const shard_names = try discoverShards(gpa, a, io, dir);

        const shard_maps = try a.alloc([]u8, shard_names.len);
        const shard_files = try a.alloc(std.Io.File, shard_names.len);
        const shard_data_start = try a.alloc(u64, shard_names.len);
        var tensors: std.StringHashMapUnmanaged(TensorInfo) = .empty;
        var mapped: usize = 0;
        errdefer for (shard_maps[0..mapped]) |m| posix.munmap(@alignCast(m));
        errdefer for (shard_files[0..mapped]) |f| f.close(io);

        for (shard_names, 0..) |name, shard| {
            // Keep the file open so weight upload can pread straight into GPU
            // buffers via readInto. That avoids faulting the mmap pages and
            // doubling peak RSS with file pages plus GPU copies.
            const file = try dir.openFile(io, name, .{});
            const size = (file.stat(io) catch |e| {
                file.close(io);
                return e;
            }).size;
            const map = posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0) catch |e| {
                file.close(io);
                return e;
            };
            shard_maps[shard] = map;
            shard_files[shard] = file;
            mapped += 1; // both map + file now owned; outer errdefers clean them

            if (size < 8) return error.TruncatedHeader;
            const header_len = std.mem.readInt(u64, map.ptr[0..8], .little);
            if (header_len > size - 8) return error.TruncatedHeader; // subtract on the trusted side (no overflow)
            shard_data_start[shard] = 8 + header_len;
            try parseHeader(a, map[8 .. 8 + header_len], shard, &tensors);
        }

        return .{
            .arena = arena,
            .io = io,
            .shard_maps = shard_maps,
            .shard_files = shard_files,
            .shard_data_start = shard_data_start,
            .tensors = tensors,
        };
    }

    pub fn deinit(self: *Repository) void {
        for (self.shard_maps) |m| posix.munmap(@alignCast(m));
        for (self.shard_files) |f| f.close(self.io);
        self.arena.deinit();
        self.* = undefined;
    }

    /// Read a tensor's raw bytes straight from the file into `dest` (which must be
    /// exactly the tensor's byte length). This keeps weight upload from faulting
    /// the mmap pages into process RSS before copying to GPU buffers.
    pub fn readInto(self: *const Repository, info: TensorInfo, dest: []u8) !void {
        const range = try self.tensorRange(info);
        if (dest.len != range.len) return error.InvalidTensorBuffer;
        const base: u64 = @intCast(range.start);
        var done: usize = 0;
        while (done < range.len) {
            const n = try self.shard_files[info.shard].readPositionalAll(self.io, dest[done..range.len], base + @as(u64, @intCast(done)));
            if (n == 0) return error.UnexpectedEof;
            done += n;
        }
    }

    pub fn get(self: *const Repository, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    pub fn tensorByteLen(self: *const Repository, info: TensorInfo) !usize {
        return (try self.tensorRange(info)).len;
    }

    const TensorRange = struct {
        start: usize,
        end: usize,
        len: usize,
    };

    fn tensorRange(self: *const Repository, info: TensorInfo) !TensorRange {
        if (info.shard >= self.shard_maps.len or info.shard >= self.shard_data_start.len or info.shard >= self.shard_files.len) {
            return error.MalformedHeader;
        }
        if (info.begin > info.end) return error.MalformedHeader;
        const map = self.shard_maps[info.shard];
        const data_start = std.math.cast(usize, self.shard_data_start[info.shard]) orelse return error.MalformedHeader;
        const begin = std.math.cast(usize, info.begin) orelse return error.MalformedHeader;
        const end = std.math.cast(usize, info.end) orelse return error.MalformedHeader;
        if (data_start > map.len) return error.MalformedHeader;
        if (end > map.len - data_start) return error.MalformedHeader;
        return .{
            .start = data_start + begin,
            .end = data_start + end,
            .len = end - begin,
        };
    }
};

/// Read the safetensors index and return the unique shard file names, sorted.
fn discoverShards(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![][]const u8 {
    const index_bytes = try dir.readFileAlloc(io, "model.safetensors.index.json", gpa, .limited(64 * 1024 * 1024));
    defer gpa.free(index_bytes);

    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, index_bytes, .{});
    defer parsed.deinit();
    const weight_map = switch (parsed.value) {
        .object => |root| switch (root.get("weight_map") orelse return error.MalformedIndex) {
            .object => |wm| wm,
            else => return error.MalformedIndex,
        },
        else => return error.MalformedIndex,
    };

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    var names: std.ArrayList([]const u8) = .empty;
    var it = weight_map.iterator();
    while (it.next()) |kv| {
        const shard = switch (kv.value_ptr.*) {
            .string => |s| s,
            else => return error.MalformedIndex,
        };
        if ((try seen.getOrPut(gpa, shard)).found_existing) continue;
        try names.append(arena, try arena.dupe(u8, shard));
    }
    if (names.items.len == 0) return error.NoShardsFound;
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return names.items;
}

/// Extract a non-negative JSON integer as u64, erroring (not panicking) on a
/// wrong type or negative value from a malformed header.
fn jsonU64(v: std.json.Value) !u64 {
    return switch (v) {
        .integer => |n| if (n < 0) error.MalformedHeader else @intCast(n),
        else => error.MalformedHeader,
    };
}

fn parseHeader(
    arena: std.mem.Allocator,
    header_json: []const u8,
    shard: usize,
    tensors: *std.StringHashMapUnmanaged(TensorInfo),
) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, header_json, .{});
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.MalformedHeader,
    };

    var entries = root.iterator();
    while (entries.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const obj = switch (kv.value_ptr.*) {
            .object => |o| o,
            else => return error.MalformedHeader,
        };
        const shape_arr = switch (obj.get("shape") orelse return error.MalformedHeader) {
            .array => |arr| arr,
            else => return error.MalformedHeader,
        };
        const offs_arr = switch (obj.get("data_offsets") orelse return error.MalformedHeader) {
            .array => |arr| arr,
            else => return error.MalformedHeader,
        };
        if (offs_arr.items.len != 2) return error.MalformedHeader;

        const shape = try arena.alloc(u64, shape_arr.items.len);
        for (shape_arr.items, 0..) |dim, i| shape[i] = try jsonU64(dim);

        const name = try arena.dupe(u8, kv.key_ptr.*);
        try tensors.put(arena, name, .{
            .shape = shape,
            .shard = shard,
            .begin = try jsonU64(offs_arr.items[0]),
            .end = try jsonU64(offs_arr.items[1]),
        });
    }
}

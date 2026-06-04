//! Kestrel-derived GEMM tile descriptions used by MLX-style prefill planning.

const std = @import("std");
const dtype_mod = @import("dtype.zig");

pub const TileConfig = struct {
    bm: usize,
    bn: usize,
    bk: usize,
    wm: usize,
    wn: usize,
};

pub const tile_32_32_16_2_2 = TileConfig{ .bm = 32, .bn = 32, .bk = 16, .wm = 2, .wn = 2 };
pub const tile_64_64_16_2_2 = TileConfig{ .bm = 64, .bn = 64, .bk = 16, .wm = 2, .wn = 2 };
pub const tile_64_64_16_1_2 = TileConfig{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
pub const tile_64_32_32_2_2 = TileConfig{ .bm = 64, .bn = 32, .bk = 32, .wm = 2, .wn = 2 };
pub const tile_32_64_16_1_2 = TileConfig{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };

pub fn eqlTile(a: TileConfig, b: TileConfig) bool {
    return a.bm == b.bm and a.bn == b.bn and a.bk == b.bk and a.wm == b.wm and a.wn == b.wn;
}

const KernelTile = struct {
    config: TileConfig,
    suffix: []const u8,
};

const nt_kernel_tiles = [_]KernelTile{
    .{ .config = tile_32_32_16_2_2, .suffix = "32_32_16_2_2" },
    .{ .config = tile_64_64_16_2_2, .suffix = "64_64_16_2_2" },
    .{ .config = tile_64_64_16_1_2, .suffix = "64_64_16_1_2" },
    .{ .config = tile_64_32_32_2_2, .suffix = "64_32_32_2_2" },
    .{ .config = tile_32_64_16_1_2, .suffix = "32_64_16_1_2" },
};

pub fn kernelNameForNtConfig(dtype: dtype_mod.DType, config: TileConfig) ![:0]const u8 {
    if (dtype != .f32 and dtype != .f16 and dtype != .bf16) return error.UnsupportedMatmulDType;

    inline for (nt_kernel_tiles) |tile| {
        if (eqlTile(config, tile.config)) {
            return switch (dtype) {
                .f32 => std.fmt.comptimePrint("gemm_nt_f32_f32_{s}", .{tile.suffix}),
                .f16 => std.fmt.comptimePrint("gemm_nt_f16_f16_{s}", .{tile.suffix}),
                .bf16 => std.fmt.comptimePrint("gemm_nt_bf16_bf16_{s}", .{tile.suffix}),
                else => unreachable,
            };
        }
    }

    return error.UnsupportedMatmulConfig;
}

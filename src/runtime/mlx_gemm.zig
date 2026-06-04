//! MLX-style GEMM launch planning ported from Kestrel's Qwen3.5 q4 prefill path.
//! `Params` must match `GEMMParams` in Kestrel's `shaders/mlx_gemm.metal`.

const std = @import("std");
const dtype_mod = @import("dtype.zig");
const matmul = @import("matmul_config.zig");
const function_constants = @import("function_constants.zig");

pub const Params = extern struct {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
};

pub const AddMMParams = extern struct {
    ldc: i32,
    fdc: i32,
    batch_stride_c: usize,
    alpha: f32,
    beta: f32,
};

pub const Plan = struct {
    kernel_name: [:0]const u8,
    tile: matmul.TileConfig,
    params: Params,
    constants: [7]function_constants.Bool,
    threadgroups_per_grid: struct { x: usize, y: usize, z: usize },
    threads_per_threadgroup: struct { x: usize, y: usize, z: usize },
};

pub fn planDecodeNt(dtype: dtype_mod.DType, m: usize, n: usize, k: usize) !Plan {
    return planDecodeNtForTile(dtype, m, n, k, matmul.tile_32_32_16_2_2);
}

pub fn planPrefillDenseNt(dtype: dtype_mod.DType, m: usize, n: usize, k: usize) !Plan {
    return planDecodeNtForTile(dtype, m, n, k, selectPrefillDenseNtTile(n, k));
}

pub fn planPrefillAttentionScoreNt(dtype: dtype_mod.DType, m: usize, n: usize, k: usize) !Plan {
    return planDecodeNtForTile(dtype, m, n, k, selectPrefillAttentionScoreNtTile(n, k));
}

pub fn planPrefillAttentionValueNt(dtype: dtype_mod.DType, m: usize, n: usize, k: usize) !Plan {
    return planDecodeNtForTile(dtype, m, n, k, selectPrefillAttentionValueNtTile(n, k));
}

pub fn planDecodeNtForTile(dtype: dtype_mod.DType, m: usize, n: usize, k: usize, tile: matmul.TileConfig) !Plan {
    if (m == 0 or n == 0 or k == 0) return error.InvalidMatmulShape;
    assertValidTile(tile);

    return .{
        .kernel_name = try matmul.kernelNameForNtConfig(dtype, tile),
        .tile = tile,
        .params = try makeParams(m, n, k, tile),
        .constants = makeConstants(m, n, k, tile),
        .threadgroups_per_grid = .{
            .x = try std.math.divCeil(usize, n, tile.bn),
            .y = try std.math.divCeil(usize, m, tile.bm),
            .z = 1,
        },
        .threads_per_threadgroup = .{
            .x = 32,
            .y = tile.wn,
            .z = tile.wm,
        },
    };
}

fn selectPrefillDenseNtTile(n: usize, k: usize) matmul.TileConfig {
    if (k == 4096 and (n == 4096 or n == 8192 or n == 12288 or n == 12352)) {
        return matmul.tile_32_64_16_1_2;
    }
    if (n == 4096 and k == 12288) {
        return matmul.tile_64_64_16_2_2;
    }
    return matmul.tile_32_32_16_2_2;
}

fn selectPrefillAttentionScoreNtTile(n: usize, k: usize) matmul.TileConfig {
    if (k == 256 and n >= 8192) {
        return matmul.tile_32_64_16_1_2;
    }
    return matmul.tile_32_32_16_2_2;
}

fn selectPrefillAttentionValueNtTile(n: usize, k: usize) matmul.TileConfig {
    if (n == 256 and k >= 16384) {
        return matmul.tile_64_64_16_2_2;
    }
    if (n == 256 and k >= 8192) {
        return matmul.tile_32_64_16_1_2;
    }
    return matmul.tile_32_32_16_2_2;
}

fn makeParams(m: usize, n: usize, k: usize, tile: matmul.TileConfig) !Params {
    return .{
        .m = try castI32(m),
        .n = try castI32(n),
        .k = try castI32(k),
        .lda = try castI32(k),
        .ldb = try castI32(k),
        .ldd = try castI32(n),
        .tiles_n = try castI32(try std.math.divCeil(usize, n, tile.bn)),
        .tiles_m = try castI32(try std.math.divCeil(usize, m, tile.bm)),
        .batch_stride_a = 0,
        .batch_stride_b = 0,
        .batch_stride_d = @as(isize, @intCast(m * n)),
        .swizzle_log = 0,
        .gemm_k_iterations_aligned = try castI32(k / tile.bk),
        .batch_ndim = 1,
    };
}

fn makeConstants(m: usize, n: usize, k: usize, tile: matmul.TileConfig) [7]function_constants.Bool {
    return .{
        .{ .index = 10, .value = false },
        .{ .index = 100, .value = false },
        .{ .index = 110, .value = false },
        .{ .index = 200, .value = m % tile.bm == 0 },
        .{ .index = 201, .value = n % tile.bn == 0 },
        .{ .index = 202, .value = k % tile.bk == 0 },
        .{ .index = 300, .value = false },
    };
}

fn assertValidTile(tile: matmul.TileConfig) void {
    std.debug.assert(tile.bm != 0);
    std.debug.assert(tile.bn != 0);
    std.debug.assert(tile.bk != 0);
}

fn castI32(value: usize) !i32 {
    if (value > @as(usize, @intCast(std.math.maxInt(i32)))) return error.MatmulShapeTooLarge;
    return @intCast(value);
}

test "planDecodeNt matches Qwen BF16 decode GEMM shape" {
    const plan = try planDecodeNt(.bf16, 1, 4096, 1024);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_32_16_2_2", plan.kernel_name);
    try std.testing.expectEqual(@as(i32, 1), plan.params.m);
    try std.testing.expectEqual(@as(i32, 4096), plan.params.n);
    try std.testing.expectEqual(@as(i32, 1024), plan.params.k);
    try std.testing.expectEqual(@as(i32, 128), plan.params.tiles_n);
    try std.testing.expectEqual(@as(i32, 1), plan.params.tiles_m);
    try std.testing.expectEqual(@as(usize, 128), plan.threads_per_threadgroup.x * plan.threads_per_threadgroup.y * plan.threads_per_threadgroup.z);
    try std.testing.expect(!plan.constants[3].value);
    try std.testing.expect(plan.constants[4].value);
    try std.testing.expect(plan.constants[5].value);
}

test "planDecodeNtForTile resolves BF16 NT tile variants" {
    const plan = try planDecodeNtForTile(.bf16, 8192, 4096, 12288, matmul.tile_64_32_32_2_2);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_64_32_32_2_2", plan.kernel_name);
    try std.testing.expectEqual(@as(usize, 128), plan.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 128), plan.threadgroups_per_grid.y);
    try std.testing.expectEqual(@as(usize, 128), plan.threads_per_threadgroup.x * plan.threads_per_threadgroup.y * plan.threads_per_threadgroup.z);
    try std.testing.expect(plan.constants[3].value);
    try std.testing.expect(plan.constants[4].value);
    try std.testing.expect(plan.constants[5].value);
}

test "planPrefillDenseNt selects Qwen3.5 9B prefill tile variants" {
    const gate_up = try planPrefillDenseNt(.bf16, 8192, 12288, 4096);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_64_16_1_2", gate_up.kernel_name);
    try std.testing.expectEqual(@as(usize, 192), gate_up.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 256), gate_up.threadgroups_per_grid.y);

    const deltanet_frontend = try planPrefillDenseNt(.bf16, 8192, 12352, 4096);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_64_16_1_2", deltanet_frontend.kernel_name);
    try std.testing.expectEqual(@as(usize, 193), deltanet_frontend.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 256), deltanet_frontend.threadgroups_per_grid.y);

    const down = try planPrefillDenseNt(.bf16, 8192, 4096, 12288);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_64_64_16_2_2", down.kernel_name);
    try std.testing.expectEqual(@as(usize, 64), down.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 128), down.threadgroups_per_grid.y);

    const fallback = try planPrefillDenseNt(.bf16, 1, 4096, 1024);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_32_16_2_2", fallback.kernel_name);
}

test "planPrefillAttentionNt selects Qwen3.5 long-cache tile variants" {
    const score_8k = try planPrefillAttentionScoreNt(.bf16, 32_768, 8_192, 256);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_64_16_1_2", score_8k.kernel_name);

    const value_8k = try planPrefillAttentionValueNt(.bf16, 32_768, 256, 8_192);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_64_16_1_2", value_8k.kernel_name);

    const value_16k = try planPrefillAttentionValueNt(.bf16, 65_536, 256, 16_384);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_64_64_16_2_2", value_16k.kernel_name);

    const short_score = try planPrefillAttentionScoreNt(.bf16, 8_192, 2_048, 256);
    try std.testing.expectEqualStrings("gemm_nt_bf16_bf16_32_32_16_2_2", short_score.kernel_name);
}

test "planPrefillAttentionValueNt covers Peregrine f32 full-attention value path" {
    const short_value = try planPrefillAttentionValueNt(.f32, 6_400, 256, 1_600);
    try std.testing.expectEqualStrings("gemm_nt_f32_f32_32_32_16_2_2", short_value.kernel_name);
    try std.testing.expectEqual(@as(usize, 8), short_value.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 200), short_value.threadgroups_per_grid.y);

    const value_8k = try planPrefillAttentionValueNt(.f32, 6_400, 256, 9_600);
    try std.testing.expectEqualStrings("gemm_nt_f32_f32_32_64_16_1_2", value_8k.kernel_name);
    try std.testing.expectEqual(@as(usize, 4), value_8k.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 200), value_8k.threadgroups_per_grid.y);

    const value_16k = try planPrefillAttentionValueNt(.f32, 6_400, 256, 16_384);
    try std.testing.expectEqualStrings("gemm_nt_f32_f32_64_64_16_2_2", value_16k.kernel_name);
    try std.testing.expectEqual(@as(usize, 4), value_16k.threadgroups_per_grid.x);
    try std.testing.expectEqual(@as(usize, 100), value_16k.threadgroups_per_grid.y);
}

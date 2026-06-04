//! Shared MLX NT GEMM launch helpers for prepared BF16 dense RHS paths.

const metal = @import("../runtime/metal.zig");
const matmul = @import("../runtime/matmul_config.zig");
const checked_math = @import("../runtime/checked_math.zig");
const linear_q4 = @import("linear_q4.zig");
const Q4Linear = linear_q4.Q4Linear;

const kernel_32_32 = "gemm_nt_bf16_bf16_32_32_16_2_2";
const kernel_32_64 = "gemm_nt_bf16_bf16_32_64_16_1_2";
const kernel_64_64 = "gemm_nt_bf16_bf16_64_64_16_2_2";

const conservative_constants = [_]metal.BoolFunctionConstant{
    .{ .index = 10, .value = false },
    .{ .index = 100, .value = false },
    .{ .index = 110, .value = false },
    .{ .index = 200, .value = false },
    .{ .index = 201, .value = false },
    .{ .index = 202, .value = false },
    .{ .index = 300, .value = false },
};

const conservative_residual_add_constants = [_]metal.BoolFunctionConstant{
    .{ .index = 10, .value = false },
    .{ .index = 100, .value = true },
    .{ .index = 110, .value = false },
    .{ .index = 200, .value = false },
    .{ .index = 201, .value = false },
    .{ .index = 202, .value = false },
    .{ .index = 300, .value = false },
};

pub const Pipelines = struct {
    nt_32_32: metal.Pipeline,
    nt_32_64: metal.Pipeline,
    nt_64_64: metal.Pipeline,

    pub fn create(
        device: *metal.Device,
        library: metal.Library,
        constants: []const metal.BoolFunctionConstant,
    ) !Pipelines {
        var nt_32_32 = try device.createPipelineWithBoolConstants(library, kernel_32_32, constants);
        errdefer nt_32_32.destroy();
        var nt_32_64 = try device.createPipelineWithBoolConstants(library, kernel_32_64, constants);
        errdefer nt_32_64.destroy();
        const nt_64_64 = try device.createPipelineWithBoolConstants(library, kernel_64_64, constants);

        return .{
            .nt_32_32 = nt_32_32,
            .nt_32_64 = nt_32_64,
            .nt_64_64 = nt_64_64,
        };
    }

    pub fn createConservative(device: *metal.Device, library: metal.Library) !Pipelines {
        return create(device, library, &conservative_constants);
    }

    pub fn destroy(self: *Pipelines) void {
        self.nt_32_32.destroy();
        self.nt_32_64.destroy();
        self.nt_64_64.destroy();
        self.* = undefined;
    }

    pub fn pipeline(self: Pipelines, tile: matmul.TileConfig) !metal.Pipeline {
        if (matmul.eqlTile(tile, matmul.tile_32_32_16_2_2)) return self.nt_32_32;
        if (matmul.eqlTile(tile, matmul.tile_32_64_16_1_2)) return self.nt_32_64;
        if (matmul.eqlTile(tile, matmul.tile_64_64_16_2_2)) return self.nt_64_64;
        return error.UnsupportedMatmulConfig;
    }
};

pub fn createConservativeResidualAddPipeline(device: *metal.Device, library: metal.Library) !metal.Pipeline {
    return device.createPipelineWithBoolConstants(library, kernel_64_64, &conservative_residual_add_constants);
}

pub const EncodeOptions = struct {
    input: metal.Buffer,
    dense_rhs: metal.Buffer,
    output: metal.Buffer,
    token_count: u32,
    out_dim: u32,
    in_dim: u32,
    input_byte_offset: usize = 0,
    output_byte_offset: usize = 0,
};

pub const ResidualAddOptions = struct {
    input: metal.Buffer,
    dense_rhs: metal.Buffer,
    residual: metal.Buffer,
    output: metal.Buffer,
    token_count: u32,
    out_dim: u32,
    in_dim: u32,
};

pub const PreparedQ4ProjectionOptions = struct {
    projection: *const Q4Linear,
    qmm_pipeline: metal.Pipeline,
    input: metal.Buffer,
    prepared_rhs: ?metal.Buffer,
    output: metal.Buffer,
    token_count: u32,
    out_dim: u32,
    in_dim: u32,
};

pub fn encode(
    ws: *metal.Workspace,
    pipelines: Pipelines,
    options: EncodeOptions,
) !void {
    if (options.token_count == 0) return error.EmptyInput;
    const input_bytes = try checked_math.product(.{ options.token_count, options.in_dim, linear_q4.BF16_BYTES });
    const output_bytes = try checked_math.product(.{ options.token_count, options.out_dim, linear_q4.BF16_BYTES });
    const dense_rhs_bytes = try linear_q4.denseRhsBf16ByteLen(options.out_dim, options.in_dim);
    if (options.input_byte_offset > options.input.length or input_bytes > options.input.length - options.input_byte_offset) return error.InputBufferTooSmall;
    if (options.dense_rhs.length < dense_rhs_bytes) return error.InputBufferTooSmall;
    if (options.output_byte_offset > options.output.length or output_bytes > options.output.length - options.output_byte_offset) return error.OutputBufferTooSmall;

    const plan = try metal.MlxGemm.planPrefillDenseNt(.bf16, options.token_count, options.out_dim, options.in_dim);
    const params = try ws.valueBuf(metal.MlxGemm.Params, plan.params);
    try ws.cmd.dispatchThreadgroups3D(
        try pipelines.pipeline(plan.tile),
        &.{
            .{ .index = 0, .buffer = options.input, .offset = options.input_byte_offset },
            .{ .index = 1, .buffer = options.dense_rhs },
            .{ .index = 3, .buffer = options.output, .offset = options.output_byte_offset },
            .{ .index = 4, .buffer = params },
        },
        try metal.Grid3D.init(plan.threadgroups_per_grid.x, plan.threadgroups_per_grid.y, plan.threadgroups_per_grid.z),
        try metal.Grid3D.init(plan.threads_per_threadgroup.x, plan.threads_per_threadgroup.y, plan.threads_per_threadgroup.z),
    );
}

pub fn encodePreparedQ4ProjectionBf16(
    ws: *metal.Workspace,
    pipelines: Pipelines,
    options: PreparedQ4ProjectionOptions,
) !void {
    if (options.prepared_rhs) |rhs| {
        try encode(ws, pipelines, .{
            .input = options.input,
            .dense_rhs = rhs,
            .output = options.output,
            .token_count = options.token_count,
            .out_dim = options.out_dim,
            .in_dim = options.in_dim,
        });
    } else {
        try options.projection.encodePrefillQmmM32N64NtBf16(
            ws,
            options.qmm_pipeline,
            options.input,
            options.output,
            options.token_count,
        );
    }
}

pub fn encodeResidualAdd(
    ws: *metal.Workspace,
    add_pipeline: metal.Pipeline,
    options: ResidualAddOptions,
) !void {
    if (options.token_count == 0) return error.EmptyInput;
    const input_bytes = try checked_math.product(.{ options.token_count, options.in_dim, linear_q4.BF16_BYTES });
    const output_bytes = try checked_math.product(.{ options.token_count, options.out_dim, linear_q4.BF16_BYTES });
    const dense_rhs_bytes = try linear_q4.denseRhsBf16ByteLen(options.out_dim, options.in_dim);
    if (options.input.length < input_bytes or options.dense_rhs.length < dense_rhs_bytes or options.residual.length < output_bytes) return error.InputBufferTooSmall;
    if (options.output.length < output_bytes) return error.OutputBufferTooSmall;

    const plan = try metal.MlxGemm.planPrefillDenseNt(.bf16, options.token_count, options.out_dim, options.in_dim);
    if (!matmul.eqlTile(plan.tile, matmul.tile_64_64_16_2_2)) return error.UnsupportedMatmulConfig;

    const params = try ws.valueBuf(metal.MlxGemm.Params, plan.params);
    const add_params = try ws.valueBuf(metal.MlxGemm.AddMMParams, .{
        .ldc = @intCast(options.out_dim),
        .fdc = 1,
        .batch_stride_c = 0,
        .alpha = 1.0,
        .beta = 1.0,
    });
    try ws.cmd.dispatchThreadgroups3D(
        add_pipeline,
        &.{
            .{ .index = 0, .buffer = options.input },
            .{ .index = 1, .buffer = options.dense_rhs },
            .{ .index = 2, .buffer = options.residual },
            .{ .index = 3, .buffer = options.output },
            .{ .index = 4, .buffer = params },
            .{ .index = 5, .buffer = add_params },
        },
        try metal.Grid3D.init(plan.threadgroups_per_grid.x, plan.threadgroups_per_grid.y, plan.threadgroups_per_grid.z),
        try metal.Grid3D.init(plan.threads_per_threadgroup.x, plan.threads_per_threadgroup.y, plan.threads_per_threadgroup.z),
    );
}

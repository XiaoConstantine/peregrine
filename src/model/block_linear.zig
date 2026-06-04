//! Linear-attention (Gated-DeltaNet) block, decode step. Composes the verified
//! kernels via Q4Linear: in_proj_{qkv,z,a,b}, causal conv1d+SiLU, q/k weightless
//! scaled RMSNorm, gate coefficients, the delta-rule recurrence, gated RMSNorm,
//! out_proj. Composition locked end-to-end vs MLX's GatedDeltaNet (rel ~2e-4).
//! Runs one token; the caller owns the conv state and recurrent state.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const dims = @import("dims.zig");
const safetensors = @import("safetensors.zig");
const linear_q4 = @import("linear_q4.zig");
const mlx_nt = @import("mlx_nt.zig");
const weight_upload = @import("upload.zig");
const Q4Linear = linear_q4.Q4Linear;
const BF16_BYTES = dims.bf16_bytes;

const HIDDEN = dims.hidden;
pub const HK = 16; // linear_num_key_heads
pub const HV = 32; // linear_num_value_heads
pub const DK = 128; // linear_key_head_dim
pub const DV = 128; // linear_value_head_dim
pub const KEY_DIM = HK * DK; // 2048
pub const VAL_DIM = HV * DV; // 4096
pub const CONV_DIM = KEY_DIM * 2 + VAL_DIM; // 8192
pub const CONV_K = 4; // linear_conv_kernel_dim
const EPS = dims.rmsnorm_eps;
const INV: f32 = 0.08838834764831845; // head_k_dim^-0.5 = 128^-0.5
const Q_SCALE: f32 = INV * INV; // 1/128
const K_SCALE: f32 = INV;

pub const Bf16Pipelines = struct {
    qmv: metal.Pipeline,
    qmm_m32n64_nt: metal.Pipeline,
    mlx_nt: mlx_nt.Pipelines,
    conv1d_decode: metal.Pipeline,
    conv1d_prefill_vec4: metal.Pipeline,
    conv1d_prefill_state: metal.Pipeline,
    qk_norm_prefill: metal.Pipeline,
    gating_many: metal.Pipeline,
    gated_delta_prefill_norm_prepared_value_major: metal.Pipeline,
    gated_delta_decode_fused: metal.Pipeline,
    rmsnorm_gated: metal.Pipeline,
    split_qkv: metal.Pipeline,
};

pub const PreparedLinearQkvZDenseRhs = struct {
    qkv: metal.Buffer,
    z: metal.Buffer,
};

pub const PreparedLinearABDenseRhs = struct {
    a: metal.Buffer,
    b: metal.Buffer,
};

pub const PreparedLinearDenseRhs = struct {
    qkv_z: ?PreparedLinearQkvZDenseRhs = null,
    a_b: ?PreparedLinearABDenseRhs = null,
    out_proj: ?metal.Buffer = null,
};

pub const PreparedLinearDenseRhsBuffers = struct {
    pub const InitOptions = struct {
        include_qkv_z: bool = false,
        include_a_b: bool = false,
        include_out_proj: bool = false,
    };

    qkv: ?metal.Buffer = null,
    z: ?metal.Buffer = null,
    a: ?metal.Buffer = null,
    b: ?metal.Buffer = null,
    out_proj: ?metal.Buffer = null,

    pub fn init(device: *metal.Device, options: InitOptions) !PreparedLinearDenseRhsBuffers {
        var result: PreparedLinearDenseRhsBuffers = .{};
        errdefer result.deinit();
        if (options.include_qkv_z) {
            result.qkv = try device.createPrivateBuffer(try preparedLinearQkvDenseRhsBf16ByteLen());
            result.z = try device.createPrivateBuffer(try preparedLinearZDenseRhsBf16ByteLen());
        }
        if (options.include_a_b) {
            const gate_bytes = try preparedLinearGateDenseRhsBf16ByteLen();
            result.a = try device.createPrivateBuffer(gate_bytes);
            result.b = try device.createPrivateBuffer(gate_bytes);
        }
        if (options.include_out_proj) {
            result.out_proj = try device.createPrivateBuffer(try preparedLinearOutProjDenseRhsBf16ByteLen());
        }
        return result;
    }

    pub fn deinit(self: *PreparedLinearDenseRhsBuffers) void {
        if (self.out_proj) |*out_proj| out_proj.destroy();
        if (self.b) |*b| b.destroy();
        if (self.a) |*a| a.destroy();
        if (self.z) |*z| z.destroy();
        if (self.qkv) |*qkv| qkv.destroy();
        self.* = undefined;
    }

    pub fn prepared(self: *const PreparedLinearDenseRhsBuffers) PreparedLinearDenseRhs {
        std.debug.assert((self.qkv == null) == (self.z == null));
        std.debug.assert((self.a == null) == (self.b == null));
        return .{
            .qkv_z = if (self.qkv) |qkv| .{ .qkv = qkv, .z = self.z.? } else null,
            .a_b = if (self.a) |a| .{ .a = a, .b = self.b.? } else null,
            .out_proj = self.out_proj,
        };
    }

    pub fn encodeBf16Command(
        self: *const PreparedLinearDenseRhsBuffers,
        cmd: *metal.CommandBuffer,
        dequant_bf16: metal.Pipeline,
        block: *const LinearAttentionBlock,
    ) !void {
        if (self.qkv) |qkv| try block.in_qkv.encodeDequantizeBf16Command(cmd, dequant_bf16, qkv);
        if (self.z) |z| try block.in_z.encodeDequantizeBf16Command(cmd, dequant_bf16, z);
        if (self.a) |a| try block.in_a.encodeDequantizeBf16Command(cmd, dequant_bf16, a);
        if (self.b) |b| try block.in_b.encodeDequantizeBf16Command(cmd, dequant_bf16, b);
        if (self.out_proj) |out_proj| try block.out_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, out_proj);
    }
};

fn preparedLinearQkvDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(CONV_DIM, HIDDEN);
}

fn preparedLinearZDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(VAL_DIM, HIDDEN);
}

fn preparedLinearGateDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(HV, HIDDEN);
}

fn preparedLinearOutProjDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(HIDDEN, VAL_DIM);
}

pub const LinearAttentionBlock = struct {
    in_qkv: Q4Linear,
    in_z: Q4Linear,
    in_a: Q4Linear,
    in_b: Q4Linear,
    out_proj: Q4Linear,
    conv_weight: metal.Buffer,
    a_log: metal.Buffer,
    dt_bias: metal.Buffer,
    norm_weight: metal.Buffer,

    pub fn upload(device: *metal.Device, queue: *metal.Queue, repo: *const safetensors.Repository, prefix: []const u8) !LinearAttentionBlock {
        var key: [256]u8 = undefined;
        var in_qkv = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.in_proj_qkv", .{prefix}));
        errdefer in_qkv.deinit();
        var in_z = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.in_proj_z", .{prefix}));
        errdefer in_z.deinit();
        var in_a = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.in_proj_a", .{prefix}));
        errdefer in_a.deinit();
        var in_b = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.in_proj_b", .{prefix}));
        errdefer in_b.deinit();
        var out_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.out_proj", .{prefix}));
        errdefer out_proj.deinit();
        var conv_weight = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.conv1d.weight", .{prefix}));
        errdefer conv_weight.destroy();
        var a_log = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.A_log", .{prefix}));
        errdefer a_log.destroy();
        var dt_bias = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.dt_bias", .{prefix}));
        errdefer dt_bias.destroy();
        const norm_weight = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.norm.weight", .{prefix}));
        return .{
            .in_qkv = in_qkv,
            .in_z = in_z,
            .in_a = in_a,
            .in_b = in_b,
            .out_proj = out_proj,
            .conv_weight = conv_weight,
            .a_log = a_log,
            .dt_bias = dt_bias,
            .norm_weight = norm_weight,
        };
    }

    pub fn deinit(self: *LinearAttentionBlock) void {
        self.in_qkv.deinit();
        self.in_z.deinit();
        self.in_a.deinit();
        self.in_b.deinit();
        self.out_proj.deinit();
        self.conv_weight.destroy();
        self.a_log.destroy();
        self.dt_bias.destroy();
        self.norm_weight.destroy();
        self.* = undefined;
    }

    pub fn decodeStepBf16(
        self: *const LinearAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        x_bf16: metal.Buffer,
        conv_state: metal.Buffer,
        recurrent_state: metal.Buffer,
        has_previous_state: bool,
        out_bf16: metal.Buffer,
    ) !void {
        const hidden_bytes = try checked_math.bytes(HIDDEN, BF16_BYTES);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const conv_state_bytes = try checked_math.product(.{ CONV_K - 1, CONV_DIM, @sizeOf(f32) });
        const recurrent_state_bytes = try checked_math.product(.{ HV, DV, DK, @sizeOf(f32) });
        if (conv_state.length < conv_state_bytes or recurrent_state.length < recurrent_state_bytes) {
            return error.InvalidLinearAttentionShape;
        }

        const qkv_bytes = try checked_math.bytes(CONV_DIM, BF16_BYTES);
        const qkv = try ws.scratch(qkv_bytes);
        try self.in_qkv.encodeBf16Fast(ws, p.qmv, x_bf16, qkv, 1);

        const z_bytes = try checked_math.bytes(VAL_DIM, BF16_BYTES);
        const z = try ws.scratch(z_bytes);
        try self.in_z.encodeBf16Fast(ws, p.qmv, x_bf16, z, 1);

        const gate_bytes = try checked_math.bytes(HV, BF16_BYTES);
        const abuf = try ws.scratch(gate_bytes);
        try self.in_a.encodeBf16Fast(ws, p.qmv, x_bf16, abuf, 1);
        const bbuf = try ws.scratch(gate_bytes);
        try self.in_b.encodeBf16Fast(ws, p.qmv, x_bf16, bbuf, 1);
        // qkv/z/a/b all read only x_bf16 and run concurrently; everything
        // below consumes their outputs.
        ws.barrier();

        const conv_out = try ws.scratch(qkv_bytes);
        const conv_dim_buf = try ws.u32buf(CONV_DIM);
        const kernel_buf = try ws.u32buf(CONV_K);
        try ws.cmd.dispatch1D(p.conv1d_decode, &.{ conv_state, qkv, self.conv_weight, conv_out, conv_dim_buf, kernel_buf }, CONV_DIM);
        ws.barrier();

        // Fused decode recurrence: q/k L2 norms and gate coefficients are
        // computed inline from conv_out and a/b, removing two dispatches and
        // one barrier level from the decode chain.
        const y = try ws.scratch(z_bytes);
        const hk_buf = try ws.u32buf(HK);
        const hv_buf = try ws.u32buf(HV);
        const dk_buf = try ws.u32buf(DK);
        const dv_buf = try ws.u32buf(DV);
        const has_state_buf = try ws.u32buf(if (has_previous_state) 1 else 0);
        const rows_per_threadgroup = 32;
        try ws.cmd.dispatch1DWithThreadgroup(
            p.gated_delta_decode_fused,
            &.{ recurrent_state, conv_out, abuf, bbuf, self.a_log, self.dt_bias, y, hk_buf, hv_buf, dk_buf, dv_buf, conv_dim_buf, has_state_buf },
            @as(usize, HV) * (DV / rows_per_threadgroup) * 128,
            128,
        );
        ws.barrier();
        const gated = try ws.scratch(z_bytes);
        try self.encodePrefillGatedNormBf16(ws, p, 1, y, z, gated);
        ws.barrier();
        try self.out_proj.encodeBf16Fast(ws, p.qmv, gated, out_bf16, 1);
    }

    pub fn encodePrefillWithPreparedDenseRhsBf16(
        self: *const LinearAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        x_buf: metal.Buffer,
        conv_state: metal.Buffer,
        recurrent_state: metal.Buffer,
        has_previous_state: bool,
        token_count: u32,
        prepared: PreparedLinearDenseRhs,
        out_buf: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const qkv_values = std.math.mul(usize, @as(usize, token_count), CONV_DIM) catch return error.ContextSizeOverflow;
        const qkv_bytes = std.math.mul(usize, qkv_values, BF16_BYTES) catch return error.ContextSizeOverflow;
        const qkv = try ws.scratch(qkv_bytes);
        const prepared_qkv_rhs = if (prepared.qkv_z) |qkv_z| qkv_z.qkv else null;
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.in_qkv,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_buf,
            .prepared_rhs = prepared_qkv_rhs,
            .output = qkv,
            .token_count = token_count,
            .out_dim = CONV_DIM,
            .in_dim = HIDDEN,
        });

        const z_values = std.math.mul(usize, @as(usize, token_count), VAL_DIM) catch return error.ContextSizeOverflow;
        const z_bytes = std.math.mul(usize, z_values, BF16_BYTES) catch return error.ContextSizeOverflow;
        const z = try ws.scratch(z_bytes);
        const prepared_z_rhs = if (prepared.qkv_z) |qkv_z| qkv_z.z else null;
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.in_z,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_buf,
            .prepared_rhs = prepared_z_rhs,
            .output = z,
            .token_count = token_count,
            .out_dim = VAL_DIM,
            .in_dim = HIDDEN,
        });

        const gate_values = std.math.mul(usize, @as(usize, token_count), HV) catch return error.ContextSizeOverflow;
        const gate_bytes = std.math.mul(usize, gate_values, BF16_BYTES) catch return error.ContextSizeOverflow;
        const abuf = try ws.scratch(gate_bytes);
        const bbuf = try ws.scratch(gate_bytes);
        const prepared_a_rhs = if (prepared.a_b) |a_b| a_b.a else null;
        const prepared_b_rhs = if (prepared.a_b) |a_b| a_b.b else null;
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.in_a,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_buf,
            .prepared_rhs = prepared_a_rhs,
            .output = abuf,
            .token_count = token_count,
            .out_dim = HV,
            .in_dim = HIDDEN,
        });
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.in_b,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_buf,
            .prepared_rhs = prepared_b_rhs,
            .output = bbuf,
            .token_count = token_count,
            .out_dim = HV,
            .in_dim = HIDDEN,
        });

        const conv_out = try ws.scratch(qkv_bytes);
        try self.encodePrefillConvBf16(ws, p, qkv, conv_state, token_count, has_previous_state, conv_out);

        const key_values = std.math.mul(usize, @as(usize, token_count), KEY_DIM) catch return error.ContextSizeOverflow;
        const q = try ws.scratch(key_values * @sizeOf(f32));
        const k = try ws.scratch(key_values * @sizeOf(f32));
        try encodePrefillQkNormsFromConvBf16(ws, p, conv_out, token_count, q, k);

        const decay = try ws.scratch(gate_values * @sizeOf(f32));
        const beta = try ws.scratch(gate_values * @sizeOf(f32));
        try self.encodePrefillGatesBf16(ws, p, abuf, bbuf, token_count, decay, beta);

        const y = try ws.scratch(z_bytes);
        try encodePrefillRecurrentPreparedBf16(ws, p, token_count, q, k, conv_out, decay, beta, recurrent_state, has_previous_state, y);
        const gated = try ws.scratch(z_bytes);
        try self.encodePrefillGatedNormBf16(ws, p, token_count, y, z, gated);
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.out_proj,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = gated,
            .prepared_rhs = prepared.out_proj,
            .output = out_buf,
            .token_count = token_count,
            .out_dim = HIDDEN,
            .in_dim = VAL_DIM,
        });
    }

    fn encodePrefillConvBf16(
        self: *const LinearAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        qkv: metal.Buffer,
        conv_state: metal.Buffer,
        token_count: u32,
        has_previous_state: bool,
        out_buf: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const tokens_buf = try ws.u32buf(token_count);
        const conv_dim_buf = try ws.u32buf(CONV_DIM);
        const kernel_buf = try ws.u32buf(CONV_K);
        const has_state_buf = try ws.u32buf(if (has_previous_state) 1 else 0);
        const token_blocks = std.math.divCeil(usize, @as(usize, token_count), 4) catch return error.ContextSizeOverflow;
        const total_threads = std.math.mul(usize, token_blocks, CONV_DIM) catch return error.ContextSizeOverflow;
        try ws.cmd.dispatch1D(
            p.conv1d_prefill_vec4,
            &.{ qkv, self.conv_weight, conv_state, out_buf, tokens_buf, conv_dim_buf, kernel_buf, has_state_buf },
            total_threads,
        );
        try ws.cmd.dispatch1D(
            p.conv1d_prefill_state,
            &.{ qkv, conv_state, tokens_buf, conv_dim_buf, kernel_buf, has_state_buf },
            CONV_DIM,
        );
    }

    fn encodePrefillQkNormsFromConvBf16(
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        conv_out: metal.Buffer,
        token_count: u32,
        q: metal.Buffer,
        k: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const tokens_buf = try ws.u32buf(token_count);
        const hk_buf = try ws.u32buf(HK);
        const dk_buf = try ws.u32buf(DK);
        const conv_dim_buf = try ws.u32buf(CONV_DIM);
        const rows = std.math.mul(usize, @as(usize, token_count), HK) catch return error.ContextSizeOverflow;
        const threads_per_threadgroup = 128;
        try ws.cmd.dispatch1DWithThreadgroup(
            p.qk_norm_prefill,
            &.{ conv_out, q, k, tokens_buf, hk_buf, dk_buf, conv_dim_buf },
            rows * threads_per_threadgroup,
            threads_per_threadgroup,
        );
    }

    fn encodePrefillGatesBf16(
        self: *const LinearAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        a: metal.Buffer,
        b: metal.Buffer,
        token_count: u32,
        g: metal.Buffer,
        beta: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hv_buf = try ws.u32buf(HV);
        const tokens_buf = try ws.u32buf(token_count);
        const total_threads = std.math.mul(usize, @as(usize, token_count), HV) catch return error.ContextSizeOverflow;
        try ws.cmd.dispatch1D(
            p.gating_many,
            &.{ self.a_log, a, self.dt_bias, b, g, beta, hv_buf, tokens_buf },
            total_threads,
        );
    }

    fn encodePrefillRecurrentPreparedBf16(
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        token_count: u32,
        q: metal.Buffer,
        k: metal.Buffer,
        conv_out: metal.Buffer,
        decay: metal.Buffer,
        beta: metal.Buffer,
        recurrent_state: metal.Buffer,
        has_previous_state: bool,
        y: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const tokens_buf = try ws.u32buf(token_count);
        const hk_buf = try ws.u32buf(HK);
        const hv_buf = try ws.u32buf(HV);
        const dk_buf = try ws.u32buf(DK);
        const dv_buf = try ws.u32buf(DV);
        const conv_dim_buf = try ws.u32buf(CONV_DIM);
        const has_state_buf = try ws.u32buf(if (has_previous_state) 1 else 0);
        // Kernel contract: 4 simdgroups per threadgroup, 8 value rows per
        // simdgroup, so each threadgroup covers 32 dv rows of one value head.
        const rows_per_threadgroup = 32;
        const tiles_per_head = DV / rows_per_threadgroup;
        const threadgroups = HV * tiles_per_head;
        const threads_per_threadgroup = 128;
        try ws.cmd.dispatch1DWithThreadgroup(
            p.gated_delta_prefill_norm_prepared_value_major,
            &.{ recurrent_state, q, k, conv_out, decay, beta, y, tokens_buf, hk_buf, hv_buf, dk_buf, dv_buf, conv_dim_buf, has_state_buf },
            threadgroups * threads_per_threadgroup,
            threads_per_threadgroup,
        );
    }

    fn encodePrefillGatedNormBf16(
        self: *const LinearAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        token_count: u32,
        y: metal.Buffer,
        z: metal.Buffer,
        out_buf: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const d_buf = try ws.u32buf(DV);
        const eps_buf = try ws.f32buf(EPS);
        const rows_buf = try ws.u32buf(std.math.mul(u32, token_count, HV) catch return error.ContextSizeOverflow);
        try ws.cmd.dispatch1D(
            p.rmsnorm_gated,
            &.{ y, self.norm_weight, z, out_buf, d_buf, eps_buf, rows_buf },
            @as(usize, token_count) * HV,
        );
    }
};

test "prepared linear out projection dense RHS byte size matches Qwen projection" {
    try std.testing.expectEqual(@as(usize, 8192 * 4096 * BF16_BYTES), try preparedLinearQkvDenseRhsBf16ByteLen());
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * BF16_BYTES), try preparedLinearOutProjDenseRhsBf16ByteLen());
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * BF16_BYTES), try preparedLinearZDenseRhsBf16ByteLen());
    try std.testing.expectEqual(@as(usize, 32 * 4096 * BF16_BYTES), try preparedLinearGateDenseRhsBf16ByteLen());
}

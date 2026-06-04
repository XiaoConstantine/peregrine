//! SwiGLU MLP block (decode step): down_proj(silu(gate_proj(x)) * up_proj(x)).
//! Composed of three Q4Linears + the silu_mul kernel. Verified against MLX's
//! Qwen3NextMLP (rel ~4e-4).

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const dims = @import("dims.zig");
const safetensors = @import("safetensors.zig");
const linear_q4 = @import("linear_q4.zig");
const mlx_nt = @import("mlx_nt.zig");
const Q4Linear = linear_q4.Q4Linear;

const HIDDEN = dims.hidden;
const INTERMEDIATE = 12288;
pub const decode_mlp_gate_up_silu_bf16_kernel_name = "linear_vec_q4_affine_group64_multi2_silu_gate_bf16";
const PREFILL_MLP_GATE_UP_CHUNK_ROWS: u32 = 8192;

pub const PreparedMlpGateUpDenseRhs = struct {
    gate: metal.Buffer,
    up: metal.Buffer,
};

pub const PreparedMlpDenseRhs = struct {
    gate_up: ?PreparedMlpGateUpDenseRhs = null,
    down: ?metal.Buffer = null,
};

pub const PreparedMlpDenseRhsBuffers = struct {
    pub const InitOptions = struct {
        include_gate_up: bool = false,
        include_down: bool = false,
    };

    gate: ?metal.Buffer = null,
    up: ?metal.Buffer = null,
    down: ?metal.Buffer = null,

    pub fn init(device: *metal.Device, options: InitOptions) !PreparedMlpDenseRhsBuffers {
        var result: PreparedMlpDenseRhsBuffers = .{};
        errdefer result.deinit();

        if (options.include_gate_up) {
            const gate_up_bytes = try preparedMlpGateUpDenseRhsBf16ByteLen();
            result.gate = try device.createPrivateBuffer(gate_up_bytes);
            result.up = try device.createPrivateBuffer(gate_up_bytes);
        }
        if (options.include_down) {
            result.down = try device.createPrivateBuffer(try preparedMlpDownDenseRhsBf16ByteLen());
        }

        return result;
    }

    pub fn deinit(self: *PreparedMlpDenseRhsBuffers) void {
        if (self.down) |*down| down.destroy();
        if (self.up) |*up| up.destroy();
        if (self.gate) |*gate| gate.destroy();
        self.* = undefined;
    }

    pub fn prepared(self: *const PreparedMlpDenseRhsBuffers) PreparedMlpDenseRhs {
        std.debug.assert((self.gate == null) == (self.up == null));
        return .{
            .gate_up = if (self.gate) |gate| .{ .gate = gate, .up = self.up.? } else null,
            .down = self.down,
        };
    }

    pub fn encodeBf16Command(
        self: *const PreparedMlpDenseRhsBuffers,
        cmd: *metal.CommandBuffer,
        dequant_bf16: metal.Pipeline,
        mlp: *const MlpBlock,
    ) !void {
        if (self.gate) |gate| try mlp.gate_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, gate);
        if (self.up) |up| try mlp.up_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, up);
        if (self.down) |down| try mlp.down_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, down);
    }
};

pub const PreparedMlpBf16Pipelines = struct {
    qmm_m32n64_nt: metal.Pipeline,
    silu_mul_pair: metal.Pipeline,
    qmm_m32n64_nt_residual_add: metal.Pipeline,
    mlx_nt: mlx_nt.Pipelines,
    mlx_nt_64_64_add: metal.Pipeline,
};

fn hiddenBf16Bytes(token_count: u32) !usize {
    return checked_math.product(.{ token_count, HIDDEN, linear_q4.BF16_BYTES });
}

fn intermediateBf16Bytes(token_count: u32) !usize {
    return checked_math.product(.{ token_count, INTERMEDIATE, linear_q4.BF16_BYTES });
}

fn preparedMlpGateUpDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(INTERMEDIATE, HIDDEN);
}

fn preparedMlpDownDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(HIDDEN, INTERMEDIATE);
}

fn prefillMlpGateUpChunkRows(token_count: u32) u32 {
    return @min(token_count, PREFILL_MLP_GATE_UP_CHUNK_ROWS);
}

pub const MlpBlock = struct {
    gate_proj: Q4Linear,
    up_proj: Q4Linear,
    down_proj: Q4Linear,

    pub fn upload(device: *metal.Device, queue: *metal.Queue, repo: *const safetensors.Repository, prefix: []const u8) !MlpBlock {
        var key: [256]u8 = undefined;
        var gate_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.gate_proj", .{prefix}));
        errdefer gate_proj.deinit();
        var up_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.up_proj", .{prefix}));
        errdefer up_proj.deinit();
        const down_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.down_proj", .{prefix}));
        return .{ .gate_proj = gate_proj, .up_proj = up_proj, .down_proj = down_proj };
    }

    pub fn deinit(self: *MlpBlock) void {
        self.gate_proj.deinit();
        self.up_proj.deinit();
        self.down_proj.deinit();
        self.* = undefined;
    }

    pub fn encodePrefillResidualWithPreparedDenseRhsBf16(
        self: *const MlpBlock,
        ws: *metal.Workspace,
        p: PreparedMlpBf16Pipelines,
        x_bf16: metal.Buffer,
        residual_bf16: metal.Buffer,
        token_count: u32,
        prepared: PreparedMlpDenseRhs,
        out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (x_bf16.length < hidden_bytes or residual_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const intermediate_bytes = try intermediateBf16Bytes(token_count);
        const gate = try ws.scratch(intermediate_bytes);

        try self.encodePrefillGateUpSiluWithPreparedDenseRhsBf16(ws, p, x_bf16, token_count, prepared.gate_up, gate);
        try self.encodePrefillDownResidualWithPreparedDenseRhsBf16(ws, p, gate, residual_bf16, token_count, prepared.down, out_bf16);
    }

    fn encodePrefillGateUpSiluWithPreparedDenseRhsBf16(
        self: *const MlpBlock,
        ws: *metal.Workspace,
        p: PreparedMlpBf16Pipelines,
        x_bf16: metal.Buffer,
        token_count: u32,
        prepared_gate_up: ?PreparedMlpGateUpDenseRhs,
        gate_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        const intermediate_values = try checked_math.product(.{ token_count, INTERMEDIATE });
        const intermediate_bytes = try intermediateBf16Bytes(token_count);
        if (gate_bf16.length < intermediate_bytes) return error.OutputBufferTooSmall;

        if (prepared_gate_up) |gate_up| {
            try mlx_nt.encode(ws, p.mlx_nt, .{
                .input = x_bf16,
                .dense_rhs = gate_up.gate,
                .output = gate_bf16,
                .token_count = token_count,
                .out_dim = INTERMEDIATE,
                .in_dim = HIDDEN,
            });

            const up_chunk_rows = prefillMlpGateUpChunkRows(token_count);
            const up = try ws.scratch(try intermediateBf16Bytes(up_chunk_rows));
            var row_offset: u32 = 0;
            while (row_offset < token_count) {
                const rows = @min(up_chunk_rows, token_count - row_offset);
                const row_intermediate_values = try checked_math.product(.{ rows, INTERMEDIATE });
                try mlx_nt.encode(
                    ws,
                    p.mlx_nt,
                    .{
                        .input = x_bf16,
                        .dense_rhs = gate_up.up,
                        .output = up,
                        .token_count = rows,
                        .out_dim = INTERMEDIATE,
                        .in_dim = HIDDEN,
                        .input_byte_offset = try matrixByteOffset(row_offset, HIDDEN),
                    },
                );
                const row_intermediate_values_buf = try ws.u32buf(std.math.cast(u32, row_intermediate_values) orelse return error.ContextSizeOverflow);
                try ws.cmd.dispatch1DWithBindings(
                    p.silu_mul_pair,
                    &.{
                        .{ .buffer = gate_bf16, .offset = try matrixByteOffset(row_offset, INTERMEDIATE) },
                        .{ .buffer = up },
                        .{ .buffer = row_intermediate_values_buf },
                    },
                    row_intermediate_values,
                );
                row_offset += rows;
            }
            return;
        }

        const up = try ws.scratch(intermediate_bytes);
        try self.gate_proj.encodePrefillQmmM32N64NtBf16(ws, p.qmm_m32n64_nt, x_bf16, gate_bf16, token_count);
        try self.up_proj.encodePrefillQmmM32N64NtBf16(ws, p.qmm_m32n64_nt, x_bf16, up, token_count);
        const intermediate_values_buf = try ws.u32buf(std.math.cast(u32, intermediate_values) orelse return error.ContextSizeOverflow);
        try ws.cmd.dispatch1D(p.silu_mul_pair, &.{ gate_bf16, up, intermediate_values_buf }, intermediate_values);
    }

    fn encodePrefillDownResidualWithPreparedDenseRhsBf16(
        self: *const MlpBlock,
        ws: *metal.Workspace,
        p: PreparedMlpBf16Pipelines,
        gate_bf16: metal.Buffer,
        residual_bf16: metal.Buffer,
        token_count: u32,
        prepared_down: ?metal.Buffer,
        out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        const intermediate_bytes = try intermediateBf16Bytes(token_count);
        if (gate_bf16.length < intermediate_bytes or residual_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        if (prepared_down) |down| {
            try mlx_nt.encodeResidualAdd(ws, p.mlx_nt_64_64_add, .{
                .input = gate_bf16,
                .dense_rhs = down,
                .residual = residual_bf16,
                .output = out_bf16,
                .token_count = token_count,
                .out_dim = HIDDEN,
                .in_dim = INTERMEDIATE,
            });
        } else {
            try self.down_proj.encodePrefillQmmM32N64NtBf16ResidualAdd(
                ws,
                p.qmm_m32n64_nt_residual_add,
                gate_bf16,
                residual_bf16,
                out_bf16,
                token_count,
            );
        }
    }

    pub fn decodeResidualBf16(
        self: *const MlpBlock,
        ws: *metal.Workspace,
        qmv_gate_up_silu_bf16: metal.Pipeline,
        qmv_residual_add_bf16: metal.Pipeline,
        x_bf16: metal.Buffer,
        residual_bf16: metal.Buffer,
        out_bf16: metal.Buffer,
    ) !void {
        const hidden_bytes = try hiddenBf16Bytes(1);
        if (x_bf16.length < hidden_bytes or residual_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const gate = try ws.scratch(try intermediateBf16Bytes(1));
        try self.encodeGateUpSiluBf16(ws, qmv_gate_up_silu_bf16, x_bf16, gate);
        ws.barrier();
        try self.down_proj.encodeBf16FastResidualAdd(ws, qmv_residual_add_bf16, gate, residual_bf16, out_bf16, 1);
    }

    fn encodeGateUpSiluBf16(
        self: *const MlpBlock,
        ws: *metal.Workspace,
        pipeline: metal.Pipeline,
        x_bf16: metal.Buffer,
        out_bf16: metal.Buffer,
    ) !void {
        if (self.gate_proj.in_dim != HIDDEN or self.up_proj.in_dim != HIDDEN) return error.UnsupportedQuantization;
        if (self.gate_proj.out_dim != INTERMEDIATE or self.up_proj.out_dim != INTERMEDIATE) return error.UnsupportedQuantization;
        const hidden_bytes = try hiddenBf16Bytes(1);
        const intermediate_bytes = try intermediateBf16Bytes(1);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < intermediate_bytes) return error.OutputBufferTooSmall;

        const grid = try linear_q4.qmvFastGrid(1, INTERMEDIATE);
        try ws.cmd.dispatch1DWithThreadgroup(
            pipeline,
            &.{
                x_bf16,
                self.gate_proj.weight,
                self.gate_proj.scales,
                self.gate_proj.biases,
                self.up_proj.weight,
                self.up_proj.scales,
                self.up_proj.biases,
                out_bf16,
                self.gate_proj.out_dim_buf,
                self.gate_proj.in_dim_buf,
            },
            grid,
            linear_q4.QMV_FAST_THREADS_PER_THREADGROUP,
        );
    }
};

fn matrixByteOffset(row_offset: u32, row_width: u32) !usize {
    return checked_math.product(.{ row_offset, row_width, linear_q4.BF16_BYTES });
}

test "prepared MLP gate-up chunk rows follow Kestrel cap" {
    try std.testing.expectEqual(@as(u32, 1), prefillMlpGateUpChunkRows(1));
    try std.testing.expectEqual(@as(u32, 8192), prefillMlpGateUpChunkRows(8192));
    try std.testing.expectEqual(@as(u32, 8192), prefillMlpGateUpChunkRows(16_000));
}

test "prepared MLP dense RHS byte sizes match Qwen projections" {
    try std.testing.expectEqual(@as(usize, 12288 * 4096 * linear_q4.BF16_BYTES), try preparedMlpGateUpDenseRhsBf16ByteLen());
    try std.testing.expectEqual(@as(usize, 4096 * 12288 * linear_q4.BF16_BYTES), try preparedMlpDownDenseRhsBf16ByteLen());
}

test "decode BF16 gate/up route uses generic Kestrel kernel" {
    try std.testing.expectEqualStrings(
        "linear_vec_q4_affine_group64_multi2_silu_gate_bf16",
        decode_mlp_gate_up_silu_bf16_kernel_name,
    );
}

//! Decoder layer (decode step) — the Qwen3.5-Next residual block:
//!     h   = x + attn(input_layernorm(x))         // attn = full OR linear
//!     out = h + mlp(post_attention_layernorm(h))
//! Layer type is fixed by index: full attention when (idx+1) % 4 == 0, else
//! linear (Gated-DeltaNet). Composition verified against MLX's
//! Qwen3NextDecoderLayer (the residual/norm wiring is the literal source).
//! Exception: this module is temporarily near the 1,000-line guideline while
//! the one-model layer keeps full-attention, linear-attention, and MLP bf16
//! serving paths together.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const dims = @import("dims.zig");
const safetensors = @import("safetensors.zig");
const block_attn = @import("block_attn.zig");
const block_linear = @import("block_linear.zig");
const block_mlp = @import("block_mlp.zig");
const mlx_nt = @import("mlx_nt.zig");
const weight_upload = @import("upload.zig");

const HIDDEN = dims.hidden;
const EPS = dims.rmsnorm_eps;
const FULL_ATTENTION_INTERVAL = 4;
const BF16_BYTES = dims.bf16_bytes;
const BF16_RMSNORM_THREADS = dims.bf16_rmsnorm_threads;
const BF16_ADD_RMSNORM_THREADS = dims.bf16_add_rmsnorm_threads;

/// Full attention at every 4th layer (idx 3,7,...,31); linear elsewhere.
pub fn isLinear(layer_idx: usize) bool {
    return (layer_idx + 1) % FULL_ATTENTION_INTERVAL != 0;
}

fn hiddenBf16Bytes(token_count: u32) !usize {
    const hidden_values = std.math.mul(usize, @as(usize, token_count), HIDDEN) catch return error.ContextSizeOverflow;
    return std.math.mul(usize, hidden_values, BF16_BYTES) catch return error.ContextSizeOverflow;
}

const CorePipelines = struct {
    qmv_bf16: metal.Pipeline,
    q4_dequantize_bf16: metal.Pipeline,
    rmsnorm_bf16: metal.Pipeline,
    add_rmsnorm_bf16: metal.Pipeline,
    silu_mul_pair_bf16: metal.Pipeline,
    sigmoid_mul_pair_bf16: metal.Pipeline,
    qmv_residual_add_bf16: metal.Pipeline,
    qmv_gate_up_silu_bf16: metal.Pipeline,
    qmm_m32n64_nt_bf16: metal.Pipeline,
    qmm_m32n64_nt_residual_add_bf16: metal.Pipeline,
    mlx_nt_bf16: mlx_nt.Pipelines,
    mlx_nt_64_64_add_bf16: metal.Pipeline,

    fn create(device: *metal.Device, library: metal.Library) !CorePipelines {
        var qmv_bf16 = try device.createPipeline(library, "linear_q4_affine_group64_qmv_fast_bf16");
        errdefer qmv_bf16.destroy();
        var q4_dequantize_bf16 = try device.createPipeline(library, "q4_affine_group64_dequantize_bf16");
        errdefer q4_dequantize_bf16.destroy();
        var rmsnorm_bf16 = try device.createPipeline(library, "rmsnorm_bf16");
        errdefer rmsnorm_bf16.destroy();
        var add_rmsnorm_bf16 = try device.createPipeline(library, "add_rmsnorm_bf16");
        errdefer add_rmsnorm_bf16.destroy();
        var silu_mul_pair_bf16 = try device.createPipeline(library, "silu_mul_pair_inplace_bf16");
        errdefer silu_mul_pair_bf16.destroy();
        var sigmoid_mul_pair_bf16 = try device.createPipeline(library, "sigmoid_mul_pair_inplace_bf16");
        errdefer sigmoid_mul_pair_bf16.destroy();
        var qmv_residual_add_bf16 = try device.createPipeline(library, "linear_q4_affine_group64_qmv_fast_residual_add_bf16");
        errdefer qmv_residual_add_bf16.destroy();
        var qmv_gate_up_silu_bf16 = try device.createPipeline(library, block_mlp.decode_mlp_gate_up_silu_bf16_kernel_name);
        errdefer qmv_gate_up_silu_bf16.destroy();
        var qmm_m32n64_nt_bf16 = try device.createPipeline(library, "linear_q4_affine_group64_prefill_qmm_m32n64_nt_bf16_tiles_bf16");
        errdefer qmm_m32n64_nt_bf16.destroy();
        var qmm_m32n64_nt_residual_add_bf16 = try device.createPipeline(library, "linear_q4_affine_group64_prefill_qmm_m32n64_nt_bf16_tiles_residual_add_bf16");
        errdefer qmm_m32n64_nt_residual_add_bf16.destroy();
        var mlx_nt_bf16 = try mlx_nt.Pipelines.createConservative(device, library);
        errdefer mlx_nt_bf16.destroy();
        const mlx_nt_64_64_add_bf16 = try mlx_nt.createConservativeResidualAddPipeline(device, library);

        return .{
            .qmv_bf16 = qmv_bf16,
            .q4_dequantize_bf16 = q4_dequantize_bf16,
            .rmsnorm_bf16 = rmsnorm_bf16,
            .add_rmsnorm_bf16 = add_rmsnorm_bf16,
            .silu_mul_pair_bf16 = silu_mul_pair_bf16,
            .sigmoid_mul_pair_bf16 = sigmoid_mul_pair_bf16,
            .qmv_residual_add_bf16 = qmv_residual_add_bf16,
            .qmv_gate_up_silu_bf16 = qmv_gate_up_silu_bf16,
            .qmm_m32n64_nt_bf16 = qmm_m32n64_nt_bf16,
            .qmm_m32n64_nt_residual_add_bf16 = qmm_m32n64_nt_residual_add_bf16,
            .mlx_nt_bf16 = mlx_nt_bf16,
            .mlx_nt_64_64_add_bf16 = mlx_nt_64_64_add_bf16,
        };
    }

    fn destroy(self: *CorePipelines) void {
        self.qmv_bf16.destroy();
        self.q4_dequantize_bf16.destroy();
        self.rmsnorm_bf16.destroy();
        self.add_rmsnorm_bf16.destroy();
        self.silu_mul_pair_bf16.destroy();
        self.sigmoid_mul_pair_bf16.destroy();
        self.qmv_residual_add_bf16.destroy();
        self.qmv_gate_up_silu_bf16.destroy();
        self.qmm_m32n64_nt_bf16.destroy();
        self.qmm_m32n64_nt_residual_add_bf16.destroy();
        self.mlx_nt_bf16.destroy();
        self.mlx_nt_64_64_add_bf16.destroy();
        self.* = undefined;
    }
};

const FullAttentionPipelines = struct {
    sdpa_flash_partials_bf16: metal.Pipeline,
    sdpa_flash_reduce_bf16: metal.Pipeline,
    prefill_score_bf16: metal.Pipeline,
    prefill_softmax_bf16: metal.Pipeline,
    prefill_softmax_scaled_masked_bf16: metal.Pipeline,
    prefill_query_compact_bf16: metal.Pipeline,
    prefill_value_compact_bf16: metal.Pipeline,
    prefill_value_scatter_bf16: metal.Pipeline,
    prefill_prefix_score_bf16: metal.Pipeline,
    prefill_prefix_value_bf16: metal.Pipeline,
    qk_prepare_prefill_bf16: metal.Pipeline,
    qgate_prepare_kv_append_bf16: metal.Pipeline,

    fn create(device: *metal.Device, library: metal.Library) !FullAttentionPipelines {
        var sdpa_flash_partials_bf16 = try device.createPipeline(library, "sdpa_decode_prefix_flash_partials_bf16");
        errdefer sdpa_flash_partials_bf16.destroy();
        var sdpa_flash_reduce_bf16 = try device.createPipeline(library, "sdpa_decode_prefix_flash_reduce_gated_bf16");
        errdefer sdpa_flash_reduce_bf16.destroy();
        var prefill_score_bf16 = try device.createPipeline(library, "prefill_score_head_major_bf16");
        errdefer prefill_score_bf16.destroy();
        var prefill_softmax_bf16 = try device.createPipeline(library, "prefill_softmax_bf16");
        errdefer prefill_softmax_bf16.destroy();
        var prefill_softmax_scaled_masked_bf16 = try device.createPipeline(library, "prefill_softmax_scaled_masked_bf16");
        errdefer prefill_softmax_scaled_masked_bf16.destroy();
        var prefill_query_compact_bf16 = try device.createPipeline(library, "attention_prefill_compact_query_group_bf16");
        errdefer prefill_query_compact_bf16.destroy();
        var prefill_value_compact_bf16 = try device.createPipeline(library, "attention_prefill_compact_value_dim_major_head_bf16");
        errdefer prefill_value_compact_bf16.destroy();
        var prefill_value_scatter_bf16 = try device.createPipeline(library, "attention_prefill_scatter_group_output_bf16");
        errdefer prefill_value_scatter_bf16.destroy();
        var prefill_prefix_score_bf16 = try device.createPipeline(library, "prefill_score_prefix_head_major_bf16");
        errdefer prefill_prefix_score_bf16.destroy();
        var prefill_prefix_value_bf16 = try device.createPipeline(library, "prefill_value_prefix_head_major_value_dim_major_bf16");
        errdefer prefill_prefix_value_bf16.destroy();
        var qk_prepare_prefill_bf16 = try device.createPipeline(library, "qwen35_qk_norm_rope_append_many_bf16");
        errdefer qk_prepare_prefill_bf16.destroy();
        const qgate_prepare_kv_append_bf16 = try device.createPipeline(library, "qwen35_q_gate_prepare_kv_append_pair_head_major_value_dim_major_bf16");

        return .{
            .sdpa_flash_partials_bf16 = sdpa_flash_partials_bf16,
            .sdpa_flash_reduce_bf16 = sdpa_flash_reduce_bf16,
            .prefill_score_bf16 = prefill_score_bf16,
            .prefill_softmax_bf16 = prefill_softmax_bf16,
            .prefill_softmax_scaled_masked_bf16 = prefill_softmax_scaled_masked_bf16,
            .prefill_query_compact_bf16 = prefill_query_compact_bf16,
            .prefill_value_compact_bf16 = prefill_value_compact_bf16,
            .prefill_value_scatter_bf16 = prefill_value_scatter_bf16,
            .prefill_prefix_score_bf16 = prefill_prefix_score_bf16,
            .prefill_prefix_value_bf16 = prefill_prefix_value_bf16,
            .qk_prepare_prefill_bf16 = qk_prepare_prefill_bf16,
            .qgate_prepare_kv_append_bf16 = qgate_prepare_kv_append_bf16,
        };
    }

    fn destroy(self: *FullAttentionPipelines) void {
        self.sdpa_flash_partials_bf16.destroy();
        self.sdpa_flash_reduce_bf16.destroy();
        self.prefill_score_bf16.destroy();
        self.prefill_softmax_bf16.destroy();
        self.prefill_softmax_scaled_masked_bf16.destroy();
        self.prefill_query_compact_bf16.destroy();
        self.prefill_value_compact_bf16.destroy();
        self.prefill_value_scatter_bf16.destroy();
        self.prefill_prefix_score_bf16.destroy();
        self.prefill_prefix_value_bf16.destroy();
        self.qk_prepare_prefill_bf16.destroy();
        self.qgate_prepare_kv_append_bf16.destroy();
        self.* = undefined;
    }
};

const LinearAttentionPipelines = struct {
    conv1d_bf16: metal.Pipeline,
    conv1d_prefill_vec4_bf16: metal.Pipeline,
    conv1d_prefill_state_bf16: metal.Pipeline,
    qk_norm_prefill_bf16: metal.Pipeline,
    gating_many_bf16: metal.Pipeline,
    gated_delta_prefill_norm_prepared_value_major_bf16: metal.Pipeline,
    gated_delta_decode_fused_bf16: metal.Pipeline,
    rmsnorm_gated_bf16: metal.Pipeline,

    fn create(device: *metal.Device, library: metal.Library) !LinearAttentionPipelines {
        var conv1d_bf16 = try device.createPipeline(library, "conv1d_silu_decode_bf16");
        errdefer conv1d_bf16.destroy();
        var conv1d_prefill_vec4_bf16 = try device.createPipeline(library, "conv1d_silu_prefill_vec4_bf16");
        errdefer conv1d_prefill_vec4_bf16.destroy();
        var conv1d_prefill_state_bf16 = try device.createPipeline(library, "conv1d_silu_prefill_state_bf16");
        errdefer conv1d_prefill_state_bf16.destroy();
        var qk_norm_prefill_bf16 = try device.createPipeline(library, "qk_l2norm_prefill_bf16");
        errdefer qk_norm_prefill_bf16.destroy();
        var gating_many_bf16 = try device.createPipeline(library, "gating_many_bf16");
        errdefer gating_many_bf16.destroy();
        var gated_delta_prefill_norm_prepared_value_major_bf16 = try device.createPipeline(library, "gated_delta_prefill_h128_norm_prepared_value_major_r8_bf16");
        errdefer gated_delta_prefill_norm_prepared_value_major_bf16.destroy();
        var gated_delta_decode_fused_bf16 = try device.createPipeline(library, "gated_delta_decode_h128_fused_bf16");
        errdefer gated_delta_decode_fused_bf16.destroy();
        const rmsnorm_gated_bf16 = try device.createPipeline(library, "rmsnorm_gated_bf16");

        return .{
            .conv1d_bf16 = conv1d_bf16,
            .conv1d_prefill_vec4_bf16 = conv1d_prefill_vec4_bf16,
            .conv1d_prefill_state_bf16 = conv1d_prefill_state_bf16,
            .qk_norm_prefill_bf16 = qk_norm_prefill_bf16,
            .gating_many_bf16 = gating_many_bf16,
            .gated_delta_prefill_norm_prepared_value_major_bf16 = gated_delta_prefill_norm_prepared_value_major_bf16,
            .gated_delta_decode_fused_bf16 = gated_delta_decode_fused_bf16,
            .rmsnorm_gated_bf16 = rmsnorm_gated_bf16,
        };
    }

    fn destroy(self: *LinearAttentionPipelines) void {
        self.conv1d_bf16.destroy();
        self.conv1d_prefill_vec4_bf16.destroy();
        self.conv1d_prefill_state_bf16.destroy();
        self.qk_norm_prefill_bf16.destroy();
        self.gating_many_bf16.destroy();
        self.gated_delta_prefill_norm_prepared_value_major_bf16.destroy();
        self.gated_delta_decode_fused_bf16.destroy();
        self.rmsnorm_gated_bf16.destroy();
        self.* = undefined;
    }
};

const LayoutPipelines = struct {
    split_qg_many_bf16: metal.Pipeline,
    split_qkv_bf16: metal.Pipeline,

    fn create(device: *metal.Device, library: metal.Library) !LayoutPipelines {
        var split_qg_many_bf16 = try device.createPipeline(library, "split_qg_many_bf16");
        errdefer split_qg_many_bf16.destroy();
        const split_qkv_bf16 = try device.createPipeline(library, "split_qkv_bf16");

        return .{
            .split_qg_many_bf16 = split_qg_many_bf16,
            .split_qkv_bf16 = split_qkv_bf16,
        };
    }

    fn destroy(self: *LayoutPipelines) void {
        self.split_qg_many_bf16.destroy();
        self.split_qkv_bf16.destroy();
        self.* = undefined;
    }
};

/// Every pipeline the bf16 serving layer can need.
pub const LayerPipelines = struct {
    core: CorePipelines,
    full: FullAttentionPipelines,
    linear: LinearAttentionPipelines,
    layout: LayoutPipelines,

    pub fn create(device: *metal.Device, library: metal.Library) !LayerPipelines {
        var core = try CorePipelines.create(device, library);
        errdefer core.destroy();
        var full = try FullAttentionPipelines.create(device, library);
        errdefer full.destroy();
        var linear = try LinearAttentionPipelines.create(device, library);
        errdefer linear.destroy();
        var layout = try LayoutPipelines.create(device, library);
        errdefer layout.destroy();

        return .{
            .core = core,
            .full = full,
            .linear = linear,
            .layout = layout,
        };
    }

    pub fn destroy(self: *LayerPipelines) void {
        self.core.destroy();
        self.full.destroy();
        self.linear.destroy();
        self.layout.destroy();
        self.* = undefined;
    }

    fn attnBf16Pipes(self: LayerPipelines) block_attn.Bf16Pipelines {
        return .{
            .qmv = self.core.qmv_bf16,
            .qmm_m32n64_nt = self.core.qmm_m32n64_nt_bf16,
            .mlx_nt = self.core.mlx_nt_bf16,
            .sdpa_flash_partials = self.full.sdpa_flash_partials_bf16,
            .sdpa_flash_reduce = self.full.sdpa_flash_reduce_bf16,
            .qgate_prepare_kv_append = self.full.qgate_prepare_kv_append_bf16,
            .qk_prepare_prefill = self.full.qk_prepare_prefill_bf16,
            .split_qg_many = self.layout.split_qg_many_bf16,
            .prefill_score = self.full.prefill_score_bf16,
            .prefill_softmax = self.full.prefill_softmax_bf16,
            .prefill_softmax_scaled_masked = self.full.prefill_softmax_scaled_masked_bf16,
            .prefill_query_compact = self.full.prefill_query_compact_bf16,
            .prefill_value_compact = self.full.prefill_value_compact_bf16,
            .prefill_value_scatter = self.full.prefill_value_scatter_bf16,
            .prefill_prefix_score = self.full.prefill_prefix_score_bf16,
            .prefill_prefix_value = self.full.prefill_prefix_value_bf16,
            .sigmoid_mul_pair = self.core.sigmoid_mul_pair_bf16,
        };
    }

    fn linearBf16Pipes(self: LayerPipelines) block_linear.Bf16Pipelines {
        return .{
            .qmv = self.core.qmv_bf16,
            .qmm_m32n64_nt = self.core.qmm_m32n64_nt_bf16,
            .mlx_nt = self.core.mlx_nt_bf16,
            .conv1d_decode = self.linear.conv1d_bf16,
            .conv1d_prefill_vec4 = self.linear.conv1d_prefill_vec4_bf16,
            .conv1d_prefill_state = self.linear.conv1d_prefill_state_bf16,
            .qk_norm_prefill = self.linear.qk_norm_prefill_bf16,
            .gating_many = self.linear.gating_many_bf16,
            .gated_delta_prefill_norm_prepared_value_major = self.linear.gated_delta_prefill_norm_prepared_value_major_bf16,
            .gated_delta_decode_fused = self.linear.gated_delta_decode_fused_bf16,
            .rmsnorm_gated = self.linear.rmsnorm_gated_bf16,
            .split_qkv = self.layout.split_qkv_bf16,
        };
    }

    fn mlpPreparedBf16Pipes(self: LayerPipelines) block_mlp.PreparedMlpBf16Pipelines {
        return .{
            .qmm_m32n64_nt = self.core.qmm_m32n64_nt_bf16,
            .silu_mul_pair = self.core.silu_mul_pair_bf16,
            .qmm_m32n64_nt_residual_add = self.core.qmm_m32n64_nt_residual_add_bf16,
            .mlx_nt = self.core.mlx_nt_bf16,
            .mlx_nt_64_64_add = self.core.mlx_nt_64_64_add_bf16,
        };
    }
};

/// Per-layer recurrent state on the GPU (owned by the caller / device model).
pub const LayerState = union(enum) {
    full: struct { cache_k: metal.Buffer, cache_v: metal.Buffer },
    linear: struct { conv: metal.Buffer, recur: metal.Buffer },
};

fn fullPrefixCache(prefix_state: ?*const LayerState, prefix_len: u32) !?block_attn.PrefixCache {
    const prefix_layer = prefix_state orelse return null;
    if (prefix_layer.* != .full) return error.LayerStateMismatch;
    return .{
        .cache_k = prefix_layer.full.cache_k,
        .cache_v = prefix_layer.full.cache_v,
        .len = prefix_len,
    };
}

pub const DecoderLayer = struct {
    input_norm: metal.Buffer,
    post_norm: metal.Buffer,
    mlp: block_mlp.MlpBlock,
    attn: Attn,

    pub const Attn = union(enum) {
        full: block_attn.FullAttentionBlock,
        linear: block_linear.LinearAttentionBlock,
    };

    pub fn upload(device: *metal.Device, queue: *metal.Queue, repo: *const safetensors.Repository, layer_idx: usize) !DecoderLayer {
        var key: [256]u8 = undefined;
        var input_norm = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "language_model.model.layers.{d}.input_layernorm.weight", .{layer_idx}));
        errdefer input_norm.destroy();
        var post_norm = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "language_model.model.layers.{d}.post_attention_layernorm.weight", .{layer_idx}));
        errdefer post_norm.destroy();
        var mlp = try block_mlp.MlpBlock.upload(device, queue, repo, try std.fmt.bufPrint(&key, "language_model.model.layers.{d}.mlp", .{layer_idx}));
        errdefer mlp.deinit();

        const attn: Attn = if (isLinear(layer_idx)) .{
            .linear = try block_linear.LinearAttentionBlock.upload(device, queue, repo, try std.fmt.bufPrint(&key, "language_model.model.layers.{d}.linear_attn", .{layer_idx})),
        } else .{
            .full = try block_attn.FullAttentionBlock.upload(device, queue, repo, try std.fmt.bufPrint(&key, "language_model.model.layers.{d}.self_attn", .{layer_idx})),
        };
        return .{ .input_norm = input_norm, .post_norm = post_norm, .mlp = mlp, .attn = attn };
    }

    pub fn deinit(self: *DecoderLayer) void {
        self.input_norm.destroy();
        self.post_norm.destroy();
        self.mlp.deinit();
        switch (self.attn) {
            .full => |*b| b.deinit(),
            .linear => |*b| b.deinit(),
        }
        self.* = undefined;
    }

    fn encodeRmsNormBf16(
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        weight: metal.Buffer,
        rows: u32,
        out_bf16: metal.Buffer,
    ) !void {
        const hidden_buf = try ws.u32buf(HIDDEN);
        const rows_buf = try ws.u32buf(rows);
        const eps_buf = try ws.f32buf(EPS);
        const threads_buf = try ws.u32buf(BF16_RMSNORM_THREADS);
        try ws.cmd.dispatch1DWithThreadgroup(
            p.core.rmsnorm_bf16,
            &.{ x_bf16, weight, out_bf16, hidden_buf, rows_buf, eps_buf, threads_buf },
            @as(usize, rows) * @as(usize, BF16_RMSNORM_THREADS),
            BF16_RMSNORM_THREADS,
        );
    }

    fn encodeAddRmsNormBf16(
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        add_bf16: metal.Buffer,
        weight: metal.Buffer,
        residual_bf16: metal.Buffer,
        out_bf16: metal.Buffer,
        rows: u32,
    ) !void {
        const hidden_buf = try ws.u32buf(HIDDEN);
        const rows_buf = try ws.u32buf(rows);
        const eps_buf = try ws.f32buf(EPS);
        const threads_buf = try ws.u32buf(BF16_ADD_RMSNORM_THREADS);
        try ws.cmd.dispatch1DWithThreadgroup(
            p.core.add_rmsnorm_bf16,
            &.{ x_bf16, add_bf16, weight, residual_bf16, out_bf16, hidden_buf, rows_buf, eps_buf, threads_buf },
            @as(usize, rows) * @as(usize, BF16_ADD_RMSNORM_THREADS),
            BF16_ADD_RMSNORM_THREADS,
        );
    }

    pub fn decodeStepBf16(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        state: *LayerState,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        prefix_state: ?*const LayerState,
        prefix_len: u32,
        out_bf16: metal.Buffer,
    ) !void {
        const hidden_bytes = try hiddenBf16Bytes(1);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const normed_bf16 = try ws.scratch(hidden_bytes);
        try encodeRmsNormBf16(ws, p, x_bf16, self.input_norm, 1, normed_bf16);
        ws.barrier();

        const attn_out_bf16 = try ws.scratch(hidden_bytes);
        switch (self.attn) {
            .full => |*b| {
                if (state.* != .full) return error.LayerStateMismatch;
                const prefix = try fullPrefixCache(prefix_state, prefix_len);
                try b.decodeStepBf16(
                    ws,
                    p.attnBf16Pipes(),
                    normed_bf16,
                    state.full.cache_k,
                    state.full.cache_v,
                    cache_pos,
                    rope_pos,
                    seq_len,
                    prefix,
                    attn_out_bf16,
                );
            },
            .linear => |*b| {
                if (state.* != .linear) return error.LayerStateMismatch;
                try b.decodeStepBf16(
                    ws,
                    p.linearBf16Pipes(),
                    normed_bf16,
                    state.linear.conv,
                    state.linear.recur,
                    rope_pos != 0,
                    attn_out_bf16,
                );
            },
        }

        ws.barrier();
        try self.encodePostAttentionMlpDecodeBf16(ws, p, x_bf16, attn_out_bf16, out_bf16);
    }

    fn encodeInputNormBf16(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        token_count: u32,
        normed_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (normed_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        try encodeRmsNormBf16(ws, p, x_bf16, self.input_norm, token_count, normed_bf16);
    }

    fn encodeAttentionBf16(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        normed_bf16: metal.Buffer,
        state: *LayerState,
        start_cache_pos: u32,
        start_rope_pos: u32,
        token_count: u32,
        prefix_state: ?*const LayerState,
        prefix_len: u32,
        full_attention_prefill_scratch: ?*const block_attn.PrefillAttentionScratch,
        prepared_full_attention_dense_rhs: block_attn.PreparedFullAttentionDenseRhs,
        prepared_linear_dense_rhs: block_linear.PreparedLinearDenseRhs,
        attn_out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (normed_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (attn_out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        switch (self.attn) {
            .full => |*b| {
                if (state.* != .full) return error.LayerStateMismatch;
                const prefix = try fullPrefixCache(prefix_state, prefix_len);
                try b.encodePrefillWithPreparedDenseRhsBf16(
                    ws,
                    p.attnBf16Pipes(),
                    normed_bf16,
                    state.full.cache_k,
                    state.full.cache_v,
                    start_cache_pos,
                    start_rope_pos,
                    token_count,
                    prefix,
                    full_attention_prefill_scratch,
                    prepared_full_attention_dense_rhs,
                    attn_out_bf16,
                );
            },
            .linear => |*b| {
                if (state.* != .linear) return error.LayerStateMismatch;
                try b.encodePrefillWithPreparedDenseRhsBf16(
                    ws,
                    p.linearBf16Pipes(),
                    normed_bf16,
                    state.linear.conv,
                    state.linear.recur,
                    start_rope_pos != 0,
                    token_count,
                    prepared_linear_dense_rhs,
                    attn_out_bf16,
                );
            },
        }
    }

    pub fn prefillStepBf16(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        state: *LayerState,
        start_cache_pos: u32,
        start_rope_pos: u32,
        token_count: u32,
        prefix_state: ?*const LayerState,
        prefix_len: u32,
        full_attention_prefill_scratch: ?*const block_attn.PrefillAttentionScratch,
        prepared_full_attention_dense_rhs: block_attn.PreparedFullAttentionDenseRhs,
        prepared_linear_dense_rhs: block_linear.PreparedLinearDenseRhs,
        prepared_mlp_dense_rhs: block_mlp.PreparedMlpDenseRhs,
        out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const normed_bf16 = try ws.scratch(hidden_bytes);
        try self.encodeInputNormBf16(ws, p, x_bf16, token_count, normed_bf16);

        const attn_out_bf16 = try ws.scratch(hidden_bytes);
        try self.encodeAttentionBf16(
            ws,
            p,
            normed_bf16,
            state,
            start_cache_pos,
            start_rope_pos,
            token_count,
            prefix_state,
            prefix_len,
            full_attention_prefill_scratch,
            prepared_full_attention_dense_rhs,
            prepared_linear_dense_rhs,
            attn_out_bf16,
        );

        try self.encodePostAttentionMlpBf16WithScratch(
            ws,
            p,
            x_bf16,
            attn_out_bf16,
            token_count,
            prepared_mlp_dense_rhs,
            normed_bf16,
            attn_out_bf16,
            out_bf16,
        );
    }

    fn encodePostAttentionMlpBf16WithScratch(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        attn_out_bf16: metal.Buffer,
        token_count: u32,
        prepared_mlp_dense_rhs: block_mlp.PreparedMlpDenseRhs,
        attention_residual: metal.Buffer,
        post_attention_normed: metal.Buffer,
        out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (x_bf16.length < hidden_bytes or attn_out_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (attention_residual.length < hidden_bytes or post_attention_normed.length < hidden_bytes) return error.OutputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        try self.encodePostAttentionResidualNormBf16(
            ws,
            p,
            x_bf16,
            attn_out_bf16,
            token_count,
            attention_residual,
            post_attention_normed,
        );

        try self.mlp.encodePrefillResidualWithPreparedDenseRhsBf16(
            ws,
            p.mlpPreparedBf16Pipes(),
            post_attention_normed,
            attention_residual,
            token_count,
            prepared_mlp_dense_rhs,
            out_bf16,
        );
    }

    fn encodePostAttentionResidualNormBf16(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        attn_out_bf16: metal.Buffer,
        token_count: u32,
        attention_residual: metal.Buffer,
        post_attention_normed: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenBf16Bytes(token_count);
        if (x_bf16.length < hidden_bytes or attn_out_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (attention_residual.length < hidden_bytes or post_attention_normed.length < hidden_bytes) return error.OutputBufferTooSmall;

        try encodeAddRmsNormBf16(ws, p, x_bf16, attn_out_bf16, self.post_norm, attention_residual, post_attention_normed, token_count);
    }

    fn encodePostAttentionMlpDecodeBf16(
        self: *DecoderLayer,
        ws: *metal.Workspace,
        p: LayerPipelines,
        x_bf16: metal.Buffer,
        attn_out_bf16: metal.Buffer,
        out_bf16: metal.Buffer,
    ) !void {
        const hidden_bytes = try hiddenBf16Bytes(1);
        if (x_bf16.length < hidden_bytes or attn_out_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const attention_residual = try ws.scratch(hidden_bytes);
        const post_attention_normed = try ws.scratch(hidden_bytes);
        try encodeAddRmsNormBf16(ws, p, x_bf16, attn_out_bf16, self.post_norm, attention_residual, post_attention_normed, 1);
        ws.barrier();

        try self.mlp.decodeResidualBf16(
            ws,
            p.core.qmv_gate_up_silu_bf16,
            p.core.qmv_residual_add_bf16,
            post_attention_normed,
            attention_residual,
            out_bf16,
        );
    }
};

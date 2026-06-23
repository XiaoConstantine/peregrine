//! The full Qwen3.5-9B q4 model on the GPU and the greedy decode loop:
//!   token -> q4 embed gather -> 32 decoder layers -> final RMSNorm -> q4 lm_head
//!   -> next-token argmax.
//! Layers 3,7,...,31 are full attention, the rest Gated-DeltaNet linear attention
//! (see decoder_layer.zig). Validated token-exact against mlx_lm greedy decode.
//! Exception: this module is temporarily above the 1,000-line guideline while
//! the one-model serving route owns upload, state, chunked prefill, prepared-RHS
//! scheduling, and decode validation in one place.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const runtime_time = @import("../runtime/time.zig");
const safetensors = @import("safetensors.zig");
const decoder_layer = @import("decoder_layer.zig");
pub const dims = @import("dims.zig");
const block_attn = @import("block_attn.zig");
const block_linear = @import("block_linear.zig");
const block_mlp = @import("block_mlp.zig");
const linear_q4 = @import("linear_q4.zig");
const prefill_arena = @import("prefill_arena.zig");
const prefill_deferred = @import("prefill_deferred.zig");
const state_mod = @import("state.zig");
const weight_upload = @import("upload.zig");
const Q4Linear = linear_q4.Q4Linear;
const argmax = @import("argmax.zig");

pub const NUM_LAYERS = state_mod.NUM_LAYERS;
pub const HIDDEN = dims.hidden;
const EPS = dims.rmsnorm_eps;
pub const MAX_CONTEXT_TOKENS = state_mod.MAX_CONTEXT_TOKENS;
pub const DEFAULT_BATCHED_PREFILL_CHUNK_TOKENS = 1600;
pub const MAX_BATCHED_PREFILL_CHUNK_TOKENS = 1600;
pub const DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE = 10;
pub const MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE = DEFAULT_BATCHED_PREFILL_CHUNK_GROUP_SIZE;
pub const FullCacheDType = state_mod.FullCacheDType;
pub const ModelState = state_mod.ModelState;
pub const BF16_BYTES = dims.bf16_bytes;
const BF16_RMSNORM_THREADS = dims.bf16_rmsnorm_threads;
const QWEN35_PREPARED_MLP_DENSE_RHS_MIN_TOKENS: usize = 3840;
const monotonicNowNs = runtime_time.monotonicNowNs;
const msFromNs = runtime_time.msFromNs;
const log = std.log.scoped(.peregrine_model);

pub const PrefillNextTokenOutput = metal.Buffer;

const LayerMajorHiddenBuffers = struct {
    h0: [MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE]metal.Buffer = undefined,
    h1: [MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE]metal.Buffer = undefined,
    made: usize = 0,

    fn deinit(self: *LayerMajorHiddenBuffers) void {
        prefill_deferred.destroyBuffers(self.h0[0..self.made]);
        prefill_deferred.destroyBuffers(self.h1[0..self.made]);
    }
};

const LayerMajorMlpDenseRhsProjections = struct {
    gate_up: bool = false,
    down: bool = false,

    fn any(self: LayerMajorMlpDenseRhsProjections) bool {
        return self.gate_up or self.down;
    }
};

const LayerMajorPrepareMlpDenseRhs = struct {
    linear_attention: LayerMajorMlpDenseRhsProjections = .{},
    full_attention: LayerMajorMlpDenseRhsProjections = .{},

    const qwen35_9b_default = LayerMajorPrepareMlpDenseRhs{
        .linear_attention = .{ .down = true },
        .full_attention = .{ .gate_up = true, .down = true },
    };
    const all = LayerMajorPrepareMlpDenseRhs{
        .linear_attention = .{ .gate_up = true, .down = true },
        .full_attention = .{ .gate_up = true, .down = true },
    };

    fn any(self: LayerMajorPrepareMlpDenseRhs) bool {
        return self.linear_attention.any() or self.full_attention.any();
    }

    fn forLayerIndex(self: LayerMajorPrepareMlpDenseRhs, layer_index: usize) LayerMajorMlpDenseRhsProjections {
        return if (decoder_layer.isLinear(layer_index)) self.linear_attention else self.full_attention;
    }
};

const HiddenPair = struct {
    first: metal.Buffer,
    second: metal.Buffer,
};

fn hiddenMatrixBytes(token_count: usize, bytes_per_value: usize) !usize {
    return checked_math.product(.{ token_count, HIDDEN, bytes_per_value });
}

fn embeddingVocabSize(shape: []const u64) !u32 {
    if (shape.len != 2) return error.InvalidTensorShape;
    const vocab = std.math.cast(u32, shape[0]) orelse return error.InvalidTensorShape;
    const width = std.math.cast(u32, shape[1]) orelse return error.InvalidTensorShape;
    if (vocab == 0 or width == 0) return error.InvalidTensorShape;
    return vocab;
}

fn prefillGroupTokenCount(token_counts: []const u32) !usize {
    var total: usize = 0;
    for (token_counts) |token_count| {
        total = std.math.add(usize, total, @as(usize, token_count)) catch return error.ContextSizeOverflow;
    }
    return total;
}

fn layerMajorPrepareMlpDenseRhsForShape(group_token_count: usize) LayerMajorPrepareMlpDenseRhs {
    if (group_token_count >= QWEN35_PREPARED_MLP_DENSE_RHS_MIN_TOKENS) return .all;
    return .qwen35_9b_default;
}

fn layerMajorPrepareMlpDenseRhsForGroup(
    group_token_count: usize,
    prepare_policy_token_count: usize,
) LayerMajorPrepareMlpDenseRhs {
    return layerMajorPrepareMlpDenseRhsForShape(@max(group_token_count, prepare_policy_token_count));
}

fn traceStartNs(enabled: bool) u64 {
    return if (enabled) monotonicNowNs() else 0;
}

fn traceLayerMajorLayerDone(
    enabled: bool,
    layer_index: usize,
    is_linear: bool,
    chunk_count: usize,
    token_count: usize,
    start_ns: u64,
    wait_ns: u64,
    waited_layer_index: ?usize,
    wait_gpu_ns: ?u64,
) void {
    if (!enabled) return;
    const elapsed_ns = monotonicNowNs() - start_ns;
    const post_wait_ns = if (elapsed_ns >= wait_ns) elapsed_ns - wait_ns else 0;
    if (waited_layer_index) |waited_index| {
        if (wait_gpu_ns) |gpu_ns| {
            log.info(
                "trace: layer-major layer done index={d} kind={s} chunks={d} tokens={d} wait_for={d} wait_kind={s} wait_ms={d:.3} wait_gpu_ms={d:.3} post_wait_ms={d:.3} elapsed_ms={d:.3}",
                .{
                    layer_index,
                    if (is_linear) "linear" else "full",
                    chunk_count,
                    token_count,
                    waited_index,
                    if (decoder_layer.isLinear(waited_index)) "linear" else "full",
                    msFromNs(wait_ns),
                    msFromNs(gpu_ns),
                    msFromNs(post_wait_ns),
                    msFromNs(elapsed_ns),
                },
            );
        } else {
            log.info(
                "trace: layer-major layer done index={d} kind={s} chunks={d} tokens={d} wait_for={d} wait_kind={s} wait_ms={d:.3} wait_gpu_ms=null post_wait_ms={d:.3} elapsed_ms={d:.3}",
                .{
                    layer_index,
                    if (is_linear) "linear" else "full",
                    chunk_count,
                    token_count,
                    waited_index,
                    if (decoder_layer.isLinear(waited_index)) "linear" else "full",
                    msFromNs(wait_ns),
                    msFromNs(post_wait_ns),
                    msFromNs(elapsed_ns),
                },
            );
        }
    } else {
        log.info(
            "trace: layer-major layer done index={d} kind={s} chunks={d} tokens={d} wait_for=none wait_kind=none wait_ms={d:.3} wait_gpu_ms=null post_wait_ms={d:.3} elapsed_ms={d:.3}",
            .{
                layer_index,
                if (is_linear) "linear" else "full",
                chunk_count,
                token_count,
                msFromNs(wait_ns),
                msFromNs(post_wait_ns),
                msFromNs(elapsed_ns),
            },
        );
    }
}

fn traceLayerMajorFinalWaitDone(
    enabled: bool,
    slot_index: usize,
    wait_ns: u64,
    waited_layer_index: ?usize,
    wait_gpu_ns: ?u64,
) void {
    if (!enabled or waited_layer_index == null) return;
    const waited_index = waited_layer_index.?;
    if (wait_gpu_ns) |gpu_ns| {
        log.info(
            "trace: layer-major final wait slot={d} wait_for={d} wait_kind={s} wait_ms={d:.3} wait_gpu_ms={d:.3}",
            .{ slot_index, waited_index, if (decoder_layer.isLinear(waited_index)) "linear" else "full", msFromNs(wait_ns), msFromNs(gpu_ns) },
        );
    } else {
        log.info(
            "trace: layer-major final wait slot={d} wait_for={d} wait_kind={s} wait_ms={d:.3} wait_gpu_ms=null",
            .{ slot_index, waited_index, if (decoder_layer.isLinear(waited_index)) "linear" else "full", msFromNs(wait_ns) },
        );
    }
}

const QueuedLayerMajorPreparation = struct {
    pending: metal.PendingCommandBuffer,

    fn wait(self: *QueuedLayerMajorPreparation) !void {
        defer self.* = undefined;
        try self.pending.wait();
    }

    fn drain(self: *QueuedLayerMajorPreparation) void {
        self.pending.wait() catch |err| {
            log.warn("failed to drain queued layer-major preparation: {s}", .{@errorName(err)});
        };
        self.* = undefined;
    }
};

fn LayerMajorPreparedRhs(comptime Buffers: type, comptime Prepared: type) type {
    return struct {
        buffers: Buffers,
        queued: ?QueuedLayerMajorPreparation = null,

        const Self = @This();

        fn prepared(self: *const Self) Prepared {
            return self.buffers.prepared();
        }

        fn releaseBuffersAfterBlockCommit(self: *Self) void {
            self.buffers.deinit();
            self.buffers = .{};
        }

        fn destroy(self: *Self) void {
            if (self.queued) |*queued| {
                queued.drain();
                self.queued = null;
            }
            self.buffers.deinit();
            self.* = undefined;
        }
    };
}

const LayerMajorPreparedLinearRhs = LayerMajorPreparedRhs(block_linear.PreparedLinearDenseRhsBuffers, block_linear.PreparedLinearDenseRhs);
const LayerMajorPreparedMlpRhs = LayerMajorPreparedRhs(block_mlp.PreparedMlpDenseRhsBuffers, block_mlp.PreparedMlpDenseRhs);
const LayerMajorPreparedFullAttentionRhs = LayerMajorPreparedRhs(block_attn.PreparedFullAttentionDenseRhsBuffers, block_attn.PreparedFullAttentionDenseRhs);

const LayerMajorPreparedRhsSet = struct {
    linear: ?LayerMajorPreparedLinearRhs = null,
    full_attention: ?LayerMajorPreparedFullAttentionRhs = null,
    mlp: ?LayerMajorPreparedMlpRhs = null,

    fn destroyAll(self: *LayerMajorPreparedRhsSet) void {
        inline for (std.meta.fields(LayerMajorPreparedRhsSet)) |field| {
            if (@field(self, field.name)) |*prepared| {
                prepared.destroy();
                @field(self, field.name) = null;
            }
        }
    }

    fn fullAttentionDense(self: *const LayerMajorPreparedRhsSet) block_attn.PreparedFullAttentionDenseRhs {
        return if (self.full_attention) |*prepared| prepared.prepared() else .{};
    }

    fn linearDense(self: *const LayerMajorPreparedRhsSet) block_linear.PreparedLinearDenseRhs {
        return if (self.linear) |*prepared| prepared.prepared() else .{};
    }

    fn mlpDense(self: *const LayerMajorPreparedRhsSet) block_mlp.PreparedMlpDenseRhs {
        return if (self.mlp) |*prepared| prepared.prepared() else .{};
    }
};

const PendingLayerMajorPreparedRhs = struct {
    linear: [2]?LayerMajorPreparedLinearRhs = .{ null, null },
    full_attention: [2]?LayerMajorPreparedFullAttentionRhs = .{ null, null },
    mlp: [2]?LayerMajorPreparedMlpRhs = .{ null, null },

    fn destroySlot(self: *PendingLayerMajorPreparedRhs, index: usize) void {
        inline for (std.meta.fields(PendingLayerMajorPreparedRhs)) |field| {
            if (@field(self, field.name)[index]) |*prepared| {
                prepared.destroy();
                @field(self, field.name)[index] = null;
            }
        }
    }

    fn destroyAll(self: *PendingLayerMajorPreparedRhs) void {
        for (0..self.linear.len) |index| {
            self.destroySlot(index);
        }
    }

    fn moveFromPrepared(self: *PendingLayerMajorPreparedRhs, pool_index: usize, prepared_rhs: *LayerMajorPreparedRhsSet) void {
        inline for (std.meta.fields(LayerMajorPreparedRhsSet)) |field| {
            if (@field(prepared_rhs, field.name)) |*prepared| {
                prepared.releaseBuffersAfterBlockCommit();
                @field(self, field.name)[pool_index] = prepared.*;
                @field(prepared_rhs, field.name) = null;
            }
        }
    }
};

fn finishPreparedRhsCommand(cmd: *metal.CommandBuffer, cmd_live: *bool, queue_prepare: bool) !?QueuedLayerMajorPreparation {
    cmd_live.* = false;
    if (!queue_prepare) {
        try cmd.commitAndWait();
        return null;
    }

    return .{ .pending = cmd.commit() };
}

fn encodeLayerMajorPreparedRhsCommand(
    queue: *metal.Queue,
    pipeline: metal.Pipeline,
    block: anytype,
    queue_prepare: bool,
    prepared: anytype,
) !void {
    var cmd = try queue.beginCommandBuffer();
    var cmd_live = true;
    errdefer if (cmd_live) cmd.abort();

    try prepared.buffers.encodeBf16Command(&cmd, pipeline, block);
    prepared.queued = try finishPreparedRhsCommand(&cmd, &cmd_live, queue_prepare);
}

fn prepareLayerMajorLinearDenseRhs(
    device: *metal.Device,
    queue: *metal.Queue,
    pipeline: metal.Pipeline,
    linear: *const block_linear.LinearAttentionBlock,
    queue_prepare: bool,
) !LayerMajorPreparedLinearRhs {
    var result = LayerMajorPreparedLinearRhs{
        .buffers = try block_linear.PreparedLinearDenseRhsBuffers.init(device, .{
            .include_qkv_z = true,
            .include_a_b = true,
            .include_out_proj = true,
        }),
    };
    errdefer result.destroy();

    try encodeLayerMajorPreparedRhsCommand(queue, pipeline, linear, queue_prepare, &result);
    return result;
}

fn prepareLayerMajorMlpDenseRhs(
    device: *metal.Device,
    queue: *metal.Queue,
    pipeline: metal.Pipeline,
    mlp: *const block_mlp.MlpBlock,
    projections: LayerMajorMlpDenseRhsProjections,
    queue_prepare: bool,
) !LayerMajorPreparedMlpRhs {
    if (!projections.any()) return error.InvalidPreparedMlpDenseRhsPolicy;
    var result = LayerMajorPreparedMlpRhs{
        .buffers = try block_mlp.PreparedMlpDenseRhsBuffers.init(device, .{
            .include_gate_up = projections.gate_up,
            .include_down = projections.down,
        }),
    };
    errdefer result.destroy();

    try encodeLayerMajorPreparedRhsCommand(queue, pipeline, mlp, queue_prepare, &result);
    return result;
}

fn prepareLayerMajorFullAttentionDenseRhs(
    device: *metal.Device,
    queue: *metal.Queue,
    pipeline: metal.Pipeline,
    full: *const block_attn.FullAttentionBlock,
    queue_prepare: bool,
) !LayerMajorPreparedFullAttentionRhs {
    var result = LayerMajorPreparedFullAttentionRhs{
        .buffers = try block_attn.PreparedFullAttentionDenseRhsBuffers.init(device, .{
            .include_setup = true,
            .include_out_proj = true,
        }),
    };
    errdefer result.destroy();

    try encodeLayerMajorPreparedRhsCommand(queue, pipeline, full, queue_prepare, &result);
    return result;
}

const LayerMajorPreparedLookahead = struct {
    prepared: LayerMajorPreparedRhsSet = .{},

    fn destroyAll(self: *LayerMajorPreparedLookahead) void {
        self.prepared.destroyAll();
    }

    fn takeOrPrepareForLayer(
        self: *LayerMajorPreparedLookahead,
        device: *metal.Device,
        queue: *metal.Queue,
        pipeline: metal.Pipeline,
        layers: []const decoder_layer.DecoderLayer,
        layer_index: usize,
        mlp_prepare: LayerMajorPrepareMlpDenseRhs,
    ) !LayerMajorPreparedRhsSet {
        var result = LayerMajorPreparedRhsSet{};
        errdefer result.destroyAll();

        if (decoder_layer.isLinear(layer_index)) {
            std.debug.assert(self.prepared.full_attention == null);
            result.linear =
                self.takeLinear() orelse
                try prepareLayerMajorLinearDenseRhs(
                    device,
                    queue,
                    pipeline,
                    &layers[layer_index].attn.linear,
                    true,
                );
        } else {
            std.debug.assert(self.prepared.linear == null);
            result.full_attention =
                self.takeFullAttention() orelse
                try prepareLayerMajorFullAttentionDenseRhs(
                    device,
                    queue,
                    pipeline,
                    &layers[layer_index].attn.full,
                    true,
                );
        }

        const mlp_projections = mlp_prepare.forLayerIndex(layer_index);
        if (mlp_projections.any()) {
            result.mlp =
                self.takeMlp() orelse
                try prepareLayerMajorMlpDenseRhs(
                    device,
                    queue,
                    pipeline,
                    &layers[layer_index].mlp,
                    mlp_projections,
                    true,
                );
        } else {
            std.debug.assert(self.prepared.mlp == null);
        }

        return result;
    }

    fn prepareNext(
        self: *LayerMajorPreparedLookahead,
        device: *metal.Device,
        queue: *metal.Queue,
        pipeline: metal.Pipeline,
        layers: []const decoder_layer.DecoderLayer,
        layer_index: usize,
        mlp_prepare: LayerMajorPrepareMlpDenseRhs,
    ) !void {
        const next_layer_index = layer_index + 1;
        if (next_layer_index >= layers.len) return;

        if (decoder_layer.isLinear(next_layer_index)) {
            std.debug.assert(self.prepared.linear == null);
            self.prepared.linear = try prepareLayerMajorLinearDenseRhs(
                device,
                queue,
                pipeline,
                &layers[next_layer_index].attn.linear,
                true,
            );
        } else {
            std.debug.assert(self.prepared.full_attention == null);
            self.prepared.full_attention = try prepareLayerMajorFullAttentionDenseRhs(
                device,
                queue,
                pipeline,
                &layers[next_layer_index].attn.full,
                true,
            );
        }

        const mlp_projections = mlp_prepare.forLayerIndex(next_layer_index);
        if (mlp_projections.any()) {
            std.debug.assert(self.prepared.mlp == null);
            self.prepared.mlp = try prepareLayerMajorMlpDenseRhs(
                device,
                queue,
                pipeline,
                &layers[next_layer_index].mlp,
                mlp_projections,
                true,
            );
        }
    }

    fn takeLinear(self: *LayerMajorPreparedLookahead) ?LayerMajorPreparedLinearRhs {
        if (self.prepared.linear) |prepared| {
            self.prepared.linear = null;
            return prepared;
        }
        return null;
    }

    fn takeFullAttention(self: *LayerMajorPreparedLookahead) ?LayerMajorPreparedFullAttentionRhs {
        if (self.prepared.full_attention) |prepared| {
            self.prepared.full_attention = null;
            return prepared;
        }
        return null;
    }

    fn takeMlp(self: *LayerMajorPreparedLookahead) ?LayerMajorPreparedMlpRhs {
        if (self.prepared.mlp) |prepared| {
            self.prepared.mlp = null;
            return prepared;
        }
        return null;
    }
};

pub const DeviceModel = struct {
    library: metal.Library,
    pipes: decoder_layer.LayerPipelines,
    embed_bf16_pipe: metal.Pipeline,
    embed_many_bf16_pipe: metal.Pipeline,
    copy_row_bf16_pipe: metal.Pipeline,
    write_row_bf16_pipe: metal.Pipeline,
    lm_head_argmax_partial_pipe: metal.Pipeline,
    lm_head_argmax2_partial_pipe: metal.Pipeline,
    argmax_pairs_pipe: metal.Pipeline,

    embed_weight: metal.Buffer,
    embed_scales: metal.Buffer,
    embed_biases: metal.Buffer,
    layers: [NUM_LAYERS]decoder_layer.DecoderLayer,
    final_norm: metal.Buffer,
    lm_head: Q4Linear,
    vocab: u32,

    // owned scratch + constant param buffers for the decode hot path
    tok_buf: metal.Buffer,
    hidden_buf: metal.Buffer,
    eps_buf: metal.Buffer,
    rows1_buf: metal.Buffer,
    scratch_pool: metal.ScratchPool,
    prefill_scratch_pool: metal.ScratchPool,
    prefill_layer_scratch_pools: [2]metal.ScratchPool,

    pub const GeneratedTokenCallback = struct {
        context: *anyopaque,
        emit: *const fn (*anyopaque, []const u32) anyerror!void,
        progress: ?*const fn (*anyopaque, usize, usize) anyerror!void = null,
    };

    pub const PrefixState = struct {
        state: *const ModelState,
        len: u32,
    };

    /// Weight uploads pread into temporary shared staging buffers, then blit into
    /// private GPU-resident buffers. This keeps weight storage in the Metal boundary
    /// without leaving host-mapped resident weight buffers in decode/prefill paths.
    pub fn upload(device: *metal.Device, queue: *metal.Queue, repo: *const safetensors.Repository) !DeviceModel {
        var library = try device.loadDefaultLibrary();
        errdefer library.destroy();
        var pipes = try decoder_layer.LayerPipelines.create(device, library);
        errdefer pipes.destroy();
        var embed_bf16_pipe = try device.createPipeline(library, "q4_embedding_gather_bf16");
        errdefer embed_bf16_pipe.destroy();
        var embed_many_bf16_pipe = try device.createPipeline(library, "q4_embedding_gather_many_bf16");
        errdefer embed_many_bf16_pipe.destroy();
        var copy_row_bf16_pipe = try device.createPipeline(library, "copy_row_bf16");
        errdefer copy_row_bf16_pipe.destroy();
        var write_row_bf16_pipe = try device.createPipeline(library, "write_row_bf16");
        errdefer write_row_bf16_pipe.destroy();
        var lm_head_argmax_partial_pipe = try device.createPipeline(library, "linear_vec_q4_affine_group64_qmv_fast_argmax_partial_bf16");
        errdefer lm_head_argmax_partial_pipe.destroy();
        var lm_head_argmax2_partial_pipe = try device.createPipeline(library, "linear_vec_q4_affine_group64_qmv_fast_argmax2_partial_bf16");
        errdefer lm_head_argmax2_partial_pipe.destroy();
        var argmax_pairs_pipe = try device.createPipeline(library, "argmax_pairs_f32_u32");
        errdefer argmax_pairs_pipe.destroy();

        const embed_info = repo.get("language_model.model.embed_tokens.weight") orelse return error.TensorNotFound;
        const vocab = try embeddingVocabSize(embed_info.shape);
        var embed_weight = try weight_upload.namedTensorPrivate(device, queue, repo, "language_model.model.embed_tokens.weight");
        errdefer embed_weight.destroy();
        var embed_scales = try weight_upload.namedTensorPrivate(device, queue, repo, "language_model.model.embed_tokens.scales");
        errdefer embed_scales.destroy();
        var embed_biases = try weight_upload.namedTensorPrivate(device, queue, repo, "language_model.model.embed_tokens.biases");
        errdefer embed_biases.destroy();

        var layers: [NUM_LAYERS]decoder_layer.DecoderLayer = undefined;
        var made: usize = 0;
        errdefer for (0..made) |i| layers[i].deinit();
        while (made < NUM_LAYERS) : (made += 1) {
            layers[made] = try decoder_layer.DecoderLayer.upload(device, queue, repo, made);
        }

        var final_norm = try weight_upload.namedTensorPrivate(device, queue, repo, "language_model.model.norm.weight");
        errdefer final_norm.destroy();
        var lm_head = try Q4Linear.upload(device, queue, repo, "language_model.lm_head");
        errdefer lm_head.deinit();

        var tok_buf = try device.createSharedBuffer(@sizeOf(u32));
        errdefer tok_buf.destroy();
        var hidden_buf = try device.createSharedBuffer(@sizeOf(u32));
        errdefer hidden_buf.destroy();
        hidden_buf.slice(u32)[0] = HIDDEN;
        var eps_buf = try device.createSharedBuffer(@sizeOf(f32));
        errdefer eps_buf.destroy();
        eps_buf.slice(f32)[0] = EPS;
        var rows1_buf = try device.createSharedBuffer(@sizeOf(u32));
        errdefer rows1_buf.destroy();
        rows1_buf.slice(u32)[0] = 1;

        return .{
            .library = library,
            .pipes = pipes,
            .embed_bf16_pipe = embed_bf16_pipe,
            .embed_many_bf16_pipe = embed_many_bf16_pipe,
            .copy_row_bf16_pipe = copy_row_bf16_pipe,
            .write_row_bf16_pipe = write_row_bf16_pipe,
            .lm_head_argmax_partial_pipe = lm_head_argmax_partial_pipe,
            .lm_head_argmax2_partial_pipe = lm_head_argmax2_partial_pipe,
            .argmax_pairs_pipe = argmax_pairs_pipe,
            .embed_weight = embed_weight,
            .embed_scales = embed_scales,
            .embed_biases = embed_biases,
            .layers = layers,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .vocab = vocab,
            .tok_buf = tok_buf,
            .hidden_buf = hidden_buf,
            .eps_buf = eps_buf,
            .rows1_buf = rows1_buf,
            .scratch_pool = .{},
            .prefill_scratch_pool = .{},
            .prefill_layer_scratch_pools = .{ .{}, .{} },
        };
    }

    pub fn deinit(self: *DeviceModel) void {
        for (&self.prefill_layer_scratch_pools) |*pool| pool.deinit();
        self.prefill_scratch_pool.deinit();
        self.scratch_pool.deinit();
        self.embed_weight.destroy();
        self.embed_scales.destroy();
        self.embed_biases.destroy();
        for (&self.layers) |*l| l.deinit();
        self.final_norm.destroy();
        self.lm_head.deinit();
        self.tok_buf.destroy();
        self.hidden_buf.destroy();
        self.eps_buf.destroy();
        self.rows1_buf.destroy();
        self.argmax_pairs_pipe.destroy();
        self.lm_head_argmax2_partial_pipe.destroy();
        self.lm_head_argmax_partial_pipe.destroy();
        self.write_row_bf16_pipe.destroy();
        self.copy_row_bf16_pipe.destroy();
        self.embed_many_bf16_pipe.destroy();
        self.embed_bf16_pipe.destroy();
        self.pipes.destroy();
        self.library.destroy();
        self.* = undefined;
    }

    fn encodeStackBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
    ) !metal.Buffer {
        try requirePrefixDType(prefix, .bf16);
        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        const h0 = try ws.scratch(hidden_bytes);
        const h1 = try ws.scratch(hidden_bytes);
        const tok_buf = try ws.u32buf(token_id);
        try ws.cmd.dispatch1D(self.embed_bf16_pipe, &.{ self.embed_weight, self.embed_scales, self.embed_biases, h0, tok_buf, self.hidden_buf }, HIDDEN);
        var src = h0;
        var dst = h1;
        for (0..NUM_LAYERS) |i| {
            // Each layer reads the previous layer's output and overwrites the
            // buffer the previous layer read (h0/h1 ping-pong).
            ws.barrier();
            const prefix_layer = if (prefix) |pref| &pref.state.layers[i] else null;
            const prefix_len: u32 = if (prefix) |pref| pref.len else 0;
            try self.layers[i].decodeStepBf16(ws, self.pipes, src, &state.layers[i], cache_pos, rope_pos, seq_len, prefix_layer, prefix_len, dst);
            const tmp = src;
            src = dst;
            dst = tmp;
        }
        ws.barrier();
        return src;
    }

    fn encodeStack2Bf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_ids: [2]u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        start_seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
    ) !HiddenPair {
        try requirePrefixDType(prefix, .bf16);
        for (token_ids) |token_id| {
            if (token_id >= self.vocab) return error.InvalidTokenId;
        }

        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        const a0 = try ws.scratch(hidden_bytes);
        const a1 = try ws.scratch(hidden_bytes);
        const b0 = try ws.scratch(hidden_bytes);
        const b1 = try ws.scratch(hidden_bytes);

        const tok0_buf = try ws.u32buf(token_ids[0]);
        const tok1_buf = try ws.u32buf(token_ids[1]);
        try ws.cmd.dispatch1D(self.embed_bf16_pipe, &.{ self.embed_weight, self.embed_scales, self.embed_biases, a0, tok0_buf, self.hidden_buf }, HIDDEN);
        try ws.cmd.dispatch1D(self.embed_bf16_pipe, &.{ self.embed_weight, self.embed_scales, self.embed_biases, a1, tok1_buf, self.hidden_buf }, HIDDEN);

        var src0 = a0;
        var src1 = a1;
        var dst0 = b0;
        var dst1 = b1;
        for (0..NUM_LAYERS) |i| {
            ws.barrier();
            const prefix_layer = if (prefix) |pref| &pref.state.layers[i] else null;
            const prefix_len: u32 = if (prefix) |pref| pref.len else 0;
            try self.layers[i].decodeStep2Bf16(
                ws,
                self.pipes,
                src0,
                src1,
                &state.layers[i],
                start_cache_pos,
                start_rope_pos,
                start_seq_len,
                prefix_layer,
                prefix_len,
                dst0,
                dst1,
            );
            const tmp0 = src0;
            const tmp1 = src1;
            src0 = dst0;
            src1 = dst1;
            dst0 = tmp0;
            dst1 = tmp1;
        }
        ws.barrier();
        return .{ .first = src0, .second = src1 };
    }

    fn encodeNextTokenBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
    ) !void {
        try self.encodeNextTokenBf16AndMaybeHidden(ws, token_id, cache_pos, rope_pos, seq_len, state, prefix, output_token, null);
    }

    fn encodeNextTokenBf16AndMaybeHidden(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
        output_normalized_hidden_bf16: ?metal.Buffer,
    ) !void {
        try state.requireFullCacheDType(.bf16);
        if (cache_pos >= state.max_seq) return error.SequenceTooLong;
        if (token_id >= self.vocab) return error.InvalidTokenId;
        const src = try self.encodeStackBf16(ws, token_id, cache_pos, rope_pos, seq_len, state, prefix);
        try self.encodeOneTokenArgmaxTailBf16(ws, src, output_token, output_normalized_hidden_bf16);
    }

    pub fn encodeEmbeddingBatchBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_ids_buf: metal.Buffer,
        token_count: u32,
        out_buf: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        if (out_buf.length < try hiddenMatrixBytes(token_count, BF16_BYTES)) return error.OutputBufferTooSmall;
        const token_count_buf = try ws.u32buf(token_count);
        const grid = std.math.mul(usize, @as(usize, token_count), HIDDEN) catch return error.ContextSizeOverflow;
        try ws.cmd.dispatch1D(
            self.embed_many_bf16_pipe,
            &.{ self.embed_weight, self.embed_scales, self.embed_biases, out_buf, token_ids_buf, self.hidden_buf, token_count_buf },
            grid,
        );
    }

    pub fn forwardPrefillBatchGroupWithPrefix(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids: []const u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        chunk_tokens: usize,
        prepare_policy_token_count: usize,
        state: *ModelState,
        prefix: ?PrefixState,
        trace: bool,
        next_token_output: ?PrefillNextTokenOutput,
        normalized_hidden_output: ?metal.Buffer,
        normalized_hidden_output_base_rows: usize,
    ) !void {
        if (token_ids.len == 0) return;
        try state.requireFullCacheDType(.bf16);
        try requirePrefixDType(prefix, .bf16);
        if (chunk_tokens == 0 or chunk_tokens > MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.InvalidPrefillChunk;
        const chunk_count = std.math.divCeil(usize, token_ids.len, chunk_tokens) catch return error.ContextSizeOverflow;
        if (chunk_count == 0 or chunk_count > MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE) return error.InvalidPrefillChunkGroup;
        const end_cache_pos = std.math.add(usize, @as(usize, start_cache_pos), token_ids.len) catch return error.SequenceTooLong;
        if (end_cache_pos > state.max_seq) return error.SequenceTooLong;
        if (normalized_hidden_output) |out| {
            const need = try hiddenMatrixBytes(std.math.cast(u32, token_ids.len) orelse return error.SequenceTooLong, BF16_BYTES);
            if (out.length < need) return error.OutputBufferTooSmall;
        }
        for (token_ids) |token_id| {
            if (token_id >= self.vocab) return error.InvalidTokenId;
        }

        var token_bufs: [MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE]metal.Buffer = undefined;
        var token_counts: [MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE]u32 = undefined;
        var token_offsets: [MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE]u32 = undefined;
        var made_token_bufs: usize = 0;
        defer prefill_deferred.destroyBuffers(token_bufs[0..made_token_bufs]);
        var pos: usize = 0;
        while (pos < token_ids.len) {
            const chunk_end = @min(token_ids.len, pos + chunk_tokens);
            const chunk = token_ids[pos..chunk_end];
            const token_bytes = std.math.mul(usize, chunk.len, @sizeOf(u32)) catch return error.ContextSizeOverflow;
            var token_ids_buf = try device.createSharedBuffer(token_bytes);
            errdefer token_ids_buf.destroy();
            @memcpy(token_ids_buf.slice(u32), chunk);
            token_bufs[made_token_bufs] = token_ids_buf;
            token_counts[made_token_bufs] = std.math.cast(u32, chunk.len) orelse return error.SequenceTooLong;
            token_offsets[made_token_bufs] = std.math.cast(u32, pos) orelse return error.SequenceTooLong;
            made_token_bufs += 1;
            pos = chunk_end;
        }

        if (made_token_bufs > 1 or (next_token_output != null and prefix == null and start_cache_pos != 0) or normalized_hidden_output != null) {
            var hidden_bufs: [MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE]metal.Buffer = undefined;
            var hidden_slice: ?[]metal.Buffer = null;
            var made_hidden_bufs: usize = 0;
            defer if (hidden_slice != null) prefill_deferred.destroyBuffers(hidden_bufs[0..made_hidden_bufs]);
            if (normalized_hidden_output != null) {
                for (token_counts[0..made_token_bufs], 0..) |count, i| {
                    const bytes = try hiddenMatrixBytes(count, BF16_BYTES);
                    var buf = try device.createPrivateBuffer(bytes);
                    errdefer buf.destroy();
                    hidden_bufs[i] = buf;
                    made_hidden_bufs += 1;
                }
                hidden_slice = hidden_bufs[0..made_token_bufs];
            }
            try self.forwardPrefillBatchGroupLayerMajorCacheOnly(
                device,
                queue,
                token_bufs[0..made_token_bufs],
                token_counts[0..made_token_bufs],
                token_offsets[0..made_token_bufs],
                start_cache_pos,
                start_rope_pos,
                prepare_policy_token_count,
                state,
                prefix,
                trace,
                next_token_output,
                hidden_slice,
            );
            if (normalized_hidden_output) |out| {
                for (token_offsets[0..made_token_bufs], 0..) |off, i| {
                    const off_rows = normalized_hidden_output_base_rows + off;
                    const off_bytes = try hiddenMatrixBytes(off_rows, BF16_BYTES);
                    const len_bytes = try hiddenMatrixBytes(token_counts[i], BF16_BYTES);
                    try queue.copyBuffer(hidden_bufs[i], 0, out, off_bytes, len_bytes);
                }
            }
            return;
        }

        if (next_token_output) |output| {
            return self.forwardPrefillBatchNextTokenWithPrefix(device, queue, token_ids, start_cache_pos, start_rope_pos, state, prefix, output);
        }

        var ws = try metal.Workspace.beginWithScratchPool(device, queue, &self.prefill_scratch_pool);
        self.encodePrefillBatchCacheOnly(&ws, token_bufs[0], token_counts[0], start_cache_pos, start_rope_pos, state, prefix) catch |e| {
            ws.abort();
            return e;
        };
        try ws.commitAndWait();
    }

    fn forwardPrefillBatchGroupLayerMajorCacheOnly(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids_bufs: []const metal.Buffer,
        token_counts: []const u32,
        token_offsets: []const u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        prepare_policy_token_count: usize,
        state: *ModelState,
        prefix: ?PrefixState,
        trace: bool,
        next_token_output: ?PrefillNextTokenOutput,
        normalized_hidden_outputs: ?[]const metal.Buffer,
    ) !void {
        if (token_ids_bufs.len == 0 or
            token_ids_bufs.len != token_counts.len or
            token_ids_bufs.len != token_offsets.len or
            token_ids_bufs.len > MAX_BATCHED_PREFILL_CHUNK_GROUP_SIZE)
        {
            return error.InvalidPrefillChunkGroup;
        }
        if (normalized_hidden_outputs) |outputs| {
            if (outputs.len != token_ids_bufs.len) return error.InvalidPrefillChunkGroup;
        }

        const group_token_count = try prefillGroupTokenCount(token_counts);
        const prepare_mlp_dense_rhs = layerMajorPrepareMlpDenseRhsForGroup(group_token_count, prepare_policy_token_count);
        var graph_arena = try prefill_arena.LayerMajorArena.init(device, token_ids_bufs.len);
        defer graph_arena.deinit();

        var hidden = try createLayerMajorHiddenBuffers(device, &graph_arena, token_counts);
        defer hidden.deinit();
        try self.encodeLayerMajorEmbeddings(device, queue, token_ids_bufs, token_counts, &hidden);

        var pending: [2]?metal.PendingCommandBuffer = .{ null, null };
        var pending_layer_index: [2]?usize = .{ null, null };
        var pending_full_attention_scratch: [2]?block_attn.PrefillAttentionScratch = .{ null, null };
        var pending_prepared_rhs = PendingLayerMajorPreparedRhs{};
        var lookahead_prepared_rhs = LayerMajorPreparedLookahead{};
        defer lookahead_prepared_rhs.destroyAll();
        errdefer {
            prefill_deferred.drainAllWithScratch(&pending, &pending_full_attention_scratch);
            pending_prepared_rhs.destroyAll();
        }

        var src = hidden.h0;
        var dst = hidden.h1;
        for (0..NUM_LAYERS) |layer_index| {
            const layer_start_ns = traceStartNs(trace);
            const pool_index = layer_index % self.prefill_layer_scratch_pools.len;
            const wait_start_ns = traceStartNs(trace);
            const waited_layer_index = pending_layer_index[pool_index];
            const wait_gpu_ns = try prefill_deferred.waitSlotWithScratchProfile(&pending, &pending_full_attention_scratch, pool_index);
            pending_layer_index[pool_index] = null;
            const wait_ns = if (trace) monotonicNowNs() - wait_start_ns else 0;
            pending_prepared_rhs.destroySlot(pool_index);
            if (prefix == null and !decoder_layer.isLinear(layer_index)) {
                pending_full_attention_scratch[pool_index] = try graph_arena.createAttentionScratch(device, token_counts, token_offsets, start_cache_pos);
            }
            const full_attention_scratch = if (pending_full_attention_scratch[pool_index]) |*scratch| scratch else null;
            var ws = try metal.Workspace.beginWithScratchPool(device, queue, &self.prefill_layer_scratch_pools[pool_index]);
            const scratch_mark = ws.scratchMark();
            const prefix_layer = if (prefix) |pref| &pref.state.layers[layer_index] else null;
            const prefix_len: u32 = if (prefix) |pref| pref.len else 0;
            var prepared_rhs = try lookahead_prepared_rhs.takeOrPrepareForLayer(
                device,
                queue,
                self.pipes.core.q4_dequantize_bf16,
                self.layers[0..],
                layer_index,
                prepare_mlp_dense_rhs,
            );
            errdefer prepared_rhs.destroyAll();
            const prepared_full_attention_dense_rhs = prepared_rhs.fullAttentionDense();
            const prepared_linear_dense_rhs = prepared_rhs.linearDense();
            const prepared_mlp_dense_rhs = prepared_rhs.mlpDense();
            for (0..token_ids_bufs.len) |chunk_index| {
                const cache_pos = std.math.add(u32, start_cache_pos, token_offsets[chunk_index]) catch |e| {
                    ws.abort();
                    return e;
                };
                const rope_pos = std.math.add(u32, start_rope_pos, token_offsets[chunk_index]) catch |e| {
                    ws.abort();
                    return e;
                };
                self.layers[layer_index].prefillStepBf16(
                    &ws,
                    self.pipes,
                    src[chunk_index],
                    &state.layers[layer_index],
                    cache_pos,
                    rope_pos,
                    token_counts[chunk_index],
                    prefix_layer,
                    prefix_len,
                    full_attention_scratch,
                    prepared_full_attention_dense_rhs,
                    prepared_linear_dense_rhs,
                    prepared_mlp_dense_rhs,
                    dst[chunk_index],
                ) catch |e| {
                    ws.abort();
                    return e;
                };
                ws.resetReusableScratchTo(scratch_mark);
            }
            if (next_token_output) |output| {
                if (layer_index + 1 == NUM_LAYERS) {
                    self.encodePrefillArgmaxTailBf16(&ws, dst[token_ids_bufs.len - 1], token_counts[token_ids_bufs.len - 1], output) catch |e| {
                        ws.abort();
                        return e;
                    };
                }
            }
            if (normalized_hidden_outputs) |outputs| {
                if (layer_index + 1 == NUM_LAYERS) {
                    for (0..token_ids_bufs.len) |chunk_index| {
                        self.encodeFinalNormRowsBf16(&ws, dst[chunk_index], token_counts[chunk_index], outputs[chunk_index]) catch |e| {
                            ws.abort();
                            return e;
                        };
                    }
                }
            }
            pending[pool_index] = ws.commitPooled();
            pending_layer_index[pool_index] = layer_index;
            if (pending_full_attention_scratch[pool_index]) |*scratch| {
                scratch.makeAliasableIfHeapBacked();
            }
            pending_prepared_rhs.moveFromPrepared(pool_index, &prepared_rhs);
            try lookahead_prepared_rhs.prepareNext(
                device,
                queue,
                self.pipes.core.q4_dequantize_bf16,
                self.layers[0..],
                layer_index,
                prepare_mlp_dense_rhs,
            );
            for (0..token_ids_bufs.len) |chunk_index| {
                const tmp = src[chunk_index];
                src[chunk_index] = dst[chunk_index];
                dst[chunk_index] = tmp;
            }
            traceLayerMajorLayerDone(
                trace,
                layer_index,
                decoder_layer.isLinear(layer_index),
                token_ids_bufs.len,
                group_token_count,
                layer_start_ns,
                wait_ns,
                waited_layer_index,
                wait_gpu_ns,
            );
        }
        try waitLayerMajorPendingFinal(trace, &pending, &pending_layer_index, &pending_full_attention_scratch);
        pending_prepared_rhs.destroyAll();
    }

    pub fn forwardPrefillBatchHiddenChunkWithPrefix(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids: []const u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        prepare_policy_token_count: usize,
        state: *ModelState,
        prefix: ?PrefixState,
        trace: bool,
        next_token_output: ?PrefillNextTokenOutput,
        normalized_hidden_output: metal.Buffer,
    ) !void {
        if (token_ids.len == 0) return;
        try state.requireFullCacheDType(.bf16);
        try requirePrefixDType(prefix, .bf16);
        if (token_ids.len > MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.PrefillChunkTooLarge;
        const token_count = std.math.cast(u32, token_ids.len) orelse return error.SequenceTooLong;
        const end_cache_pos = std.math.add(usize, @as(usize, start_cache_pos), token_ids.len) catch return error.SequenceTooLong;
        if (end_cache_pos > state.max_seq) return error.SequenceTooLong;
        if (normalized_hidden_output.length < try hiddenMatrixBytes(token_count, BF16_BYTES)) return error.OutputBufferTooSmall;
        for (token_ids) |token_id| {
            if (token_id >= self.vocab) return error.InvalidTokenId;
        }

        const token_bytes = std.math.mul(usize, token_ids.len, @sizeOf(u32)) catch return error.ContextSizeOverflow;
        var token_ids_buf = try device.createSharedBuffer(token_bytes);
        defer token_ids_buf.destroy();
        @memcpy(token_ids_buf.slice(u32), token_ids);

        var token_bufs = [_]metal.Buffer{token_ids_buf};
        var token_counts = [_]u32{token_count};
        var token_offsets = [_]u32{0};
        var hidden_outputs = [_]metal.Buffer{normalized_hidden_output};
        try self.forwardPrefillBatchGroupLayerMajorCacheOnly(
            device,
            queue,
            token_bufs[0..],
            token_counts[0..],
            token_offsets[0..],
            start_cache_pos,
            start_rope_pos,
            prepare_policy_token_count,
            state,
            prefix,
            trace,
            next_token_output,
            hidden_outputs[0..],
        );
    }

    fn createLayerMajorHiddenBuffers(
        device: *metal.Device,
        graph_arena: *prefill_arena.LayerMajorArena,
        token_counts: []const u32,
    ) !LayerMajorHiddenBuffers {
        var hidden = LayerMajorHiddenBuffers{};
        errdefer hidden.deinit();
        for (token_counts, 0..) |token_count, chunk_index| {
            if (token_count == 0 or token_count > MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.InvalidPrefillChunk;
            const hidden_bytes = try hiddenMatrixBytes(token_count, BF16_BYTES);
            hidden.h0[chunk_index] = try graph_arena.createPrivateBuffer(device, hidden_bytes);
            hidden.h1[chunk_index] = graph_arena.createPrivateBuffer(device, hidden_bytes) catch |e| {
                hidden.h0[chunk_index].destroy();
                return e;
            };
            hidden.made += 1;
        }
        return hidden;
    }

    fn encodeLayerMajorEmbeddings(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids_bufs: []const metal.Buffer,
        token_counts: []const u32,
        hidden: *const LayerMajorHiddenBuffers,
    ) !void {
        var ws = try metal.Workspace.beginWithScratchPool(device, queue, &self.prefill_scratch_pool);
        for (token_ids_bufs, token_counts, 0..) |token_ids_buf, token_count, chunk_index| {
            self.encodeEmbeddingBatchBf16(&ws, token_ids_buf, token_count, hidden.h0[chunk_index]) catch |e| {
                ws.abort();
                return e;
            };
        }
        try ws.commitAndWait();
    }

    fn waitLayerMajorPendingFinal(
        trace: bool,
        pending: *[2]?metal.PendingCommandBuffer,
        pending_layer_index: *[2]?usize,
        pending_full_attention_scratch: *[2]?block_attn.PrefillAttentionScratch,
    ) !void {
        for (0..pending.len) |pool_index| {
            const wait_start_ns = traceStartNs(trace);
            const waited_layer_index = pending_layer_index[pool_index];
            const wait_gpu_ns = try prefill_deferred.waitSlotWithScratchProfile(pending, pending_full_attention_scratch, pool_index);
            pending_layer_index[pool_index] = null;
            const wait_ns = if (trace) monotonicNowNs() - wait_start_ns else 0;
            traceLayerMajorFinalWaitDone(trace, pool_index, wait_ns, waited_layer_index, wait_gpu_ns);
        }
    }

    pub fn forwardPrefillBatchNextTokenWithPrefix(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids: []const u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        next_token_output: PrefillNextTokenOutput,
    ) !void {
        if (token_ids.len == 0) return;
        try state.requireFullCacheDType(.bf16);
        try requirePrefixDType(prefix, .bf16);
        if (token_ids.len > MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.PrefillChunkTooLarge;
        const token_count = std.math.cast(u32, token_ids.len) orelse return error.SequenceTooLong;
        const end_cache_pos = std.math.add(usize, @as(usize, start_cache_pos), token_ids.len) catch return error.SequenceTooLong;
        if (end_cache_pos > state.max_seq) return error.SequenceTooLong;
        for (token_ids) |token_id| {
            if (token_id >= self.vocab) return error.InvalidTokenId;
        }

        const token_bytes = std.math.mul(usize, token_ids.len, @sizeOf(u32)) catch return error.ContextSizeOverflow;
        var token_ids_buf = try device.createSharedBuffer(token_bytes);
        defer token_ids_buf.destroy();
        @memcpy(token_ids_buf.slice(u32), token_ids);

        var ws = try metal.Workspace.beginWithScratchPool(device, queue, &self.prefill_scratch_pool);
        const hidden = self.encodePrefillBatchHidden(&ws, token_ids_buf, token_count, start_cache_pos, start_rope_pos, state, prefix) catch |e| {
            ws.abort();
            return e;
        };
        self.encodePrefillArgmaxTailBf16(&ws, hidden, token_count, next_token_output) catch |e| {
            ws.abort();
            return e;
        };
        try ws.commitAndWait();
    }

    pub fn forwardPrefillBatchNextTokenAndHiddenWithPrefix(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids: []const u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        next_token_output: PrefillNextTokenOutput,
        normalized_hidden_output: metal.Buffer,
    ) !void {
        if (token_ids.len == 0) return;
        try state.requireFullCacheDType(.bf16);
        try requirePrefixDType(prefix, .bf16);
        if (token_ids.len > MAX_BATCHED_PREFILL_CHUNK_TOKENS) return error.PrefillChunkTooLarge;
        const token_count = std.math.cast(u32, token_ids.len) orelse return error.SequenceTooLong;
        if (normalized_hidden_output.length < try hiddenMatrixBytes(token_count, BF16_BYTES)) return error.OutputBufferTooSmall;
        const end_cache_pos = std.math.add(usize, @as(usize, start_cache_pos), token_ids.len) catch return error.SequenceTooLong;
        if (end_cache_pos > state.max_seq) return error.SequenceTooLong;
        for (token_ids) |token_id| {
            if (token_id >= self.vocab) return error.InvalidTokenId;
        }

        const token_bytes = std.math.mul(usize, token_ids.len, @sizeOf(u32)) catch return error.ContextSizeOverflow;
        var token_ids_buf = try device.createSharedBuffer(token_bytes);
        defer token_ids_buf.destroy();
        @memcpy(token_ids_buf.slice(u32), token_ids);

        var ws = try metal.Workspace.beginWithScratchPool(device, queue, &self.prefill_scratch_pool);
        var committed = false;
        errdefer if (!committed) ws.abort();

        const hidden = try self.encodePrefillBatchHidden(&ws, token_ids_buf, token_count, start_cache_pos, start_rope_pos, state, prefix);
        try self.encodeFinalNormRowsBf16(&ws, hidden, token_count, normalized_hidden_output);

        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        const last_hidden = try ws.scratch(hidden_bytes);
        const last_row = try ws.u32buf(token_count - 1);
        try ws.cmd.dispatch1D(self.copy_row_bf16_pipe, &.{ normalized_hidden_output, last_hidden, last_row, self.hidden_buf }, HIDDEN);
        try self.encodeArgmaxFromNormalizedHiddenBf16(&ws, last_hidden, next_token_output, false);
        committed = true;
        try ws.commitAndWait();
    }

    pub fn encodeFinalNormRowsBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        hidden_bf16: metal.Buffer,
        token_count: u32,
        output_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenMatrixBytes(token_count, BF16_BYTES);
        if (hidden_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (output_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const rows_buf = try ws.u32buf(token_count);
        const threads_buf = try ws.u32buf(BF16_RMSNORM_THREADS);
        try ws.cmd.dispatch1DWithThreadgroup(
            self.pipes.core.rmsnorm_bf16,
            &.{ hidden_bf16, self.final_norm, output_bf16, self.hidden_buf, rows_buf, self.eps_buf, threads_buf },
            @as(usize, token_count) * @as(usize, BF16_RMSNORM_THREADS),
            BF16_RMSNORM_THREADS,
        );
    }

    fn encodePrefillArgmaxTailBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        hidden: metal.Buffer,
        token_count: u32,
        output_token: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        if (hidden.length < try hiddenMatrixBytes(token_count, BF16_BYTES)) return error.InputBufferTooSmall;
        if (output_token.length < @sizeOf(u32)) return error.OutputBufferTooSmall;

        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        const last_hidden = try ws.scratch(hidden_bytes);

        const last_row = try ws.u32buf(token_count - 1);
        try ws.cmd.dispatch1D(self.copy_row_bf16_pipe, &.{ hidden, last_hidden, last_row, self.hidden_buf }, HIDDEN);
        try self.encodeArgmaxFromHiddenBf16(ws, last_hidden, output_token, false, null);
    }

    fn encodeOneTokenArgmaxTailBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        hidden_bf16: metal.Buffer,
        output_token: metal.Buffer,
        output_normalized_hidden_bf16: ?metal.Buffer,
    ) !void {
        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        if (hidden_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (output_token.length < @sizeOf(u32)) return error.OutputBufferTooSmall;
        if (output_normalized_hidden_bf16) |out| {
            if (out.length < hidden_bytes) return error.OutputBufferTooSmall;
        }
        try self.encodeArgmaxFromHiddenBf16(ws, hidden_bf16, output_token, true, output_normalized_hidden_bf16);
    }

    fn encodeArgmaxFromHiddenBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        hidden_bf16: metal.Buffer,
        output_token: metal.Buffer,
        with_serial_barriers: bool,
        output_normalized_hidden_bf16: ?metal.Buffer,
    ) !void {
        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        const final_normed = try ws.scratch(hidden_bytes);

        const threads_buf = try ws.u32buf(BF16_RMSNORM_THREADS);
        try ws.cmd.dispatch1DWithThreadgroup(
            self.pipes.core.rmsnorm_bf16,
            &.{ hidden_bf16, self.final_norm, final_normed, self.hidden_buf, self.rows1_buf, self.eps_buf, threads_buf },
            BF16_RMSNORM_THREADS,
            BF16_RMSNORM_THREADS,
        );
        if (with_serial_barriers) ws.barrier();
        if (output_normalized_hidden_bf16) |out| {
            const row0 = try ws.u32buf(0);
            try ws.cmd.dispatch1D(self.copy_row_bf16_pipe, &.{ final_normed, out, row0, self.hidden_buf }, HIDDEN);
            if (with_serial_barriers) ws.barrier();
        }
        try self.encodeArgmaxFromNormalizedHiddenBf16(ws, final_normed, output_token, with_serial_barriers);
    }

    pub fn encodeArgmaxFromNormalizedHiddenBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        normalized_hidden_bf16: metal.Buffer,
        output_token: metal.Buffer,
        with_serial_barriers: bool,
    ) !void {
        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        if (normalized_hidden_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (output_token.length < @sizeOf(u32)) return error.OutputBufferTooSmall;
        const partial_count = try argmax.q4AffineFastArgmaxPartialCount(self.vocab);
        const partial_values_bytes = try checked_math.product(.{ partial_count, @sizeOf(f32) });
        const partial_indices_bytes = try checked_math.product(.{ partial_count, @sizeOf(u32) });
        const partial_values = try ws.scratch(partial_values_bytes);
        const partial_indices = try ws.scratch(partial_indices_bytes);

        try self.lm_head.encodeBf16ArgmaxPartial(ws, self.lm_head_argmax_partial_pipe, normalized_hidden_bf16, partial_values, partial_indices);
        if (with_serial_barriers) ws.barrier();
        try argmax.encodePairsF32U32(ws, self.argmax_pairs_pipe, partial_values, partial_indices, output_token, partial_count);
    }

    fn encodeArgmaxFromTwoHiddenBf16(
        self: *DeviceModel,
        ws: *metal.Workspace,
        hidden0_bf16: metal.Buffer,
        hidden1_bf16: metal.Buffer,
        output0_token: metal.Buffer,
        output1_token: metal.Buffer,
        output_normalized_hidden_bf16: ?metal.Buffer,
    ) !void {
        const hidden_bytes = try checked_math.product(.{ HIDDEN, BF16_BYTES });
        if (hidden0_bf16.length < hidden_bytes or hidden1_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (output0_token.length < @sizeOf(u32) or output1_token.length < @sizeOf(u32)) return error.OutputBufferTooSmall;
        if (output_normalized_hidden_bf16) |out| {
            if (out.length < try hiddenMatrixBytes(2, BF16_BYTES)) return error.OutputBufferTooSmall;
        }

        const norm0 = try ws.scratch(hidden_bytes);
        const norm1 = try ws.scratch(hidden_bytes);
        const threads_buf = try ws.u32buf(BF16_RMSNORM_THREADS);
        try ws.cmd.dispatch1DWithThreadgroup(
            self.pipes.core.rmsnorm_bf16,
            &.{ hidden0_bf16, self.final_norm, norm0, self.hidden_buf, self.rows1_buf, self.eps_buf, threads_buf },
            BF16_RMSNORM_THREADS,
            BF16_RMSNORM_THREADS,
        );
        try ws.cmd.dispatch1DWithThreadgroup(
            self.pipes.core.rmsnorm_bf16,
            &.{ hidden1_bf16, self.final_norm, norm1, self.hidden_buf, self.rows1_buf, self.eps_buf, threads_buf },
            BF16_RMSNORM_THREADS,
            BF16_RMSNORM_THREADS,
        );
        ws.barrier();
        if (output_normalized_hidden_bf16) |out| {
            const row0 = try ws.u32buf(0);
            const row1 = try ws.u32buf(1);
            try ws.cmd.dispatch1D(self.write_row_bf16_pipe, &.{ norm0, out, row0, self.hidden_buf }, HIDDEN);
            try ws.cmd.dispatch1D(self.write_row_bf16_pipe, &.{ norm1, out, row1, self.hidden_buf }, HIDDEN);
            ws.barrier();
        }

        const partial_count = try argmax.q4AffineFastArgmaxPartialCount(self.vocab);
        const partial_values_bytes = try checked_math.product(.{ partial_count, @sizeOf(f32) });
        const partial_indices_bytes = try checked_math.product(.{ partial_count, @sizeOf(u32) });
        const partial_values0 = try ws.scratch(partial_values_bytes);
        const partial_indices0 = try ws.scratch(partial_indices_bytes);
        const partial_values1 = try ws.scratch(partial_values_bytes);
        const partial_indices1 = try ws.scratch(partial_indices_bytes);

        try self.lm_head.encodeBf16ArgmaxPartial2(
            ws,
            self.lm_head_argmax2_partial_pipe,
            norm0,
            norm1,
            partial_values0,
            partial_indices0,
            partial_values1,
            partial_indices1,
        );
        ws.barrier();
        try argmax.encodePairsF32U32(ws, self.argmax_pairs_pipe, partial_values0, partial_indices0, output0_token, partial_count);
        try argmax.encodePairsF32U32(ws, self.argmax_pairs_pipe, partial_values1, partial_indices1, output1_token, partial_count);
    }

    fn encodePrefillBatchCacheOnly(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_ids_buf: metal.Buffer,
        token_count: u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        state: *ModelState,
        prefix: ?PrefixState,
    ) !void {
        _ = try self.encodePrefillBatchHidden(ws, token_ids_buf, token_count, start_cache_pos, start_rope_pos, state, prefix);
    }

    fn encodePrefillBatchHidden(
        self: *DeviceModel,
        ws: *metal.Workspace,
        token_ids_buf: metal.Buffer,
        token_count: u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        state: *ModelState,
        prefix: ?PrefixState,
    ) !metal.Buffer {
        try state.requireFullCacheDType(.bf16);
        try requirePrefixDType(prefix, .bf16);
        const hidden_bytes = try hiddenMatrixBytes(token_count, BF16_BYTES);
        const h0 = try ws.scratch(hidden_bytes);
        const h1 = try ws.scratch(hidden_bytes);
        try self.encodeEmbeddingBatchBf16(ws, token_ids_buf, token_count, h0);

        var src = h0;
        var dst = h1;
        const scratch_mark = ws.scratchMark();
        for (0..NUM_LAYERS) |i| {
            const prefix_layer = if (prefix) |pref| &pref.state.layers[i] else null;
            const prefix_len: u32 = if (prefix) |pref| pref.len else 0;
            try self.layers[i].prefillStepBf16(
                ws,
                self.pipes,
                src,
                &state.layers[i],
                start_cache_pos,
                start_rope_pos,
                token_count,
                prefix_layer,
                prefix_len,
                null,
                .{},
                .{},
                .{},
                dst,
            );
            const tmp = src;
            src = dst;
            dst = tmp;
            ws.resetReusableScratchTo(scratch_mark);
        }
        return src;
    }

    fn requirePrefixDType(prefix: ?PrefixState, dtype: FullCacheDType) !void {
        if (prefix) |pref| try pref.state.requireFullCacheDType(dtype);
    }

    fn encodeNextTokenWorkspace(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
    ) !metal.Workspace {
        var ws = try metal.Workspace.beginConcurrentWithScratchPool(device, queue, &self.scratch_pool);
        errdefer ws.abort();
        try self.encodeNextTokenBf16(&ws, token_id, cache_pos, rope_pos, seq_len, state, prefix, output_token);
        return ws;
    }

    fn encodeNextTokenWorkspaceAndMaybeHidden(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
        output_normalized_hidden_bf16: ?metal.Buffer,
    ) !metal.Workspace {
        var ws = try metal.Workspace.beginConcurrentWithScratchPool(device, queue, &self.scratch_pool);
        errdefer ws.abort();
        try self.encodeNextTokenBf16AndMaybeHidden(
            &ws,
            token_id,
            cache_pos,
            rope_pos,
            seq_len,
            state,
            prefix,
            output_token,
            output_normalized_hidden_bf16,
        );
        return ws;
    }

    pub fn forwardNextTokenBf16WithPrefix(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
    ) !void {
        var ws = try self.encodeNextTokenWorkspace(device, queue, token_id, cache_pos, rope_pos, seq_len, state, prefix, output_token);
        try ws.commitAndWait();
    }

    /// Trace-only variant of `forwardNextTokenBf16WithPrefix` that reports the
    /// command buffer's GPU execution time. Serving uses the unprofiled path.
    pub fn forwardNextTokenBf16WithPrefixProfiled(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
    ) !?u64 {
        var ws = try self.encodeNextTokenWorkspace(device, queue, token_id, cache_pos, rope_pos, seq_len, state, prefix, output_token);
        var pending = ws.commitPooled();
        return try pending.waitProfile();
    }

    pub fn forwardNextTokenBf16WithPrefixAndHiddenProfiled(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_id: u32,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        output_token: metal.Buffer,
        output_normalized_hidden_bf16: metal.Buffer,
    ) !?u64 {
        var ws = try self.encodeNextTokenWorkspaceAndMaybeHidden(
            device,
            queue,
            token_id,
            cache_pos,
            rope_pos,
            seq_len,
            state,
            prefix,
            output_token,
            output_normalized_hidden_bf16,
        );
        var pending = ws.commitPooled();
        return try pending.waitProfile();
    }

    pub fn forwardTwoTokenDecodeVerifierBf16WithPrefixProfiled(
        self: *DeviceModel,
        device: *metal.Device,
        queue: *metal.Queue,
        token_ids: [2]u32,
        start_cache_pos: u32,
        start_rope_pos: u32,
        start_seq_len: u32,
        state: *ModelState,
        prefix: ?PrefixState,
        verify_token_output: metal.Buffer,
        bonus_token_output: metal.Buffer,
        normalized_hidden_output: ?metal.Buffer,
    ) !?u64 {
        try state.requireFullCacheDType(.bf16);
        try requirePrefixDType(prefix, .bf16);
        if (verify_token_output.length < @sizeOf(u32) or bonus_token_output.length < @sizeOf(u32)) return error.OutputBufferTooSmall;
        const end_cache_pos = std.math.add(usize, @as(usize, start_cache_pos), 2) catch return error.SequenceTooLong;
        if (end_cache_pos > state.max_seq) return error.SequenceTooLong;
        for (token_ids) |token_id| {
            if (token_id >= self.vocab) return error.InvalidTokenId;
        }

        var ws = try metal.Workspace.beginConcurrentWithScratchPool(device, queue, &self.scratch_pool);
        var committed = false;
        errdefer if (!committed) ws.abort();

        const hidden = try self.encodeStack2Bf16(&ws, token_ids, start_cache_pos, start_rope_pos, start_seq_len, state, prefix);
        try self.encodeArgmaxFromTwoHiddenBf16(&ws, hidden.first, hidden.second, verify_token_output, bonus_token_output, normalized_hidden_output);

        var pending = ws.commitPooled();
        committed = true;
        return try pending.waitProfile();
    }
};

test "layer-major prepared RHS policy matches serving defaults" {
    try std.testing.expectEqual(@as(usize, 3840), try prefillGroupTokenCount(&.{ 1600, 1600, 640 }));

    const default_mlp_rhs = layerMajorPrepareMlpDenseRhsForShape(QWEN35_PREPARED_MLP_DENSE_RHS_MIN_TOKENS - 1);
    try std.testing.expect(default_mlp_rhs.any());
    try std.testing.expect(!default_mlp_rhs.linear_attention.gate_up);
    try std.testing.expect(default_mlp_rhs.linear_attention.down);
    try std.testing.expect(default_mlp_rhs.full_attention.gate_up);
    try std.testing.expect(default_mlp_rhs.full_attention.down);

    const long_mlp_rhs = layerMajorPrepareMlpDenseRhsForShape(QWEN35_PREPARED_MLP_DENSE_RHS_MIN_TOKENS);
    try std.testing.expect(long_mlp_rhs.linear_attention.gate_up);
    try std.testing.expect(long_mlp_rhs.linear_attention.down);
    try std.testing.expect(long_mlp_rhs.full_attention.gate_up);
    try std.testing.expect(long_mlp_rhs.full_attention.down);

    const short_tail_long_request_rhs = layerMajorPrepareMlpDenseRhsForGroup(153, 16_153);
    try std.testing.expect(short_tail_long_request_rhs.linear_attention.gate_up);
    try std.testing.expect(short_tail_long_request_rhs.linear_attention.down);
    try std.testing.expect(short_tail_long_request_rhs.full_attention.gate_up);
    try std.testing.expect(short_tail_long_request_rhs.full_attention.down);
}

test "embeddingVocabSize validates safetensors embedding shape" {
    try std.testing.expectEqual(@as(u32, 151936), try embeddingVocabSize(&.{ 151936, 512 }));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{}));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{151936}));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{ 151936, 512, 1 }));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{ 0, 512 }));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{ 151936, 0 }));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{ std.math.maxInt(u64), 512 }));
    try std.testing.expectError(error.InvalidTensorShape, embeddingVocabSize(&.{ 151936, std.math.maxInt(u64) }));
}

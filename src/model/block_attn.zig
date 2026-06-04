//! Full-attention block (decode step) — composes the verified kernels via
//! Q4Linear: q/k/v/o projections, q/k head RMSNorm, half-split RoPE, GQA-4 causal
//! SDPA over the KV cache, output * sigmoid(gate), o_proj. The composition was
//! locked end-to-end against MLX's Qwen3NextAttention (residual = bf16 vs f32).
//! Runs one token at position `pos`; the caller owns the KV cache.
//! Exception: this module is temporarily above the 1,000-line guideline while
//! full-attention prefill, prefix-attention variants, and prepared-RHS parity
//! wiring remain colocated for Qwen3.5-9B validation.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const dims = @import("dims.zig");
const safetensors = @import("safetensors.zig");
const linear_q4 = @import("linear_q4.zig");
const mlx_nt = @import("mlx_nt.zig");
const weight_upload = @import("upload.zig");
const Q4Linear = linear_q4.Q4Linear;

// Qwen3.5-9B full-attention dims.
pub const NUM_Q = 16;
pub const NUM_KV = 4;
pub const HEAD_DIM = 256;
const ROT = 64;
const ROPE_BASE: f32 = 1e7;
const HIDDEN = dims.hidden;
const EPS = dims.rmsnorm_eps;
const SCALE: f32 = 1.0 / 16.0; // head_dim^-0.5 = 256^-0.5
const PREFILL_SCORE_THREADS = 128;
const PREFILL_SCORE_TILE_M = 32;
const PREFILL_SCORE_TILE_N = 32;
const PREFILL_VALUE_THREADS = 128;
const PREFILL_VALUE_TILE_M = 32;
const PREFILL_VALUE_TILE_N = 32;
const PREFILL_SOFTMAX_THREADS = 256;
const PREFILL_QK_PREPARE_THREADS = 256;
const DECODE_QK_PREPARE_THREADS = 256;
// Flash-decoding contract shared with shaders/attention_decode.metal.
const DECODE_FLASH_CHUNK_TOKENS: u32 = 1024;
const DECODE_FLASH_THREADS = 256;
const BF16_BYTES = dims.bf16_bytes;

const DecodePrepareParams = extern struct {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    position_offset: u32,
    theta: f32,
    cache_stride_tokens: u32,
    dst_token_index: u32,
    threads_per_threadgroup: u32,
};

pub const PrefillAttentionScratch = struct {
    score_probs: metal.Buffer,
    compact_value: metal.Buffer,
    group_output: metal.Buffer,
    element_bytes: usize,

    pub fn deinit(self: *PrefillAttentionScratch) void {
        self.group_output.destroy();
        self.compact_value.destroy();
        self.score_probs.destroy();
        self.* = undefined;
    }

    pub fn makeAliasableIfHeapBacked(self: *const PrefillAttentionScratch) void {
        self.score_probs.makeAliasableIfHeapBacked();
        self.compact_value.makeAliasableIfHeapBacked();
        self.group_output.makeAliasableIfHeapBacked();
    }
};

fn encodeAttentionScoreMlxNtBf16(
    ws: *metal.Workspace,
    p: Bf16Pipelines,
    compact_query: metal.Buffer,
    cache_k: metal.Buffer,
    score_probs: metal.Buffer,
    group_rows: u32,
    compact_query_offset: usize,
    cache_len: u32,
    cache_stride: u32,
    kv_head_index: u32,
) !void {
    if (group_rows == 0 or cache_len == 0) return error.EmptyInput;
    const query_bytes = try checked_math.product(.{ group_rows, HEAD_DIM, BF16_BYTES });
    const score_bytes = try checked_math.product(.{ group_rows, cache_len, BF16_BYTES });
    const cache_head_offset = try checked_math.product(.{ kv_head_index, cache_stride, HEAD_DIM, BF16_BYTES });
    const cache_head_bytes = try checked_math.product(.{ cache_len, HEAD_DIM, BF16_BYTES });
    if (compact_query_offset > compact_query.length or query_bytes > compact_query.length - compact_query_offset) return error.InputBufferTooSmall;
    if (score_probs.length < score_bytes) return error.OutputBufferTooSmall;
    if (cache_head_offset > cache_k.length or cache_head_bytes > cache_k.length - cache_head_offset) return error.InputBufferTooSmall;

    const plan = try metal.MlxGemm.planPrefillAttentionScoreNt(.bf16, group_rows, cache_len, HEAD_DIM);
    const params = try ws.valueBuf(metal.MlxGemm.Params, plan.params);
    try ws.cmd.dispatchThreadgroups3D(
        try p.mlx_nt.pipeline(plan.tile),
        &.{
            .{ .index = 0, .buffer = compact_query, .offset = compact_query_offset },
            .{ .index = 1, .buffer = cache_k, .offset = cache_head_offset },
            .{ .index = 3, .buffer = score_probs },
            .{ .index = 4, .buffer = params },
        },
        try metal.Grid3D.init(plan.threadgroups_per_grid.x, plan.threadgroups_per_grid.y, plan.threadgroups_per_grid.z),
        try metal.Grid3D.init(plan.threads_per_threadgroup.x, plan.threads_per_threadgroup.y, plan.threads_per_threadgroup.z),
    );
}

test "prepared full attention dense RHS byte sizes match Qwen projections" {
    try std.testing.expectEqual(@as(usize, 8192 * 4096 * BF16_BYTES), try preparedFullAttentionQDenseRhsBf16ByteLen());
    try std.testing.expectEqual(@as(usize, 1024 * 4096 * BF16_BYTES), try preparedFullAttentionKvDenseRhsBf16ByteLen());
    try std.testing.expectEqual(@as(usize, 4096 * 4096 * BF16_BYTES), try preparedFullAttentionOutProjDenseRhsBf16ByteLen());
}

fn cacheTokenCapacity(cache: metal.Buffer, element_bytes: usize) !u32 {
    const row_bytes = try checked_math.product(.{ NUM_KV, HEAD_DIM, element_bytes });
    if (row_bytes == 0 or cache.length % row_bytes != 0) return error.InvalidKvCacheTensorShape;
    return std.math.cast(u32, cache.length / row_bytes) orelse error.SequenceTooLong;
}

pub const Bf16Pipelines = struct {
    qmv: metal.Pipeline,
    qmm_m32n64_nt: metal.Pipeline,
    mlx_nt: mlx_nt.Pipelines,
    sdpa_flash_partials: metal.Pipeline,
    sdpa_flash_reduce: metal.Pipeline,
    qgate_prepare_kv_append: metal.Pipeline,
    qk_prepare_prefill: metal.Pipeline,
    split_qg_many: metal.Pipeline,
    prefill_score: metal.Pipeline,
    prefill_softmax: metal.Pipeline,
    prefill_softmax_scaled_masked: metal.Pipeline,
    prefill_query_compact: metal.Pipeline,
    prefill_value_compact: metal.Pipeline,
    prefill_value_scatter: metal.Pipeline,
    prefill_prefix_score: metal.Pipeline,
    prefill_prefix_value: metal.Pipeline,
    sigmoid_mul_pair: metal.Pipeline,
};

pub const PreparedFullAttentionSetupDenseRhs = struct {
    q: ?metal.Buffer = null,
    k: ?metal.Buffer = null,
    v: ?metal.Buffer = null,
};

pub const PreparedFullAttentionDenseRhs = struct {
    setup: PreparedFullAttentionSetupDenseRhs = .{},
    out_proj: ?metal.Buffer = null,
};

pub const PreparedFullAttentionDenseRhsBuffers = struct {
    pub const InitOptions = struct {
        include_setup: bool = false,
        include_out_proj: bool = false,
    };

    q: ?metal.Buffer = null,
    k: ?metal.Buffer = null,
    v: ?metal.Buffer = null,
    out_proj: ?metal.Buffer = null,

    pub fn init(device: *metal.Device, options: InitOptions) !PreparedFullAttentionDenseRhsBuffers {
        var result: PreparedFullAttentionDenseRhsBuffers = .{};
        errdefer result.deinit();

        if (options.include_setup) {
            result.q = try device.createPrivateBuffer(try preparedFullAttentionQDenseRhsBf16ByteLen());
            result.k = try device.createPrivateBuffer(try preparedFullAttentionKvDenseRhsBf16ByteLen());
            result.v = try device.createPrivateBuffer(try preparedFullAttentionKvDenseRhsBf16ByteLen());
        }
        if (options.include_out_proj) {
            result.out_proj = try device.createPrivateBuffer(try preparedFullAttentionOutProjDenseRhsBf16ByteLen());
        }

        return result;
    }

    pub fn deinit(self: *PreparedFullAttentionDenseRhsBuffers) void {
        if (self.out_proj) |*out_proj| out_proj.destroy();
        if (self.v) |*v| v.destroy();
        if (self.k) |*k| k.destroy();
        if (self.q) |*q| q.destroy();
        self.* = undefined;
    }

    pub fn prepared(self: *const PreparedFullAttentionDenseRhsBuffers) PreparedFullAttentionDenseRhs {
        return .{
            .setup = .{ .q = self.q, .k = self.k, .v = self.v },
            .out_proj = self.out_proj,
        };
    }

    pub fn encodeBf16Command(
        self: *const PreparedFullAttentionDenseRhsBuffers,
        cmd: *metal.CommandBuffer,
        dequant_bf16: metal.Pipeline,
        block: *const FullAttentionBlock,
    ) !void {
        if (self.q) |q| try block.q_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, q);
        if (self.k) |k| try block.k_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, k);
        if (self.v) |v| try block.v_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, v);
        if (self.out_proj) |out_proj| try block.o_proj.encodeDequantizeBf16Command(cmd, dequant_bf16, out_proj);
    }
};

fn preparedFullAttentionQDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(NUM_Q * 2 * HEAD_DIM, HIDDEN);
}

fn preparedFullAttentionKvDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(NUM_KV * HEAD_DIM, HIDDEN);
}

fn preparedFullAttentionOutProjDenseRhsBf16ByteLen() !usize {
    return linear_q4.denseRhsBf16ByteLen(HIDDEN, NUM_Q * HEAD_DIM);
}

pub const PrefixCache = struct {
    cache_k: metal.Buffer,
    cache_v: metal.Buffer,
    len: u32,
};

pub const FullAttentionBlock = struct {
    q_proj: Q4Linear,
    k_proj: Q4Linear,
    v_proj: Q4Linear,
    o_proj: Q4Linear,
    q_norm: metal.Buffer,
    k_norm: metal.Buffer,

    pub fn upload(device: *metal.Device, queue: *metal.Queue, repo: *const safetensors.Repository, prefix: []const u8) !FullAttentionBlock {
        var key: [256]u8 = undefined;
        var q_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.q_proj", .{prefix}));
        errdefer q_proj.deinit();
        var k_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.k_proj", .{prefix}));
        errdefer k_proj.deinit();
        var v_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.v_proj", .{prefix}));
        errdefer v_proj.deinit();
        var o_proj = try Q4Linear.upload(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.o_proj", .{prefix}));
        errdefer o_proj.deinit();
        var q_norm = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.q_norm.weight", .{prefix}));
        errdefer q_norm.destroy();
        const k_norm = try weight_upload.namedTensorPrivate(device, queue, repo, try std.fmt.bufPrint(&key, "{s}.k_norm.weight", .{prefix}));
        return .{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj, .o_proj = o_proj, .q_norm = q_norm, .k_norm = k_norm };
    }

    pub fn deinit(self: *FullAttentionBlock) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.q_norm.destroy();
        self.k_norm.destroy();
        self.* = undefined;
    }

    pub fn decodeStepBf16(
        self: *const FullAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        x_bf16: metal.Buffer,
        cache_k_bf16: metal.Buffer,
        cache_v_bf16: metal.Buffer,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        prefix: ?PrefixCache,
        out_bf16: metal.Buffer,
    ) !void {
        const cap = try cacheTokenCapacity(cache_k_bf16, BF16_BYTES);
        if (try cacheTokenCapacity(cache_v_bf16, BF16_BYTES) != cap) return error.InvalidKvCacheTensorShape;
        if (cache_pos >= cap) return error.SequenceTooLong;

        if (prefix) |pref| {
            const prefix_cap = try cacheTokenCapacity(pref.cache_k, BF16_BYTES);
            if (try cacheTokenCapacity(pref.cache_v, BF16_BYTES) != prefix_cap) return error.InvalidKvCacheTensorShape;
            if (pref.len > prefix_cap or seq_len < pref.len) return error.SequenceTooLong;
            if (seq_len - pref.len > cap) return error.SequenceTooLong;
        } else if (seq_len > cap) return error.SequenceTooLong;

        const hidden_bytes = try checked_math.bytes(HIDDEN, BF16_BYTES);
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const hd_buf = try ws.u32buf(HEAD_DIM);
        const nq_buf = try ws.u32buf(NUM_Q);
        const nkv_buf = try ws.u32buf(NUM_KV);
        const sl_buf = try ws.u32buf(seq_len);
        const scale_buf = try ws.f32buf(SCALE);
        const cap_buf = try ws.u32buf(cap);
        const prepare_params = try ws.valueBuf(DecodePrepareParams, .{
            .num_heads = NUM_Q,
            .num_kv_heads = NUM_KV,
            .head_dim = HEAD_DIM,
            .rotary_dim = ROT,
            .position_offset = rope_pos,
            .theta = ROPE_BASE,
            .cache_stride_tokens = cap,
            .dst_token_index = cache_pos,
            .threads_per_threadgroup = DECODE_QK_PREPARE_THREADS,
        });

        const q_values = NUM_Q * HEAD_DIM;
        const q_bytes = try checked_math.bytes(q_values, BF16_BYTES);
        const qp = try ws.scratch(try checked_math.product(.{ NUM_Q, 2, HEAD_DIM, BF16_BYTES }));
        try self.q_proj.encodeBf16Fast(ws, p.qmv, x_bf16, qp, 1);
        const queries = try ws.scratch(q_bytes);
        const gate = try ws.scratch(q_bytes);

        const kv_bytes = try checked_math.product(.{ NUM_KV, HEAD_DIM, BF16_BYTES });
        const kbuf = try ws.scratch(kv_bytes);
        try self.k_proj.encodeBf16Fast(ws, p.qmv, x_bf16, kbuf, 1);
        const vbuf = try ws.scratch(kv_bytes);
        try self.v_proj.encodeBf16Fast(ws, p.qmv, x_bf16, vbuf, 1);
        // q/k/v projections all read only x_bf16 and run concurrently.
        ws.barrier();

        try ws.cmd.dispatch1DWithThreadgroup(
            p.qgate_prepare_kv_append,
            &.{ qp, self.q_norm, queries, gate, kbuf, vbuf, self.k_norm, cache_k_bf16, cache_v_bf16, prepare_params },
            (NUM_Q + NUM_KV) * DECODE_QK_PREPARE_THREADS,
            DECODE_QK_PREPARE_THREADS,
        );
        ws.barrier();

        const context = try ws.scratch(q_bytes);
        // Flash-decoding pair: split-context partials over (kv head, chunk)
        // threadgroups, then a per-head softmax merge. One route for every
        // context length and both cache shapes; the non-prefix path binds the
        // local cache as the prefix buffers with prefix_len=0.
        std.debug.assert(seq_len != 0);
        const chunk_count = std.math.divCeil(u32, seq_len, DECODE_FLASH_CHUNK_TOKENS) catch return error.SequenceTooLong;
        const partials_values = std.math.mul(usize, @as(usize, NUM_Q) * (HEAD_DIM + 2), chunk_count) catch return error.ContextSizeOverflow;
        const partials = try ws.scratch(try checked_math.bytes(partials_values, @sizeOf(f32)));
        const chunk_count_buf = try ws.u32buf(chunk_count);
        const prefix_k = if (prefix) |pref| pref.cache_k else cache_k_bf16;
        const prefix_v = if (prefix) |pref| pref.cache_v else cache_v_bf16;
        const prefix_len_buf = try ws.u32buf(if (prefix) |pref| pref.len else 0);
        const prefix_stride_buf = if (prefix) |pref| try ws.u32buf(try cacheTokenCapacity(pref.cache_k, BF16_BYTES)) else cap_buf;
        try ws.cmd.dispatch1DWithThreadgroup(
            p.sdpa_flash_partials,
            &.{ queries, prefix_k, prefix_v, cache_k_bf16, cache_v_bf16, partials, nq_buf, nkv_buf, hd_buf, sl_buf, prefix_len_buf, scale_buf, prefix_stride_buf, cap_buf, chunk_count_buf },
            @as(usize, NUM_KV) * chunk_count * DECODE_FLASH_THREADS,
            DECODE_FLASH_THREADS,
        );
        ws.barrier();
        // Fused merge + attention output gate in one dispatch.
        try ws.cmd.dispatch1D(
            p.sdpa_flash_reduce,
            &.{ partials, gate, context, nq_buf, hd_buf, chunk_count_buf },
            NUM_Q * HEAD_DIM,
        );
        ws.barrier();
        try self.o_proj.encodeBf16Fast(ws, p.qmv, context, out_bf16, 1);
    }

    pub fn encodePrefillWithPreparedDenseRhsBf16(
        self: *const FullAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        x_bf16: metal.Buffer,
        cache_k_bf16: metal.Buffer,
        cache_v_bf16: metal.Buffer,
        start_cache_pos: u32,
        start_rope_pos: u32,
        token_count: u32,
        prefix: ?PrefixCache,
        prefill_scratch: ?*const PrefillAttentionScratch,
        prepared: PreparedFullAttentionDenseRhs,
        out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const cap = try cacheTokenCapacity(cache_k_bf16, BF16_BYTES);
        if (try cacheTokenCapacity(cache_v_bf16, BF16_BYTES) != cap) return error.InvalidKvCacheTensorShape;
        const end_cache_pos = std.math.add(u32, start_cache_pos, token_count) catch return error.SequenceTooLong;
        if (end_cache_pos > cap) return error.SequenceTooLong;

        const hidden_bytes = try checked_math.product(.{ token_count, HIDDEN, BF16_BYTES });
        if (x_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (out_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const hd_buf = try ws.u32buf(HEAD_DIM);
        const nq_buf = try ws.u32buf(NUM_Q);
        const start_cache_buf = try ws.u32buf(start_cache_pos);
        const start_rope_buf = try ws.u32buf(start_rope_pos);
        const tokens_buf = try ws.u32buf(token_count);
        const cap_buf = try ws.u32buf(cap);

        const qp_values = std.math.mul(usize, @as(usize, token_count), NUM_Q * 2 * HEAD_DIM) catch return error.ContextSizeOverflow;
        const qp = try ws.scratch(try checked_math.bytes(qp_values, BF16_BYTES));
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.q_proj,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_bf16,
            .prepared_rhs = prepared.setup.q,
            .output = qp,
            .token_count = token_count,
            .out_dim = NUM_Q * 2 * HEAD_DIM,
            .in_dim = HIDDEN,
        });

        const q_values = std.math.mul(usize, @as(usize, token_count), NUM_Q * HEAD_DIM) catch return error.ContextSizeOverflow;
        const queries = try ws.scratch(try checked_math.bytes(q_values, BF16_BYTES));
        const gate = try ws.scratch(try checked_math.bytes(q_values, BF16_BYTES));
        try ws.cmd.dispatch1D(p.split_qg_many, &.{ qp, queries, gate, hd_buf, nq_buf, tokens_buf }, q_values);

        const kv_values = std.math.mul(usize, @as(usize, token_count), NUM_KV * HEAD_DIM) catch return error.ContextSizeOverflow;
        const kbuf = try ws.scratch(try checked_math.bytes(kv_values, BF16_BYTES));
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.k_proj,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_bf16,
            .prepared_rhs = prepared.setup.k,
            .output = kbuf,
            .token_count = token_count,
            .out_dim = NUM_KV * HEAD_DIM,
            .in_dim = HIDDEN,
        });
        const vbuf = try ws.scratch(try checked_math.bytes(kv_values, BF16_BYTES));
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.v_proj,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = x_bf16,
            .prepared_rhs = prepared.setup.v,
            .output = vbuf,
            .token_count = token_count,
            .out_dim = NUM_KV * HEAD_DIM,
            .in_dim = HIDDEN,
        });

        const prepare_rows = std.math.mul(usize, @as(usize, token_count), NUM_Q + NUM_KV) catch return error.ContextSizeOverflow;
        try ws.cmd.dispatch1DWithThreadgroup(
            p.qk_prepare_prefill,
            &.{ queries, self.q_norm, kbuf, self.k_norm, vbuf, cache_k_bf16, cache_v_bf16, start_rope_buf, tokens_buf, start_cache_buf, cap_buf },
            prepare_rows * PREFILL_QK_PREPARE_THREADS,
            PREFILL_QK_PREPARE_THREADS,
        );

        const context = try ws.scratch(try checked_math.bytes(q_values, BF16_BYTES));
        try encodePrefillAttentionBf16(ws, p, queries, cache_k_bf16, cache_v_bf16, context, token_count, start_cache_pos, cap, prefix, prefill_scratch);
        try self.encodePrefillOutputWithPreparedDenseRhsBf16(ws, p, context, gate, token_count, prepared.out_proj, out_bf16);
    }

    fn encodePrefillAttentionBf16(
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        queries: metal.Buffer,
        cache_k: metal.Buffer,
        cache_v: metal.Buffer,
        attn: metal.Buffer,
        token_count: u32,
        start_cache_pos: u32,
        cache_stride: u32,
        prefix: ?PrefixCache,
        prefill_scratch: ?*const PrefillAttentionScratch,
    ) !void {
        if (prefix) |pref| {
            const prefix_cap = try cacheTokenCapacity(pref.cache_k, BF16_BYTES);
            if (try cacheTokenCapacity(pref.cache_v, BF16_BYTES) != prefix_cap) return error.InvalidKvCacheTensorShape;
            try encodeMaterializedPrefixPrefillBf16(ws, p, queries, pref.cache_k, pref.cache_v, cache_k, cache_v, attn, token_count, pref.len, start_cache_pos, prefix_cap, cache_stride);
        } else {
            try encodeMaterializedPrefillBf16(ws, p, queries, cache_k, cache_v, attn, token_count, start_cache_pos, cache_stride, prefill_scratch);
        }
    }

    fn encodePrefillOutputWithPreparedDenseRhsBf16(
        self: *const FullAttentionBlock,
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        context_bf16: metal.Buffer,
        gate_bf16: metal.Buffer,
        token_count: u32,
        prepared_out_proj: ?metal.Buffer,
        out_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const values = std.math.mul(usize, @as(usize, token_count), HIDDEN) catch return error.ContextSizeOverflow;
        const byte_len = std.math.mul(usize, values, BF16_BYTES) catch return error.ContextSizeOverflow;
        if (context_bf16.length < byte_len or gate_bf16.length < byte_len) return error.InputBufferTooSmall;
        if (out_bf16.length < byte_len) return error.OutputBufferTooSmall;

        const values_u32 = std.math.cast(u32, values) orelse return error.ContextSizeOverflow;
        const values_buf = try ws.u32buf(values_u32);
        try ws.cmd.dispatch1D(p.sigmoid_mul_pair, &.{ context_bf16, gate_bf16, values_buf }, values);
        try mlx_nt.encodePreparedQ4ProjectionBf16(ws, p.mlx_nt, .{
            .projection = &self.o_proj,
            .qmm_pipeline = p.qmm_m32n64_nt,
            .input = context_bf16,
            .prepared_rhs = prepared_out_proj,
            .output = out_bf16,
            .token_count = token_count,
            .out_dim = HIDDEN,
            .in_dim = NUM_Q * HEAD_DIM,
        });
    }
    fn encodeMaterializedPrefillBf16(
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        queries: metal.Buffer,
        cache_k: metal.Buffer,
        cache_v: metal.Buffer,
        attn: metal.Buffer,
        token_count: u32,
        cache_prefix_len: u32,
        cache_stride: u32,
        prefill_scratch: ?*const PrefillAttentionScratch,
    ) !void {
        const group_count = NUM_Q / NUM_KV;
        const group_rows_u32 = std.math.mul(u32, token_count, group_count) catch return error.ContextSizeOverflow;
        const cache_len_u32 = std.math.add(u32, cache_prefix_len, token_count) catch return error.ContextSizeOverflow;
        const score_values = std.math.mul(usize, @as(usize, group_rows_u32), cache_len_u32) catch return error.ContextSizeOverflow;
        const score_bytes = try checked_math.bytes(score_values, BF16_BYTES);
        const score_probs = if (prefill_scratch) |scratch| blk: {
            if (scratch.element_bytes != BF16_BYTES) return error.UnsupportedScratchDType;
            if (scratch.score_probs.length < score_bytes) return error.InvalidPrefillScratch;
            break :blk scratch.score_probs;
        } else try ws.scratch(score_bytes);

        const group_rows_buf = try ws.u32buf(group_rows_u32);
        const cache_len_buf = try ws.u32buf(cache_len_u32);
        const hd_buf = try ws.u32buf(HEAD_DIM);
        const tokens_buf = try ws.u32buf(token_count);
        const nq_buf = try ws.u32buf(NUM_Q);
        const prefix_buf = try ws.u32buf(cache_prefix_len);
        const scale_buf = try ws.f32buf(SCALE);

        const softmax_grid = std.math.mul(usize, @as(usize, group_rows_u32), PREFILL_SOFTMAX_THREADS) catch return error.ContextSizeOverflow;
        const query_compact_values = std.math.mul(usize, @as(usize, token_count), NUM_Q * HEAD_DIM) catch return error.ContextSizeOverflow;
        const query_compact = try ws.scratch(try checked_math.bytes(query_compact_values, BF16_BYTES));
        const all_heads_buf = try ws.u32buf(NUM_Q);
        const zero_buf = try ws.u32buf(0);
        try ws.cmd.dispatch1D(
            p.prefill_query_compact,
            &.{ queries, query_compact, tokens_buf, nq_buf, zero_buf, all_heads_buf, hd_buf },
            query_compact_values,
        );

        for (0..NUM_KV) |kvh| {
            const kvh_u32: u32 = @intCast(kvh);
            const head_start: u32 = @intCast(kvh * group_count);
            const query_offset = try checked_math.product(.{ head_start, token_count, HEAD_DIM, BF16_BYTES });
            try encodeAttentionScoreMlxNtBf16(
                ws,
                p,
                query_compact,
                cache_k,
                score_probs,
                group_rows_u32,
                query_offset,
                cache_len_u32,
                cache_stride,
                kvh_u32,
            );
            try ws.cmd.dispatch1DWithThreadgroup(
                p.prefill_softmax_scaled_masked,
                &.{ score_probs, score_probs, group_rows_buf, cache_len_buf, tokens_buf, prefix_buf, scale_buf },
                softmax_grid,
                PREFILL_SOFTMAX_THREADS,
            );
            try encodeValueMlxNtScatterBf16(ws, p, score_probs, cache_v, attn, group_rows_u32, cache_len_u32, token_count, head_start, kvh_u32, cache_stride, prefill_scratch);
        }
    }

    fn encodeValueMlxNtScatterBf16(
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        score_probs: metal.Buffer,
        cache_v: metal.Buffer,
        attn: metal.Buffer,
        group_rows: u32,
        cache_len: u32,
        token_count: u32,
        head_start: u32,
        kv_head_index: u32,
        cache_stride: u32,
        prefill_scratch: ?*const PrefillAttentionScratch,
    ) !void {
        if (group_rows == 0 or cache_len == 0 or token_count == 0) return error.EmptyInput;
        const compact_values = std.math.mul(usize, @as(usize, cache_len), HEAD_DIM) catch return error.ContextSizeOverflow;
        const compact_bytes = try checked_math.bytes(compact_values, BF16_BYTES);
        const compact = if (prefill_scratch) |scratch| blk: {
            if (scratch.element_bytes != BF16_BYTES) return error.UnsupportedScratchDType;
            if (scratch.compact_value.length < compact_bytes) return error.InvalidPrefillScratch;
            break :blk scratch.compact_value;
        } else try ws.scratch(compact_bytes);
        const group_output_values = std.math.mul(usize, @as(usize, group_rows), HEAD_DIM) catch return error.ContextSizeOverflow;
        const group_output_bytes = try checked_math.bytes(group_output_values, BF16_BYTES);
        const group_output = if (prefill_scratch) |scratch| blk: {
            if (scratch.element_bytes != BF16_BYTES) return error.UnsupportedScratchDType;
            if (scratch.group_output.length < group_output_bytes) return error.InvalidPrefillScratch;
            break :blk scratch.group_output;
        } else try ws.scratch(group_output_bytes);

        const cache_len_buf = try ws.u32buf(cache_len);
        const cache_stride_buf = try ws.u32buf(cache_stride);
        const kv_head_buf = try ws.u32buf(kv_head_index);
        const head_dim_buf = try ws.u32buf(HEAD_DIM);
        try ws.cmd.dispatch1D(
            p.prefill_value_compact,
            &.{ cache_v, compact, cache_len_buf, cache_stride_buf, kv_head_buf, head_dim_buf },
            compact_values,
        );

        const plan = try metal.MlxGemm.planPrefillAttentionValueNt(.bf16, group_rows, HEAD_DIM, cache_len);
        const params = try ws.valueBuf(metal.MlxGemm.Params, plan.params);
        try ws.cmd.dispatchThreadgroups3D(
            try p.mlx_nt.pipeline(plan.tile),
            &.{
                .{ .index = 0, .buffer = score_probs },
                .{ .index = 1, .buffer = compact },
                .{ .index = 3, .buffer = group_output },
                .{ .index = 4, .buffer = params },
            },
            try metal.Grid3D.init(plan.threadgroups_per_grid.x, plan.threadgroups_per_grid.y, plan.threadgroups_per_grid.z),
            try metal.Grid3D.init(plan.threads_per_threadgroup.x, plan.threads_per_threadgroup.y, plan.threads_per_threadgroup.z),
        );

        const token_count_buf = try ws.u32buf(token_count);
        const head_count_buf = try ws.u32buf(NUM_Q);
        const head_start_buf = try ws.u32buf(head_start);
        const group_count_buf = try ws.u32buf(NUM_Q / NUM_KV);
        try ws.cmd.dispatch1D(
            p.prefill_value_scatter,
            &.{ group_output, attn, token_count_buf, head_count_buf, head_start_buf, group_count_buf, head_dim_buf },
            group_output_values,
        );
    }

    fn encodeMaterializedPrefixPrefillBf16(
        ws: *metal.Workspace,
        p: Bf16Pipelines,
        queries: metal.Buffer,
        prefix_cache_k: metal.Buffer,
        prefix_cache_v: metal.Buffer,
        local_cache_k: metal.Buffer,
        local_cache_v: metal.Buffer,
        attn: metal.Buffer,
        token_count: u32,
        prefix_len: u32,
        local_prefix_len: u32,
        prefix_stride: u32,
        local_stride: u32,
    ) !void {
        const group_count = NUM_Q / NUM_KV;
        const group_rows_u32 = std.math.mul(u32, token_count, group_count) catch return error.ContextSizeOverflow;
        const local_cache_len = std.math.add(u32, local_prefix_len, token_count) catch return error.ContextSizeOverflow;
        const cache_len_u32 = std.math.add(u32, prefix_len, local_cache_len) catch return error.ContextSizeOverflow;
        const score_values = std.math.mul(usize, @as(usize, group_rows_u32), cache_len_u32) catch return error.ContextSizeOverflow;
        const score_probs = try ws.scratch(try checked_math.bytes(score_values, BF16_BYTES));

        const group_rows_buf = try ws.u32buf(group_rows_u32);
        const cache_len_buf = try ws.u32buf(cache_len_u32);
        const hd_buf = try ws.u32buf(HEAD_DIM);
        const tokens_buf = try ws.u32buf(token_count);
        const nq_buf = try ws.u32buf(NUM_Q);
        const nkv_buf = try ws.u32buf(NUM_KV);
        const prefix_len_buf = try ws.u32buf(prefix_len);
        const scale_buf = try ws.f32buf(SCALE);
        const prefix_stride_buf = try ws.u32buf(prefix_stride);
        const local_stride_buf = try ws.u32buf(local_stride);

        const score_tiles_m = std.math.divCeil(usize, group_rows_u32, PREFILL_SCORE_TILE_M) catch return error.ContextSizeOverflow;
        const score_tiles_n = std.math.divCeil(usize, cache_len_u32, PREFILL_SCORE_TILE_N) catch return error.ContextSizeOverflow;
        const score_groups = std.math.mul(usize, score_tiles_m, score_tiles_n) catch return error.ContextSizeOverflow;
        const score_grid = std.math.mul(usize, score_groups, PREFILL_SCORE_THREADS) catch return error.ContextSizeOverflow;

        const softmax_grid = std.math.mul(usize, @as(usize, group_rows_u32), PREFILL_SOFTMAX_THREADS) catch return error.ContextSizeOverflow;

        const value_tiles_m = std.math.divCeil(usize, group_rows_u32, PREFILL_VALUE_TILE_M) catch return error.ContextSizeOverflow;
        const value_tiles_n = std.math.divCeil(usize, HEAD_DIM, PREFILL_VALUE_TILE_N) catch return error.ContextSizeOverflow;
        const value_groups = std.math.mul(usize, value_tiles_m, value_tiles_n) catch return error.ContextSizeOverflow;
        const value_grid = std.math.mul(usize, value_groups, PREFILL_VALUE_THREADS) catch return error.ContextSizeOverflow;

        for (0..NUM_KV) |kvh| {
            const kvh_u32: u32 = @intCast(kvh);
            const head_start: u32 = @intCast(kvh * group_count);
            const kvh_buf = try ws.u32buf(kvh_u32);
            const head_start_buf = try ws.u32buf(head_start);

            try ws.cmd.dispatch1DWithThreadgroup(
                p.prefill_prefix_score,
                &.{ queries, prefix_cache_k, local_cache_k, score_probs, group_rows_buf, cache_len_buf, hd_buf, tokens_buf, nq_buf, nkv_buf, head_start_buf, kvh_buf, prefix_len_buf, scale_buf, prefix_stride_buf, local_stride_buf },
                score_grid,
                PREFILL_SCORE_THREADS,
            );
            try ws.cmd.dispatch1DWithThreadgroup(
                p.prefill_softmax,
                &.{ score_probs, score_probs, group_rows_buf, cache_len_buf },
                softmax_grid,
                PREFILL_SOFTMAX_THREADS,
            );
            try ws.cmd.dispatch1DWithThreadgroup(
                p.prefill_prefix_value,
                &.{ score_probs, prefix_cache_v, local_cache_v, attn, group_rows_buf, hd_buf, cache_len_buf, tokens_buf, nq_buf, nkv_buf, head_start_buf, kvh_buf, prefix_len_buf, prefix_stride_buf, local_stride_buf },
                value_grid,
                PREFILL_VALUE_THREADS,
            );
        }
    }
};

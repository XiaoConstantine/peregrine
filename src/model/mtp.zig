//! Experimental Qwen3.5 MTP drafter support.
//!
//! The MLX MTP sidecar is not a standalone model: it owns the pre-fc norms,
//! fc projection, one full-attention decoder layer, and final norm. Embeddings
//! and lm_head are shared with the target Qwen3.5 model.

const std = @import("std");
const metal = @import("../runtime/metal.zig");
const checked_math = @import("../runtime/checked_math.zig");
const block_attn = @import("block_attn.zig");
const decoder_layer = @import("decoder_layer.zig");
const dims = @import("dims.zig");
const linear_q4 = @import("linear_q4.zig");
const model_mod = @import("model.zig");
const safetensors = @import("safetensors.zig");
const state_mod = @import("state.zig");
const weight_upload = @import("upload.zig");

const DeviceModel = model_mod.DeviceModel;
const HIDDEN = dims.hidden;
const BF16_BYTES = dims.bf16_bytes;
const BF16_RMSNORM_THREADS = dims.bf16_rmsnorm_threads;
const EPS = dims.rmsnorm_eps;
const Q4Linear = linear_q4.Q4Linear;

const RootCandidate = struct {
    prefix: []const u8,
    fc_suffix: []const u8 = "fc.weight",
};

const root_candidates = [_]RootCandidate{
    .{ .prefix = "" },
    .{ .prefix = "mtp." },
    .{ .prefix = "language_model.mtp." },
    .{ .prefix = "model.mtp." },
};

pub const State = struct {
    layer: decoder_layer.LayerState,
    max_seq: usize,
    full_cache_stride: usize,

    pub fn initBf16(device: *metal.Device, max_seq: usize) !State {
        if (max_seq == 0 or max_seq > state_mod.MAX_CONTEXT_TOKENS) return error.InvalidContextLength;
        const full_cache_stride = try state_mod.fullCacheStrideForMaxSeq(max_seq);
        const full_cache_bytes = try checked_math.product(.{
            block_attn.NUM_KV,
            full_cache_stride,
            block_attn.HEAD_DIM,
            BF16_BYTES,
        });
        var cache_k = try device.createSharedBuffer(full_cache_bytes);
        errdefer cache_k.destroy();
        @memset(cache_k.slice(u8), 0);
        var cache_v = try device.createSharedBuffer(full_cache_bytes);
        errdefer cache_v.destroy();
        @memset(cache_v.slice(u8), 0);
        return .{
            .layer = .{ .full = .{ .cache_k = cache_k, .cache_v = cache_v } },
            .max_seq = max_seq,
            .full_cache_stride = full_cache_stride,
        };
    }

    pub fn deinit(self: *State) void {
        self.layer.full.cache_k.destroy();
        self.layer.full.cache_v.destroy();
        self.* = undefined;
    }
};

pub const Drafter = struct {
    fc: Q4Linear,
    layer: decoder_layer.DecoderLayer,
    norm: metal.Buffer,
    pre_fc_norm_embedding: metal.Buffer,
    pre_fc_norm_hidden: metal.Buffer,
    rmsnorm_bf16: metal.Pipeline,
    qmv_bf16: metal.Pipeline,
    qmm_m32n64_nt_bf16: metal.Pipeline,
    concat_two_rows_bf16: metal.Pipeline,
    copy_row_bf16: metal.Pipeline,

    pub fn upload(device: *metal.Device, queue: *metal.Queue, library: metal.Library, repo: *const safetensors.Repository) !Drafter {
        const root = try detectRoot(repo);
        var key: [256]u8 = undefined;

        var fc = try Q4Linear.upload(device, queue, repo, try rooted(&key, root, "fc"));
        errdefer fc.deinit();
        var layer = try decoder_layer.DecoderLayer.uploadFullAttention(device, queue, repo, try rooted(&key, root, "layers.0"));
        errdefer layer.deinit();
        var norm = try weight_upload.namedTensorPrivate(device, queue, repo, try rooted(&key, root, "norm.weight"));
        errdefer norm.destroy();
        var pre_fc_norm_embedding = try weight_upload.namedTensorPrivate(device, queue, repo, try rooted(&key, root, "pre_fc_norm_embedding.weight"));
        errdefer pre_fc_norm_embedding.destroy();
        var pre_fc_norm_hidden = try weight_upload.namedTensorPrivate(device, queue, repo, try rooted(&key, root, "pre_fc_norm_hidden.weight"));
        errdefer pre_fc_norm_hidden.destroy();

        var rmsnorm_bf16 = try device.createPipeline(library, "rmsnorm_bf16");
        errdefer rmsnorm_bf16.destroy();
        var qmv_bf16 = try device.createPipeline(library, "linear_q4_affine_group64_qmv_fast_bf16");
        errdefer qmv_bf16.destroy();
        var qmm_m32n64_nt_bf16 = try device.createPipeline(library, "linear_q4_affine_group64_prefill_qmm_m32n64_nt_bf16_tiles_bf16");
        errdefer qmm_m32n64_nt_bf16.destroy();
        var concat_two_rows_bf16 = try device.createPipeline(library, "concat_two_rows_bf16");
        errdefer concat_two_rows_bf16.destroy();
        const copy_row_bf16 = try device.createPipeline(library, "copy_row_bf16");

        return .{
            .fc = fc,
            .layer = layer,
            .norm = norm,
            .pre_fc_norm_embedding = pre_fc_norm_embedding,
            .pre_fc_norm_hidden = pre_fc_norm_hidden,
            .rmsnorm_bf16 = rmsnorm_bf16,
            .qmv_bf16 = qmv_bf16,
            .qmm_m32n64_nt_bf16 = qmm_m32n64_nt_bf16,
            .concat_two_rows_bf16 = concat_two_rows_bf16,
            .copy_row_bf16 = copy_row_bf16,
        };
    }

    pub fn deinit(self: *Drafter) void {
        self.copy_row_bf16.destroy();
        self.concat_two_rows_bf16.destroy();
        self.qmm_m32n64_nt_bf16.destroy();
        self.qmv_bf16.destroy();
        self.rmsnorm_bf16.destroy();
        self.pre_fc_norm_hidden.destroy();
        self.pre_fc_norm_embedding.destroy();
        self.norm.destroy();
        self.layer.deinit();
        self.fc.deinit();
        self.* = undefined;
    }

    pub fn forwardPrefill(
        self: *Drafter,
        device: *metal.Device,
        queue: *metal.Queue,
        target: *DeviceModel,
        token_ids: []const u32,
        target_hidden_bf16: metal.Buffer,
        state: *State,
        start_cache_pos: u32,
        start_rope_pos: u32,
    ) !void {
        if (token_ids.len == 0) return;
        const token_count = std.math.cast(u32, token_ids.len) orelse return error.SequenceTooLong;
        const end_cache_pos = std.math.add(usize, @as(usize, start_cache_pos), token_ids.len) catch return error.SequenceTooLong;
        if (end_cache_pos > state.max_seq) return error.SequenceTooLong;
        if (target_hidden_bf16.length < try hiddenRowsBytes(token_count)) return error.InputBufferTooSmall;
        for (token_ids) |token_id| {
            if (token_id >= target.vocab) return error.InvalidTokenId;
        }

        const token_bytes = try checked_math.product(.{ token_ids.len, @sizeOf(u32) });
        var token_ids_buf = try device.createSharedBuffer(token_bytes);
        defer token_ids_buf.destroy();
        @memcpy(token_ids_buf.slice(u32), token_ids);

        var ws = try metal.Workspace.beginWithScratchPool(device, queue, &target.prefill_scratch_pool);
        var committed = false;
        errdefer if (!committed) ws.abort();

        const fused = try self.encodePreFcRows(&ws, target, token_ids_buf, token_count, target_hidden_bf16);
        const out = try ws.scratch(try hiddenRowsBytes(token_count));
        try self.layer.prefillStepBf16(
            &ws,
            target.pipes,
            fused,
            &state.layer,
            start_cache_pos,
            start_rope_pos,
            token_count,
            null,
            0,
            null,
            .{},
            .{},
            .{},
            out,
        );
        committed = true;
        try ws.commitAndWait();
    }

    pub fn forwardDraftTokenProfiled(
        self: *Drafter,
        device: *metal.Device,
        queue: *metal.Queue,
        target: *DeviceModel,
        token_id: u32,
        target_hidden_rows_bf16: metal.Buffer,
        target_hidden_row: u32,
        state: *State,
        cache_pos: u32,
        rope_pos: u32,
        seq_len: u32,
        output_token: metal.Buffer,
    ) !?u64 {
        if (cache_pos >= state.max_seq) return error.SequenceTooLong;
        if (token_id >= target.vocab) return error.InvalidTokenId;
        const hidden_row_count = std.math.add(u32, target_hidden_row, 1) catch return error.SequenceTooLong;
        if (target_hidden_rows_bf16.length < try hiddenRowsBytes(hidden_row_count)) return error.InputBufferTooSmall;

        var ws = try metal.Workspace.beginConcurrentWithScratchPool(device, queue, &target.scratch_pool);
        var committed = false;
        errdefer if (!committed) ws.abort();

        const token_id_buf = try ws.u32buf(token_id);
        const target_hidden = try ws.scratch(try hiddenRowsBytes(1));
        const target_row_buf = try ws.u32buf(target_hidden_row);
        try ws.cmd.dispatch1D(self.copy_row_bf16, &.{ target_hidden_rows_bf16, target_hidden, target_row_buf, target.hidden_buf }, HIDDEN);
        ws.barrier();

        const fused = try self.encodePreFcRows(&ws, target, token_id_buf, 1, target_hidden);
        ws.barrier();

        const layer_out = try ws.scratch(try hiddenRowsBytes(1));
        try self.layer.decodeStepBf16(
            &ws,
            target.pipes,
            fused,
            &state.layer,
            cache_pos,
            rope_pos,
            seq_len,
            null,
            0,
            layer_out,
        );
        ws.barrier();

        const normed = try ws.scratch(try hiddenRowsBytes(1));
        try self.encodeFinalNormRows(&ws, layer_out, 1, normed);
        ws.barrier();
        try target.encodeArgmaxFromNormalizedHiddenBf16(&ws, normed, output_token, true);
        var pending = ws.commitPooled();
        committed = true;
        return try pending.waitProfile();
    }

    fn encodePreFcRows(
        self: *Drafter,
        ws: *metal.Workspace,
        target: *DeviceModel,
        token_ids_buf: metal.Buffer,
        token_count: u32,
        target_hidden_bf16: metal.Buffer,
    ) !metal.Buffer {
        if (token_count == 0) return error.EmptyInput;
        if (target_hidden_bf16.length < try hiddenRowsBytes(token_count)) return error.InputBufferTooSmall;

        const embeds = try ws.scratch(try hiddenRowsBytes(token_count));
        try target.encodeEmbeddingBatchBf16(ws, token_ids_buf, token_count, embeds);
        const normed_embeds = try ws.scratch(try hiddenRowsBytes(token_count));
        const normed_hidden = try ws.scratch(try hiddenRowsBytes(token_count));
        try self.encodeRmsNormRows(ws, embeds, self.pre_fc_norm_embedding, token_count, normed_embeds);
        try self.encodeRmsNormRows(ws, target_hidden_bf16, self.pre_fc_norm_hidden, token_count, normed_hidden);
        ws.barrier();

        const fc_input = try ws.scratch(try checked_math.product(.{ token_count, HIDDEN * 2, BF16_BYTES }));
        const width_buf = try ws.u32buf(HIDDEN);
        const rows_buf = try ws.u32buf(token_count);
        try ws.cmd.dispatch1D(
            self.concat_two_rows_bf16,
            &.{ normed_embeds, normed_hidden, fc_input, width_buf, rows_buf },
            @as(usize, token_count) * @as(usize, HIDDEN),
        );
        ws.barrier();

        const fused = try ws.scratch(try hiddenRowsBytes(token_count));
        if (token_count == 1) {
            try self.fc.encodeBf16Fast(ws, self.qmv_bf16, fc_input, fused, 1);
        } else {
            try self.fc.encodePrefillQmmM32N64NtBf16(ws, self.qmm_m32n64_nt_bf16, fc_input, fused, token_count);
        }
        return fused;
    }

    fn encodeFinalNormRows(
        self: *Drafter,
        ws: *metal.Workspace,
        hidden_bf16: metal.Buffer,
        token_count: u32,
        output_bf16: metal.Buffer,
    ) !void {
        try self.encodeRmsNormRows(ws, hidden_bf16, self.norm, token_count, output_bf16);
    }

    fn encodeRmsNormRows(
        self: *Drafter,
        ws: *metal.Workspace,
        input_bf16: metal.Buffer,
        weight: metal.Buffer,
        token_count: u32,
        output_bf16: metal.Buffer,
    ) !void {
        if (token_count == 0) return error.EmptyInput;
        const hidden_bytes = try hiddenRowsBytes(token_count);
        if (input_bf16.length < hidden_bytes) return error.InputBufferTooSmall;
        if (output_bf16.length < hidden_bytes) return error.OutputBufferTooSmall;

        const hidden_buf = try ws.u32buf(HIDDEN);
        const rows_buf = try ws.u32buf(token_count);
        const eps_buf = try ws.f32buf(EPS);
        const threads_buf = try ws.u32buf(BF16_RMSNORM_THREADS);
        try ws.cmd.dispatch1DWithThreadgroup(
            self.rmsnorm_bf16,
            &.{ input_bf16, weight, output_bf16, hidden_buf, rows_buf, eps_buf, threads_buf },
            @as(usize, token_count) * @as(usize, BF16_RMSNORM_THREADS),
            BF16_RMSNORM_THREADS,
        );
    }
};

pub fn verifyFingerprint(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8) !void {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);

    const bytes = try dir.readFileAlloc(io, "config.json", gpa, .limited(4 * 1024 * 1024));
    defer gpa.free(bytes);

    var parsed = try std.json.parseFromSlice(RawMtpConfig, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const raw = parsed.value;
    const tc = raw.text_config;
    const q = if (raw.quantization.bits != 0) raw.quantization else raw.quantization_config;

    if (!std.mem.eql(u8, raw.model_type, "qwen3_5_mtp")) return error.UnsupportedArchitecture;
    if (!std.mem.eql(u8, tc.model_type, "qwen3_5_text")) return error.UnsupportedArchitecture;
    if (tc.mtp_num_hidden_layers != 1) return error.UnsupportedArchitecture;
    if (!std.mem.eql(u8, q.mode, "affine")) return error.UnsupportedQuantization;
    if (q.bits != 4 or q.group_size != 64) return error.UnsupportedQuantization;
    if (tc.hidden_size != HIDDEN or
        tc.intermediate_size != 12288 or
        tc.num_attention_heads != block_attn.NUM_Q or
        tc.num_key_value_heads != block_attn.NUM_KV or
        tc.head_dim != block_attn.HEAD_DIM or
        tc.vocab_size != 248320)
    {
        return error.NotQwen35_9B;
    }
}

fn detectRoot(repo: *const safetensors.Repository) ![]const u8 {
    var key: [256]u8 = undefined;
    for (root_candidates) |candidate| {
        if (repo.get(try rooted(&key, candidate.prefix, candidate.fc_suffix)) != null) {
            return candidate.prefix;
        }
    }
    return error.TensorNotFound;
}

fn rooted(buf: []u8, root: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ root, suffix });
}

const hiddenRowsBytes = dims.hiddenRowsBytes;

const QuantizationConfig = struct {
    group_size: u32 = 0,
    bits: u32 = 0,
    mode: []const u8 = "",
};

const RawMtpConfig = struct {
    model_type: []const u8 = "",
    quantization: QuantizationConfig = .{},
    quantization_config: QuantizationConfig = .{},
    text_config: struct {
        model_type: []const u8 = "",
        hidden_size: u32 = 0,
        intermediate_size: u32 = 0,
        mtp_num_hidden_layers: u32 = 0,
        num_attention_heads: u32 = 0,
        num_key_value_heads: u32 = 0,
        head_dim: u32 = 0,
        vocab_size: u32 = 0,
    } = .{},
};

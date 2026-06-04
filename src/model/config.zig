//! Qwen3.5-9B text-backbone config fingerprint. Peregrine runs one model, so
//! wrong checkpoints are rejected before weight upload.

const std = @import("std");

const RawConfig = struct {
    text_config: struct {
        model_type: []const u8 = "",
        hidden_size: u32,
        intermediate_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        head_dim: u32,
        full_attention_interval: u32,
        linear_num_key_heads: u32,
        linear_num_value_heads: u32,
        linear_key_head_dim: u32,
        linear_value_head_dim: u32,
        linear_conv_kernel_dim: u32,
        vocab_size: u32,
        rope_parameters: struct {
            partial_rotary_factor: f64 = 0.25,
        } = .{},
    },
    quantization: struct {
        group_size: u32,
        bits: u32,
        mode: []const u8,
    },
};

/// Load-time guard for the exact Qwen3.5-9B MLX q4 checkpoint.
pub fn verifyFingerprint(gpa: std.mem.Allocator, io: std.Io, dir_path: []const u8) !void {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);

    const bytes = try dir.readFileAlloc(io, "config.json", gpa, .limited(4 * 1024 * 1024));
    defer gpa.free(bytes);

    var parsed = try std.json.parseFromSlice(RawConfig, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const raw = parsed.value;
    const tc = raw.text_config;

    if (!std.mem.eql(u8, tc.model_type, "qwen3_5_text")) return error.UnsupportedArchitecture;
    if (!std.mem.eql(u8, raw.quantization.mode, "affine")) return error.UnsupportedQuantization;
    if (raw.quantization.bits != 4 or raw.quantization.group_size != 64) return error.UnsupportedQuantization;
    if (tc.num_attention_heads == 0 or tc.num_key_value_heads == 0) return error.InvalidConfig;
    if (tc.num_attention_heads % tc.num_key_value_heads != 0) return error.InvalidConfig;
    if (tc.full_attention_interval == 0) return error.InvalidConfig;
    if (tc.num_hidden_layers == 0) return error.InvalidConfig;
    if (tc.head_dim == 0) return error.InvalidConfig;

    const rotary_dim: u32 = @intFromFloat(@round(tc.rope_parameters.partial_rotary_factor * @as(f64, @floatFromInt(tc.head_dim))));
    if (rotary_dim == 0 or rotary_dim > tc.head_dim) return error.InvalidConfig;

    const is_qwen35_9b =
        tc.hidden_size == 4096 and
        tc.intermediate_size == 12288 and
        tc.num_hidden_layers == 32 and
        tc.num_attention_heads == 16 and
        tc.num_key_value_heads == 4 and
        tc.head_dim == 256 and
        rotary_dim == 64 and
        tc.full_attention_interval == 4 and
        tc.linear_num_key_heads == 16 and
        tc.linear_num_value_heads == 32 and
        tc.linear_key_head_dim == 128 and
        tc.linear_value_head_dim == 128 and
        tc.linear_conv_kernel_dim == 4 and
        tc.vocab_size == 248320;
    if (!is_qwen35_9b) return error.NotQwen35_9B;
}

fn expectFingerprint(config_json: []const u8) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = config_json,
    });
    var path_buf: [4096]u8 = undefined;
    const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);

    try verifyFingerprint(std.testing.allocator, std.testing.io, path_buf[0..path_len]);
}

fn expectFingerprintError(expected_error: anyerror, config_json: []const u8) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "config.json",
        .data = config_json,
    });
    var path_buf: [4096]u8 = undefined;
    const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);

    try std.testing.expectError(
        expected_error,
        verifyFingerprint(std.testing.allocator, std.testing.io, path_buf[0..path_len]),
    );
}

fn qwen35ConfigJson(
    comptime hidden_size: u32,
    comptime quant_bits: u32,
    comptime quant_group_size: u32,
    comptime quant_mode: []const u8,
) []const u8 {
    return std.fmt.comptimePrint(
        \\{{
        \\  "text_config": {{
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": {d},
        \\    "intermediate_size": 12288,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 256,
        \\    "full_attention_interval": 4,
        \\    "linear_num_key_heads": 16,
        \\    "linear_num_value_heads": 32,
        \\    "linear_key_head_dim": 128,
        \\    "linear_value_head_dim": 128,
        \\    "linear_conv_kernel_dim": 4,
        \\    "vocab_size": 248320,
        \\    "rope_parameters": {{
        \\      "partial_rotary_factor": 0.25
        \\    }}
        \\  }},
        \\  "quantization": {{
        \\    "group_size": {d},
        \\    "bits": {d},
        \\    "mode": "{s}"
        \\  }}
        \\}}
    , .{ hidden_size, quant_group_size, quant_bits, quant_mode });
}

test "verifyFingerprint accepts exact Qwen3.5-9B affine q4 config" {
    try expectFingerprint(qwen35ConfigJson(4096, 4, 64, "affine"));
}

test "verifyFingerprint rejects non-q4 quantization" {
    try expectFingerprintError(
        error.UnsupportedQuantization,
        qwen35ConfigJson(4096, 8, 64, "affine"),
    );
}

test "verifyFingerprint rejects non-9B Qwen config" {
    try expectFingerprintError(
        error.NotQwen35_9B,
        qwen35ConfigJson(1024, 4, 64, "affine"),
    );
}

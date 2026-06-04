//! Aggregates model/runtime unit tests that need the src/ module root.

test {
    _ = @import("model/argmax.zig");
    _ = @import("model/block_attn.zig");
    _ = @import("model/block_linear.zig");
    _ = @import("model/block_mlp.zig");
    _ = @import("model/config.zig");
    _ = @import("model/linear_q4.zig");
    _ = @import("model/model.zig");
    _ = @import("model/prefill.zig");
    _ = @import("model/prefill_arena.zig");
    _ = @import("model/state.zig");
    _ = @import("model/tokenizer.zig");
    _ = @import("runtime/mlx_gemm.zig");
    _ = @import("server/prefix_state_cache.zig");
    _ = @import("server/prefix_persist.zig");
}

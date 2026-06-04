//! Peregrine package surface used by the CLI and server.

pub const metal = @import("runtime/metal.zig");
pub const config = @import("model/config.zig");
pub const safetensors = @import("model/safetensors.zig");
pub const model = @import("model/model.zig");
pub const prefill = @import("model/prefill.zig");
pub const tokenizer = @import("model/tokenizer.zig");
pub const server = @import("server.zig");

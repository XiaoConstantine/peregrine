//! Minimal runtime dtype tags used by Kestrel-derived GEMM planning.

pub const DType = enum {
    f32,
    f16,
    bf16,
    u32,

    pub fn sizeInBytes(self: DType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .u32 => 4,
        };
    }
};

//! Metal function-constant descriptors shared by runtime planning code.

pub const max_bool_function_constants: usize = 16;

pub const Bool = struct {
    index: usize,
    value: bool,
};

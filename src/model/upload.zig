//! Weight upload helpers for resident Qwen tensors.

const metal = @import("../runtime/metal.zig");
const safetensors = @import("safetensors.zig");

pub fn tensorPrivate(
    device: *metal.Device,
    queue: *metal.Queue,
    repo: *const safetensors.Repository,
    info: safetensors.TensorInfo,
) !metal.Buffer {
    const byte_len = try repo.tensorByteLen(info);
    var staging = try device.createSharedBuffer(byte_len);
    defer staging.destroy();
    try repo.readInto(info, staging.slice(u8));

    var destination = try device.createPrivateBuffer(byte_len);
    errdefer destination.destroy();
    try queue.copyBuffer(staging, 0, destination, 0, byte_len);
    return destination;
}

pub fn namedTensorPrivate(
    device: *metal.Device,
    queue: *metal.Queue,
    repo: *const safetensors.Repository,
    name: []const u8,
) !metal.Buffer {
    const info = repo.get(name) orelse return error.TensorNotFound;
    return tensorPrivate(device, queue, repo, info);
}

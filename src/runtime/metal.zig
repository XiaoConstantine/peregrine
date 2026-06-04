//! The Metal runtime boundary: a small, checked, lifetime-tracked Zig facade
//! over the Objective-C bridge. This is the only place Metal touches the rest
//! of the engine. Everything is explicit — no global device, no hidden state.

const std = @import("std");
const ffi = @import("ffi.zig");
const build_options = @import("build_options");
const function_constants = @import("function_constants.zig");
const mlx_gemm = @import("mlx_gemm.zig");

pub const BoolFunctionConstant = function_constants.Bool;
pub const MlxGemm = mlx_gemm;

pub const Grid3D = struct {
    x: usize,
    y: usize = 1,
    z: usize = 1,

    pub fn init(x: usize, y: usize, z: usize) Error!Grid3D {
        if (x == 0 or y == 0 or z == 0) return error.DispatchFailed;
        return .{ .x = x, .y = y, .z = z };
    }
};

pub const IndexedBuffer = struct {
    index: usize,
    buffer: Buffer,
    offset: usize = 0,
};

pub const BufferBinding = struct {
    buffer: Buffer,
    offset: usize = 0,
};

pub const Error = error{
    DeviceUnavailable,
    QueueCreateFailed,
    BufferCreateFailed,
    HeapCreateFailed,
    FenceCreateFailed,
    LibraryLoadFailed,
    FunctionNotFound,
    DispatchFailed,
    TooManyBuffers,
    TooManyFunctionConstants,
};

const max_bound_buffers = 16;

const BoundBufferHandles = struct {
    handles: [max_bound_buffers]*ffi.Buffer = undefined,
    len: usize = 0,
};

fn collectHandles(buffers: []const Buffer) Error!BoundBufferHandles {
    var result = BoundBufferHandles{ .len = buffers.len };
    if (buffers.len > result.handles.len) return error.TooManyBuffers;
    for (buffers, 0..) |buffer, index| {
        result.handles[index] = buffer.handle;
    }
    return result;
}

const BoundBufferBindings = struct {
    handles: [max_bound_buffers]*ffi.Buffer = undefined,
    offsets: [max_bound_buffers]usize = undefined,
    indices: [max_bound_buffers]usize = undefined,
    len: usize = 0,
};

fn collectBindingHandles(bindings: []const BufferBinding) Error!BoundBufferBindings {
    var result = BoundBufferBindings{ .len = bindings.len };
    if (bindings.len > result.handles.len) return error.TooManyBuffers;
    for (bindings, 0..) |binding, index| {
        if (binding.offset > binding.buffer.length) return error.DispatchFailed;
        result.handles[index] = binding.buffer.handle;
        result.offsets[index] = binding.offset;
    }
    return result;
}

fn collectIndexedHandles(bindings: []const IndexedBuffer) Error!BoundBufferBindings {
    var result = BoundBufferBindings{ .len = bindings.len };
    if (bindings.len > result.handles.len) return error.TooManyBuffers;
    for (bindings, 0..) |binding, index| {
        if (binding.offset > binding.buffer.length) return error.DispatchFailed;
        result.handles[index] = binding.buffer.handle;
        result.indices[index] = binding.index;
        result.offsets[index] = binding.offset;
    }
    return result;
}

/// Live and peak shared-buffer bytes, for memory budgeting. The DoD requires
/// peak resident memory ≤ MLX, so the device tracks every allocation it owns.
pub const MemoryUsage = struct {
    current: usize = 0,
    peak: usize = 0,

    fn add(self: *MemoryUsage, bytes: usize) void {
        self.current += bytes;
        if (self.current > self.peak) self.peak = self.current;
    }

    fn sub(self: *MemoryUsage, bytes: usize) void {
        std.debug.assert(self.current >= bytes);
        // Saturate even in ReleaseFast: an accounting slip must not wrap
        // `current` to a huge value and poison the `peak` metric the DoD relies on.
        self.current -= @min(self.current, bytes);
    }
};

/// The Metal device. Buffers carry a `*Device` back-pointer for memory
/// accounting, so a `Device` value must stay pinned for as long as any buffer
/// it allocated is alive — do not copy or relocate it (hold it via pointer).
pub const Device = struct {
    handle: *ffi.Device,
    memory: MemoryUsage = .{},

    /// Acquire the system default Metal device. Returns `DeviceUnavailable`
    /// when no GPU is reachable (e.g. a headless/sandboxed environment).
    pub fn create() Error!Device {
        const handle = ffi.prg_device_create() orelse return error.DeviceUnavailable;
        return .{ .handle = handle };
    }

    pub fn destroy(self: *Device) void {
        ffi.prg_device_destroy(self.handle);
        self.* = undefined;
    }

    pub fn createQueue(self: *Device) Error!Queue {
        const handle = ffi.prg_queue_create(self.handle) orelse return error.QueueCreateFailed;
        return .{ .handle = handle };
    }

    /// A shared (CPU+GPU visible) buffer of `length` bytes, tracked against this
    /// device's memory usage. Free it with `Buffer.destroy`.
    pub fn createSharedBuffer(self: *Device, length: usize) Error!Buffer {
        const handle = ffi.prg_buffer_create_shared(self.handle, length) orelse return error.BufferCreateFailed;
        self.memory.add(length);
        return .{ .handle = handle, .length = length, .accounted_length = length, .device = self, .storage = .shared };
    }

    /// A GPU-only buffer for intermediate tensors. Do not call `slice` on it.
    pub fn createPrivateBuffer(self: *Device, length: usize) Error!Buffer {
        const handle = ffi.prg_buffer_create_private(self.handle, length) orelse return error.BufferCreateFailed;
        self.memory.add(length);
        return .{ .handle = handle, .length = length, .accounted_length = length, .device = self, .storage = .private };
    }

    pub fn createPrivateHeap(self: *Device, length: usize) Error!Heap {
        if (length == 0) return error.HeapCreateFailed;
        const handle = ffi.prg_heap_create_private(self.handle, length) orelse return error.HeapCreateFailed;
        self.memory.add(length);
        return .{ .handle = handle, .length = length, .device = self };
    }

    pub fn createFence(self: *Device) Error!Fence {
        const handle = ffi.prg_fence_create(self.handle) orelse return error.FenceCreateFailed;
        return .{ .handle = handle };
    }

    pub fn loadLibrary(self: *Device, path: [*:0]const u8) Error!Library {
        const handle = ffi.prg_library_create_from_path(self.handle, path) orelse return error.LibraryLoadFailed;
        return .{ .handle = handle };
    }

    /// Load the engine's own compiled `peregrine.metallib` (path baked at build time).
    pub fn loadDefaultLibrary(self: *Device) Error!Library {
        return self.loadLibrary(build_options.metallib_path);
    }

    pub fn createPipeline(self: *Device, library: Library, function_name: [*:0]const u8) Error!Pipeline {
        const handle = ffi.prg_pipeline_create(self.handle, library.handle, function_name) orelse return error.FunctionNotFound;
        return .{ .handle = handle };
    }

    pub fn createPipelineWithBoolConstants(
        self: *Device,
        library: Library,
        function_name: [*:0]const u8,
        constants: []const BoolFunctionConstant,
    ) Error!Pipeline {
        var indices: [function_constants.max_bool_function_constants]usize = undefined;
        var values: [function_constants.max_bool_function_constants]bool = undefined;
        if (constants.len > indices.len) return error.TooManyFunctionConstants;
        for (constants, 0..) |constant, index| {
            indices[index] = constant.index;
            values[index] = constant.value;
        }
        const handle = ffi.prg_pipeline_create_with_bool_constants(
            self.handle,
            library.handle,
            function_name,
            indices[0..constants.len].ptr,
            values[0..constants.len].ptr,
            constants.len,
        ) orelse return error.FunctionNotFound;
        return .{ .handle = handle };
    }
};

pub const Heap = struct {
    handle: *ffi.Heap,
    length: usize,
    device: *Device,

    pub fn destroy(self: *Heap) void {
        self.device.memory.sub(self.length);
        ffi.prg_heap_destroy(self.handle);
        self.* = undefined;
    }

    pub fn createPrivateBuffer(self: *Heap, length: usize) Error!Buffer {
        const handle = ffi.prg_heap_create_private_buffer(self.handle, length) orelse return error.BufferCreateFailed;
        return .{ .handle = handle, .length = length, .accounted_length = 0, .device = self.device, .storage = .private };
    }

    pub fn maxAvailableSize(self: *const Heap, alignment: usize) usize {
        return ffi.prg_heap_max_available_size(self.handle, alignment);
    }
};

pub const Fence = struct {
    handle: *ffi.Fence,

    pub fn destroy(self: *Fence) void {
        ffi.prg_fence_destroy(self.handle);
        self.* = undefined;
    }
};

pub const Queue = struct {
    handle: *ffi.Queue,

    pub fn destroy(self: *Queue) void {
        ffi.prg_queue_destroy(self.handle);
        self.* = undefined;
    }

    /// Copy bytes between Metal buffers with a blit command. This is used during
    /// model upload to stage safetensors bytes into GPU-private resident weights.
    pub fn copyBuffer(
        self: *Queue,
        source: Buffer,
        source_offset: usize,
        destination: Buffer,
        destination_offset: usize,
        length: usize,
    ) Error!void {
        if (source_offset > source.length or length > source.length - source_offset) return error.DispatchFailed;
        if (destination_offset > destination.length or length > destination.length - destination_offset) return error.DispatchFailed;
        const rc = ffi.prg_blit_copy_buffer(
            self.handle,
            source.handle,
            source_offset,
            destination.handle,
            destination_offset,
            length,
        );
        if (rc != 0) return error.DispatchFailed;
    }

    /// Begin a batched command buffer: encode many kernels with `CommandBuffer.dispatch1D`
    /// (they run in order, each seeing the prior's writes), then `commitAndWait` ONCE.
    /// This is the decode hot path — it removes the per-kernel commit+wait round-trip.
    pub fn beginCommandBuffer(self: *Queue) Error!CommandBuffer {
        const h = ffi.prg_cmdbuf_create(self.handle) orelse return error.DispatchFailed;
        return .{ .handle = h };
    }

    /// Begin a command buffer whose encoder uses Metal's concurrent dispatch
    /// type: encoded kernels may overlap, and the caller owns correctness by
    /// placing `CommandBuffer.barrier` between every dependent dispatch pair.
    pub fn beginConcurrentCommandBuffer(self: *Queue) Error!CommandBuffer {
        const h = ffi.prg_cmdbuf_create_concurrent(self.handle) orelse return error.DispatchFailed;
        return .{ .handle = h, .concurrent = true };
    }
};

/// A command buffer with one open compute encoder (serial by default). Encode
/// N kernels, then commit once. Consumed by `commitAndWait`/`abort` (do not
/// reuse after).
pub const CommandBuffer = struct {
    handle: *ffi.CmdBuf,
    concurrent: bool = false,

    /// Buffer-scope memory barrier. No-op on serial encoders, where dispatch
    /// order already implies it; required between dependent dispatches on
    /// concurrent encoders.
    pub fn barrier(self: *CommandBuffer) void {
        if (self.concurrent) ffi.prg_cmdbuf_barrier(self.handle);
    }

    /// Encode one kernel (no commit). Same binding contract as Queue.dispatch1D.
    pub fn dispatch1D(self: *CommandBuffer, pipeline: Pipeline, buffers: []const Buffer, grid: usize) Error!void {
        var handles = try collectHandles(buffers);
        ffi.prg_cmdbuf_encode(self.handle, pipeline.handle, handles.handles[0..handles.len].ptr, handles.len, grid);
    }

    /// Encode one kernel with per-buffer byte offsets.
    pub fn dispatch1DWithBindings(self: *CommandBuffer, pipeline: Pipeline, bindings: []const BufferBinding, grid: usize) Error!void {
        var bindings_handles = try collectBindingHandles(bindings);
        ffi.prg_cmdbuf_encode_offsets(self.handle, pipeline.handle, bindings_handles.handles[0..bindings_handles.len].ptr, bindings_handles.offsets[0..bindings_handles.len].ptr, bindings_handles.len, grid);
    }

    /// Encode one kernel with a fixed threadgroup width into this command buffer.
    pub fn dispatch1DWithThreadgroup(self: *CommandBuffer, pipeline: Pipeline, buffers: []const Buffer, grid: usize, threads_per_threadgroup: usize) Error!void {
        std.debug.assert(threads_per_threadgroup != 0);
        var handles = try collectHandles(buffers);
        ffi.prg_cmdbuf_encode_tg(self.handle, pipeline.handle, handles.handles[0..handles.len].ptr, handles.len, grid, threads_per_threadgroup);
    }

    /// Encode a 3-D threadgroup dispatch with explicit Metal buffer bindings.
    /// This is the minimal command shape required by Kestrel's MLX GEMM kernels:
    /// A/B/D/params are not bound contiguously, and the launcher must dispatch
    /// threadgroups rather than logical threads.
    pub fn dispatchThreadgroups3D(
        self: *CommandBuffer,
        pipeline: Pipeline,
        buffers: []const IndexedBuffer,
        threadgroups_per_grid: Grid3D,
        threads_per_threadgroup: Grid3D,
    ) Error!void {
        std.debug.assert(threadgroups_per_grid.x != 0);
        std.debug.assert(threadgroups_per_grid.y != 0);
        std.debug.assert(threadgroups_per_grid.z != 0);
        std.debug.assert(threads_per_threadgroup.x != 0);
        std.debug.assert(threads_per_threadgroup.y != 0);
        std.debug.assert(threads_per_threadgroup.z != 0);
        var bindings = try collectIndexedHandles(buffers);
        ffi.prg_cmdbuf_encode_threadgroups_3d(
            self.handle,
            pipeline.handle,
            bindings.handles[0..bindings.len].ptr,
            bindings.indices[0..bindings.len].ptr,
            bindings.offsets[0..bindings.len].ptr,
            bindings.len,
            threadgroups_per_grid.x,
            threadgroups_per_grid.y,
            threadgroups_per_grid.z,
            threads_per_threadgroup.x,
            threads_per_threadgroup.y,
            threads_per_threadgroup.z,
        );
    }

    pub fn updateFence(self: *CommandBuffer, fence: Fence) void {
        ffi.prg_cmdbuf_update_fence(self.handle, fence.handle);
    }

    pub fn waitFence(self: *CommandBuffer, fence: Fence) void {
        ffi.prg_cmdbuf_wait_fence(self.handle, fence.handle);
    }

    /// End encoding, commit, and block until the GPU finishes. Consumes `self`.
    pub fn commitAndWait(self: *CommandBuffer) Error!void {
        const rc = ffi.prg_cmdbuf_commit_wait(self.handle);
        self.* = undefined;
        if (rc != 0) return error.DispatchFailed;
    }

    /// End encoding and commit without waiting. The returned value must be
    /// waited before buffers referenced by this command are reused unsafely.
    pub fn commit(self: *CommandBuffer) PendingCommandBuffer {
        const handle = self.handle;
        ffi.prg_cmdbuf_commit(handle);
        self.* = undefined;
        return .{ .handle = handle };
    }

    /// Discard without committing (error-cleanup path). Consumes `self`.
    pub fn abort(self: *CommandBuffer) void {
        ffi.prg_cmdbuf_destroy(self.handle);
        self.* = undefined;
    }
};

pub const PendingCommandBuffer = struct {
    handle: *ffi.CmdBuf,

    pub fn wait(self: *PendingCommandBuffer) Error!void {
        const rc = ffi.prg_cmdbuf_wait_destroy(self.handle);
        self.* = undefined;
        if (rc != 0) return error.DispatchFailed;
    }

    pub fn waitProfile(self: *PendingCommandBuffer) Error!?u64 {
        var gpu_time_ns: u64 = 0;
        const rc = ffi.prg_cmdbuf_wait_destroy_profile(self.handle, &gpu_time_ns);
        self.* = undefined;
        if (rc != 0) return error.DispatchFailed;
        if (gpu_time_ns == 0) return null;
        return gpu_time_ns;
    }
};

const workspace_scratch_slot_count = 2048;
const workspace_constant_slot_count = 32768;

/// Reusable scratch buffers for repeated token forwards. A token step uses a
/// deterministic scratch allocation order; after commit/wait, those buffers can
/// be reused by the next token instead of recreated through Metal.
pub const ScratchPool = struct {
    buffers: [workspace_scratch_slot_count]?Buffer = [_]?Buffer{null} ** workspace_scratch_slot_count,
    constants: [workspace_constant_slot_count]?Buffer = [_]?Buffer{null} ** workspace_constant_slot_count,

    pub fn deinit(self: *ScratchPool) void {
        for (&self.buffers) |*slot| {
            if (slot.*) |*buffer| {
                buffer.destroy();
                slot.* = null;
            }
        }
        for (&self.constants) |*slot| {
            if (slot.*) |*buffer| {
                buffer.destroy();
                slot.* = null;
            }
        }
    }

    fn getScratch(self: *ScratchPool, device: *Device, index: usize, length: usize) Error!Buffer {
        return getFromSlots(&self.buffers, device, index, length, .private);
    }

    fn getConstant(self: *ScratchPool, device: *Device, index: usize, length: usize) Error!Buffer {
        return getFromSlots(&self.constants, device, index, length, .shared);
    }

    const Storage = enum { shared, private };

    fn getFromSlots(slots: []?Buffer, device: *Device, index: usize, length: usize, storage: Storage) Error!Buffer {
        if (index >= slots.len) return error.TooManyBuffers;
        if (slots[index]) |*buffer| {
            if (buffer.length == length) return buffer.*;
            buffer.destroy();
            slots[index] = null;
        }
        const buffer = switch (storage) {
            .shared => try device.createSharedBuffer(length),
            .private => try device.createPrivateBuffer(length),
        };
        slots[index] = buffer;
        return buffer;
    }
};

/// A batched encode scope: one command buffer plus a free-list of scratch buffers
/// whose lifetime must extend to the single commit (the GPU runs the encoded
/// kernels only at commit, so per-call scratch cannot be freed earlier). Encode
/// kernels via `self.cmd`, allocate intermediates via `scratch`/`u32buf`/`f32buf`,
/// then `commitAndWait` — which commits, blocks, and frees all the scratch.
pub const Workspace = struct {
    device: *Device,
    queue: *Queue,
    cmd: CommandBuffer,
    scratch_pool: *ScratchPool,
    n: usize = 0,
    const_n: usize = 0,

    pub fn beginWithScratchPool(device: *Device, queue: *Queue, scratch_pool: *ScratchPool) Error!Workspace {
        return .{ .device = device, .queue = queue, .cmd = try queue.beginCommandBuffer(), .scratch_pool = scratch_pool };
    }

    /// Concurrent-encoder variant for the one-token decode path: dispatches
    /// may overlap, and the encode code owns correctness by calling
    /// `Workspace.barrier` between every dependent stage.
    pub fn beginConcurrentWithScratchPool(device: *Device, queue: *Queue, scratch_pool: *ScratchPool) Error!Workspace {
        return .{ .device = device, .queue = queue, .cmd = try queue.beginConcurrentCommandBuffer(), .scratch_pool = scratch_pool };
    }

    /// Buffer-scope memory barrier; no-op on serial workspaces.
    pub fn barrier(self: *Workspace) void {
        self.cmd.barrier();
    }

    /// A tracked scratch buffer, freed by commitAndWait/abort (not the caller).
    pub fn scratch(self: *Workspace, length: usize) Error!Buffer {
        const b = try self.scratch_pool.getScratch(self.device, self.n, length);
        self.n += 1;
        return b;
    }

    fn constant(self: *Workspace, length: usize) Error!Buffer {
        const b = try self.scratch_pool.getConstant(self.device, self.const_n, length);
        self.const_n += 1;
        return b;
    }

    pub fn u32buf(self: *Workspace, v: u32) Error!Buffer {
        var b = try self.constant(@sizeOf(u32));
        b.slice(u32)[0] = v;
        return b;
    }

    pub fn f32buf(self: *Workspace, v: f32) Error!Buffer {
        var b = try self.constant(@sizeOf(f32));
        b.slice(f32)[0] = v;
        return b;
    }

    pub fn valueBuf(self: *Workspace, comptime T: type, v: T) Error!Buffer {
        var b = try self.constant(@sizeOf(T));
        b.slice(T)[0] = v;
        return b;
    }

    pub fn scratchMark(self: *const Workspace) usize {
        return self.n;
    }

    pub fn resetReusableScratchTo(self: *Workspace, mark: usize) void {
        std.debug.assert(mark <= self.n);
        self.n = mark;
    }

    pub fn commitAndWait(self: *Workspace) Error!void {
        defer self.freePool();
        try self.cmd.commitAndWait();
    }

    pub fn commitPooled(self: *Workspace) PendingCommandBuffer {
        defer self.freePool();
        return self.cmd.commit();
    }

    pub fn abort(self: *Workspace) void {
        self.cmd.abort();
        self.freePool();
    }

    fn freePool(self: *Workspace) void {
        self.n = 0;
        self.const_n = 0;
    }
};

pub const Buffer = struct {
    handle: *ffi.Buffer,
    length: usize,
    accounted_length: usize,
    device: *Device,
    storage: Storage,

    const Storage = enum { shared, private };

    pub fn destroy(self: *Buffer) void {
        if (self.accounted_length != 0) self.device.memory.sub(self.accounted_length);
        ffi.prg_buffer_destroy(self.handle);
        self.* = undefined;
    }

    /// The buffer's CPU-visible contents as a typed slice.
    pub fn slice(self: Buffer, comptime T: type) []T {
        std.debug.assert(self.storage == .shared);
        std.debug.assert(self.length % @sizeOf(T) == 0);
        const raw = ffi.prg_buffer_contents(self.handle).?;
        const ptr: [*]T = @ptrCast(@alignCast(raw));
        return ptr[0..@divExact(self.length, @sizeOf(T))];
    }

    pub fn isHeapBacked(self: Buffer) bool {
        return self.accounted_length == 0;
    }

    pub fn makeAliasableIfHeapBacked(self: Buffer) void {
        if (self.isHeapBacked()) self.makeAliasable();
    }

    pub fn makeAliasable(self: Buffer) void {
        ffi.prg_resource_make_aliasable(@ptrCast(self.handle));
    }
};

pub const Library = struct {
    handle: *ffi.Library,

    pub fn destroy(self: *Library) void {
        ffi.prg_library_destroy(self.handle);
        self.* = undefined;
    }
};

pub const Pipeline = struct {
    handle: *ffi.Pipeline,

    pub fn destroy(self: *Pipeline) void {
        ffi.prg_pipeline_destroy(self.handle);
        self.* = undefined;
    }
};

test "heap-backed buffers and fences are usable runtime primitives" {
    var device = Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return,
        else => return err,
    };
    defer device.destroy();

    const before = device.memory.current;
    {
        var heap = try device.createPrivateHeap(64 * 1024);
        defer heap.destroy();
        try std.testing.expectEqual(before + 64 * 1024, device.memory.current);

        {
            var heap_buffer = try heap.createPrivateBuffer(4096);
            defer heap_buffer.destroy();
            try std.testing.expectEqual(device.memory.current, before + 64 * 1024);
            try std.testing.expect(heap_buffer.isHeapBacked());
            heap_buffer.makeAliasable();
        }
    }
    try std.testing.expectEqual(before, device.memory.current);

    {
        var private_buffer = try device.createPrivateBuffer(4096);
        defer private_buffer.destroy();
        try std.testing.expect(!private_buffer.isHeapBacked());
    }

    var fence = try device.createFence();
    defer fence.destroy();
    var queue = try device.createQueue();
    defer queue.destroy();

    var update = try queue.beginCommandBuffer();
    update.updateFence(fence);
    try update.commitAndWait();

    var wait = try queue.beginCommandBuffer();
    wait.waitFence(fence);
    try wait.commitAndWait();
}

test "blit copy moves bytes through a private buffer" {
    var device = Device.create() catch |err| switch (err) {
        error.DeviceUnavailable => return,
        else => return err,
    };
    defer device.destroy();
    var queue = try device.createQueue();
    defer queue.destroy();

    var source = try device.createSharedBuffer(16);
    defer source.destroy();
    var private = try device.createPrivateBuffer(16);
    defer private.destroy();
    var destination = try device.createSharedBuffer(16);
    defer destination.destroy();

    const src = source.slice(u8);
    for (src, 0..) |*value, index| value.* = @intCast(index + 1);
    @memset(destination.slice(u8), 0);

    try queue.copyBuffer(source, 0, private, 0, 16);
    try queue.copyBuffer(private, 0, destination, 0, 16);
    try std.testing.expectEqualSlices(u8, source.slice(u8), destination.slice(u8));
}

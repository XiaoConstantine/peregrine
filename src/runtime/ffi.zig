//! Raw C declarations for the Objective-C Metal bridge (`bridge.h`/`bridge.m`).
//! Nothing outside `metal.zig` should import this directly — it is the unsafe
//! boundary; `metal.zig` wraps it in a checked, lifetime-tracked facade.

pub const Device = opaque {};
pub const Queue = opaque {};
pub const Buffer = opaque {};
pub const Heap = opaque {};
pub const Fence = opaque {};
pub const Library = opaque {};
pub const Pipeline = opaque {};

pub extern fn prg_device_create() ?*Device;
pub extern fn prg_device_destroy(device: *Device) void;

pub extern fn prg_queue_create(device: *Device) ?*Queue;
pub extern fn prg_queue_destroy(queue: *Queue) void;

pub extern fn prg_buffer_create_shared(device: *Device, length: usize) ?*Buffer;
pub extern fn prg_buffer_create_private(device: *Device, length: usize) ?*Buffer;
pub extern fn prg_buffer_destroy(buffer: *Buffer) void;
pub extern fn prg_buffer_contents(buffer: *Buffer) ?*anyopaque;
pub extern fn prg_blit_copy_buffer(queue: *Queue, source: *Buffer, source_offset: usize, destination: *Buffer, destination_offset: usize, length: usize) c_int;

pub extern fn prg_heap_create_private(device: *Device, length: usize) ?*Heap;
pub extern fn prg_heap_destroy(heap: *Heap) void;
pub extern fn prg_heap_create_private_buffer(heap: *Heap, length: usize) ?*Buffer;
pub extern fn prg_heap_max_available_size(heap: *Heap, alignment: usize) usize;
pub extern fn prg_resource_make_aliasable(resource: *anyopaque) void;
pub extern fn prg_fence_create(device: *Device) ?*Fence;
pub extern fn prg_fence_destroy(fence: *Fence) void;

pub extern fn prg_library_create_from_path(device: *Device, path: [*:0]const u8) ?*Library;
pub extern fn prg_library_destroy(library: *Library) void;

pub extern fn prg_pipeline_create(device: *Device, library: *Library, function_name: [*:0]const u8) ?*Pipeline;
pub extern fn prg_pipeline_create_with_bool_constants(
    device: *Device,
    library: *Library,
    function_name: [*:0]const u8,
    indices: [*]const usize,
    values: [*]const bool,
    count: usize,
) ?*Pipeline;
pub extern fn prg_pipeline_destroy(pipeline: *Pipeline) void;

pub const CmdBuf = opaque {};
pub extern fn prg_cmdbuf_create(queue: *Queue) ?*CmdBuf;
pub extern fn prg_cmdbuf_create_concurrent(queue: *Queue) ?*CmdBuf;
pub extern fn prg_cmdbuf_barrier(cb: *CmdBuf) void;
pub extern fn prg_cmdbuf_encode(cb: *CmdBuf, pipeline: *Pipeline, buffers: [*]const *Buffer, buffer_count: usize, grid: usize) void;
pub extern fn prg_cmdbuf_encode_offsets(cb: *CmdBuf, pipeline: *Pipeline, buffers: [*]const *Buffer, offsets: [*]const usize, buffer_count: usize, grid: usize) void;
pub extern fn prg_cmdbuf_encode_tg(cb: *CmdBuf, pipeline: *Pipeline, buffers: [*]const *Buffer, buffer_count: usize, grid: usize, threads_per_threadgroup: usize) void;
pub extern fn prg_cmdbuf_encode_threadgroups_3d(
    cb: *CmdBuf,
    pipeline: *Pipeline,
    buffers: [*]const *Buffer,
    indices: [*]const usize,
    offsets: [*]const usize,
    buffer_count: usize,
    groups_x: usize,
    groups_y: usize,
    groups_z: usize,
    threads_x: usize,
    threads_y: usize,
    threads_z: usize,
) void;
pub extern fn prg_cmdbuf_update_fence(cb: *CmdBuf, fence: *Fence) void;
pub extern fn prg_cmdbuf_wait_fence(cb: *CmdBuf, fence: *Fence) void;
pub extern fn prg_cmdbuf_commit(cb: *CmdBuf) void;
pub extern fn prg_cmdbuf_wait_destroy(cb: *CmdBuf) c_int;
pub extern fn prg_cmdbuf_wait_destroy_profile(cb: *CmdBuf, gpu_time_ns: *u64) c_int;
pub extern fn prg_cmdbuf_commit_wait(cb: *CmdBuf) c_int;
pub extern fn prg_cmdbuf_destroy(cb: *CmdBuf) void;

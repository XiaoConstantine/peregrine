#ifndef PEREGRINE_BRIDGE_H
#define PEREGRINE_BRIDGE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles. Each is an Objective-C Metal object; ownership is manual
// (the bridge is compiled without ARC), so every create/new pairs with a
// matching destroy.
typedef struct PrgDevice PrgDevice;
typedef struct PrgQueue PrgQueue;
typedef struct PrgBuffer PrgBuffer;
typedef struct PrgHeap PrgHeap;
typedef struct PrgFence PrgFence;
typedef struct PrgLibrary PrgLibrary;
typedef struct PrgPipeline PrgPipeline;

PrgDevice *prg_device_create(void);
void prg_device_destroy(PrgDevice *device);

PrgQueue *prg_queue_create(PrgDevice *device);
void prg_queue_destroy(PrgQueue *queue);

// Shared (CPU+GPU visible) buffer. Returns NULL on failure.
PrgBuffer *prg_buffer_create_shared(PrgDevice *device, size_t length);
// Private GPU-only buffer. Returns NULL on failure.
PrgBuffer *prg_buffer_create_private(PrgDevice *device, size_t length);
void prg_buffer_destroy(PrgBuffer *buffer);
void *prg_buffer_contents(PrgBuffer *buffer);
int prg_blit_copy_buffer(PrgQueue *queue, PrgBuffer *source,
                         size_t source_offset, PrgBuffer *destination,
                         size_t destination_offset, size_t length);

// Private heap and aliasing support used by queued layer-major prefill.
PrgHeap *prg_heap_create_private(PrgDevice *device, size_t length);
void prg_heap_destroy(PrgHeap *heap);
PrgBuffer *prg_heap_create_private_buffer(PrgHeap *heap, size_t length);
size_t prg_heap_max_available_size(PrgHeap *heap, size_t alignment);
void prg_resource_make_aliasable(void *resource);

PrgFence *prg_fence_create(PrgDevice *device);
void prg_fence_destroy(PrgFence *fence);

// Library loaded from a .metallib file path. Returns NULL on failure.
PrgLibrary *prg_library_create_from_path(PrgDevice *device, const char *path);
void prg_library_destroy(PrgLibrary *library);

// Compute pipeline for a named kernel function. Returns NULL on failure.
PrgPipeline *prg_pipeline_create(PrgDevice *device, PrgLibrary *library,
                                 const char *function_name);
PrgPipeline *prg_pipeline_create_with_bool_constants(
    PrgDevice *device, PrgLibrary *library, const char *function_name,
    const size_t *indices, const bool *values, size_t count);
void prg_pipeline_destroy(PrgPipeline *pipeline);

// A command buffer with one open serial compute encoder. Encode many kernels —
// they execute in order, each observing the previous kernel's writes — then
// commit ONCE. Returns NULL on failure.
typedef struct PrgCmdBuf PrgCmdBuf;
PrgCmdBuf *prg_cmdbuf_create(PrgQueue *queue);
// Same, but the encoder uses MTLDispatchTypeConcurrent: encoded kernels may
// run concurrently and the caller MUST place prg_cmdbuf_barrier between every
// dependent pair of dispatches. Used by the one-token decode path so
// independent projections overlap instead of paying serial-encoder drains.
PrgCmdBuf *prg_cmdbuf_create_concurrent(PrgQueue *queue);
// Buffer-scope memory barrier inside the open encoder (concurrent encoders).
void prg_cmdbuf_barrier(PrgCmdBuf *cb);
// Encode one 1-D kernel into the open encoder (no commit).
void prg_cmdbuf_encode(PrgCmdBuf *cb, PrgPipeline *pipeline,
                       PrgBuffer *const *buffers, size_t buffer_count, size_t grid);
void prg_cmdbuf_encode_offsets(PrgCmdBuf *cb, PrgPipeline *pipeline,
                               PrgBuffer *const *buffers, const size_t *offsets,
                               size_t buffer_count, size_t grid);
void prg_cmdbuf_encode_tg(PrgCmdBuf *cb, PrgPipeline *pipeline,
                          PrgBuffer *const *buffers, size_t buffer_count, size_t grid,
                          size_t threads_per_threadgroup);
void prg_cmdbuf_encode_threadgroups_3d(
    PrgCmdBuf *cb, PrgPipeline *pipeline, PrgBuffer *const *buffers,
    const size_t *indices, const size_t *offsets, size_t buffer_count,
    size_t groups_x, size_t groups_y, size_t groups_z,
    size_t threads_x, size_t threads_y, size_t threads_z);
void prg_cmdbuf_update_fence(PrgCmdBuf *cb, PrgFence *fence);
void prg_cmdbuf_wait_fence(PrgCmdBuf *cb, PrgFence *fence);
// End encoding and commit without waiting. Follow with prg_cmdbuf_wait_destroy.
void prg_cmdbuf_commit(PrgCmdBuf *cb);
// Wait for a committed command buffer, then free `cb`.
int prg_cmdbuf_wait_destroy(PrgCmdBuf *cb);
int prg_cmdbuf_wait_destroy_profile(PrgCmdBuf *cb, uint64_t *gpu_time_ns);
// End encoding, commit, wait, and free `cb`. Returns 0 on success, non-zero on GPU error.
int prg_cmdbuf_commit_wait(PrgCmdBuf *cb);
// Abort: end encoding and free `cb` WITHOUT committing (error-cleanup path).
void prg_cmdbuf_destroy(PrgCmdBuf *cb);

#ifdef __cplusplus
}
#endif

#endif // PEREGRINE_BRIDGE_H

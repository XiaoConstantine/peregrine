#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "bridge.h"

// Manual reference counting (compiled without -fobjc-arc): every object we hand
// back across the C boundary is +1 retained by its create/new* call and freed
// by the matching destroy. Opaque Prg* pointers are the id values themselves.

static uint64_t prg_metal_time_interval_to_ns(CFTimeInterval time_interval) {
    if (time_interval <= 0.0) return 0;
    return (uint64_t)(time_interval * 1000000000.0);
}

PrgDevice *prg_device_create(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (PrgDevice *)device;
}

void prg_device_destroy(PrgDevice *device) {
    [(id<MTLDevice>)device release];
}

PrgQueue *prg_queue_create(PrgDevice *device) {
    id<MTLCommandQueue> queue = [(id<MTLDevice>)device newCommandQueue];
    return (PrgQueue *)queue;
}

void prg_queue_destroy(PrgQueue *queue) {
    [(id<MTLCommandQueue>)queue release];
}

PrgBuffer *prg_buffer_create_shared(PrgDevice *device, size_t length) {
    id<MTLBuffer> buffer = [(id<MTLDevice>)device newBufferWithLength:length
                                                             options:MTLResourceStorageModeShared];
    return (PrgBuffer *)buffer;
}

PrgBuffer *prg_buffer_create_private(PrgDevice *device, size_t length) {
    id<MTLBuffer> buffer = [(id<MTLDevice>)device newBufferWithLength:length
                                                             options:MTLResourceStorageModePrivate];
    return (PrgBuffer *)buffer;
}

PrgHeap *prg_heap_create_private(PrgDevice *device, size_t length) {
    if (device == NULL || length == 0) return NULL;
    MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor alloc] init];
    if (descriptor == nil) return NULL;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.size = length;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;

    id<MTLHeap> heap = [(id<MTLDevice>)device newHeapWithDescriptor:descriptor];
    [descriptor release];
    return (PrgHeap *)heap;
}

void prg_heap_destroy(PrgHeap *heap) {
    [(id<MTLHeap>)heap release];
}

PrgBuffer *prg_heap_create_private_buffer(PrgHeap *heap, size_t length) {
    if (heap == NULL || length == 0) return NULL;
    id<MTLBuffer> buffer = [(id<MTLHeap>)heap newBufferWithLength:length
                                                          options:MTLResourceStorageModePrivate];
    return (PrgBuffer *)buffer;
}

size_t prg_heap_max_available_size(PrgHeap *heap, size_t alignment) {
    if (heap == NULL || alignment == 0) return 0;
    return (size_t)[(id<MTLHeap>)heap maxAvailableSizeWithAlignment:alignment];
}

void prg_resource_make_aliasable(void *resource) {
    if (resource == NULL) return;
    [(id<MTLResource>)resource makeAliasable];
}

PrgFence *prg_fence_create(PrgDevice *device) {
    if (device == NULL) return NULL;
    id<MTLFence> fence = [(id<MTLDevice>)device newFence];
    return (PrgFence *)fence;
}

void prg_fence_destroy(PrgFence *fence) {
    [(id<MTLFence>)fence release];
}

void prg_buffer_destroy(PrgBuffer *buffer) {
    [(id<MTLBuffer>)buffer release];
}

void *prg_buffer_contents(PrgBuffer *buffer) {
    return [(id<MTLBuffer>)buffer contents];
}

int prg_blit_copy_buffer(PrgQueue *queue, PrgBuffer *source,
                         size_t source_offset, PrgBuffer *destination,
                         size_t destination_offset, size_t length) {
    @autoreleasepool {
        if (queue == NULL || source == NULL || destination == NULL) return 1;
        id<MTLCommandBuffer> command = [(id<MTLCommandQueue>)queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (command == nil || encoder == nil) return 1;
        [encoder copyFromBuffer:(id<MTLBuffer>)source
                   sourceOffset:source_offset
                       toBuffer:(id<MTLBuffer>)destination
              destinationOffset:destination_offset
                           size:length];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return ([command status] == MTLCommandBufferStatusError) ? 1 : 0;
    }
}

PrgLibrary *prg_library_create_from_path(PrgDevice *device, const char *path) {
    @autoreleasepool {
        NSString *ns_path = [NSString stringWithUTF8String:path];
        NSURL *url = [NSURL fileURLWithPath:ns_path];
        NSError *error = nil;
        id<MTLLibrary> library = [(id<MTLDevice>)device newLibraryWithURL:url error:&error];
        if (library == nil) {
            NSLog(@"peregrine: newLibraryWithURL failed: %@", error);
            return NULL;
        }
        return (PrgLibrary *)library;
    }
}

void prg_library_destroy(PrgLibrary *library) {
    [(id<MTLLibrary>)library release];
}

PrgPipeline *prg_pipeline_create(PrgDevice *device, PrgLibrary *library,
                                 const char *function_name) {
    @autoreleasepool {
        NSString *name = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [(id<MTLLibrary>)library newFunctionWithName:name];
        if (function == nil) {
            NSLog(@"peregrine: function '%s' not found in library", function_name);
            return NULL;
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [(id<MTLDevice>)device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (pipeline == nil) {
            NSLog(@"peregrine: newComputePipelineStateWithFunction failed: %@", error);
            return NULL;
        }
        return (PrgPipeline *)pipeline;
    }
}

PrgPipeline *prg_pipeline_create_with_bool_constants(
    PrgDevice *device, PrgLibrary *library, const char *function_name,
    const size_t *indices, const bool *values, size_t count) {
    @autoreleasepool {
        if (indices == NULL || values == NULL) return NULL;
        NSString *name = [NSString stringWithUTF8String:function_name];
        MTLFunctionConstantValues *constant_values = [[MTLFunctionConstantValues alloc] init];
        for (size_t i = 0; i < count; i++) {
            bool value = values[i];
            [constant_values setConstantValue:&value type:MTLDataTypeBool atIndex:indices[i]];
        }
        NSError *error = nil;
        id<MTLFunction> function =
            [(id<MTLLibrary>)library newFunctionWithName:name constantValues:constant_values error:&error];
        [constant_values release];
        if (function == nil) {
            NSLog(@"peregrine: function '%s' with constants not found in library: %@", function_name, error);
            return NULL;
        }
        id<MTLComputePipelineState> pipeline =
            [(id<MTLDevice>)device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (pipeline == nil) {
            NSLog(@"peregrine: newComputePipelineStateWithFunction failed: %@", error);
            return NULL;
        }
        return (PrgPipeline *)pipeline;
    }
}

void prg_pipeline_destroy(PrgPipeline *pipeline) {
    [(id<MTLComputePipelineState>)pipeline release];
}

// Encode one 1-D dispatch into an already-open compute encoder.
static void encode_dispatch_tg_offsets(id<MTLComputeCommandEncoder> encoder,
                                       id<MTLComputePipelineState> pso,
                                       PrgBuffer *const *buffers,
                                       const size_t *offsets,
                                       size_t buffer_count, size_t grid,
                                       size_t requested_threads_per_threadgroup) {
    [encoder setComputePipelineState:pso];
    for (size_t i = 0; i < buffer_count; i++) {
        const size_t offset = offsets == NULL ? 0 : offsets[i];
        [encoder setBuffer:(id<MTLBuffer>)buffers[i] offset:offset atIndex:i];
    }
    NSUInteger width = requested_threads_per_threadgroup;
    if (width == 0) width = [pso threadExecutionWidth];
    if (width == 0) width = 1;
    if (width > grid) width = grid == 0 ? 1 : grid;
    [encoder dispatchThreads:MTLSizeMake(grid, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

static void encode_dispatch_tg(id<MTLComputeCommandEncoder> encoder,
                               id<MTLComputePipelineState> pso,
                               PrgBuffer *const *buffers, size_t buffer_count, size_t grid,
                               size_t requested_threads_per_threadgroup) {
    encode_dispatch_tg_offsets(encoder, pso, buffers, NULL, buffer_count, grid,
                               requested_threads_per_threadgroup);
}

static void encode_dispatch(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pso,
                            PrgBuffer *const *buffers, size_t buffer_count, size_t grid) {
    encode_dispatch_tg(encoder, pso, buffers, buffer_count, grid, 0);
}

static void encode_dispatch_threadgroups_3d(id<MTLComputeCommandEncoder> encoder,
                                            id<MTLComputePipelineState> pso,
                                            PrgBuffer *const *buffers,
                                            const size_t *indices,
                                            const size_t *offsets,
                                            size_t buffer_count,
                                            size_t groups_x,
                                            size_t groups_y,
                                            size_t groups_z,
                                            size_t threads_x,
                                            size_t threads_y,
                                            size_t threads_z) {
    if (encoder == nil || pso == nil ||
        groups_x == 0 || groups_y == 0 || groups_z == 0 ||
        threads_x == 0 || threads_y == 0 || threads_z == 0) {
        return;
    }
    if (buffer_count > 0 && (buffers == NULL || indices == NULL)) {
        return;
    }

    [encoder setComputePipelineState:pso];
    for (size_t i = 0; i < buffer_count; i++) {
        const size_t offset = offsets == NULL ? 0 : offsets[i];
        [encoder setBuffer:(id<MTLBuffer>)buffers[i] offset:offset atIndex:indices[i]];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z)
            threadsPerThreadgroup:MTLSizeMake(threads_x, threads_y, threads_z)];
}

// A command buffer + its open serial compute encoder, both +1 retained so they
// outlive the create call's autorelease pool.
struct PrgCmdBuf {
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
};

static PrgCmdBuf *create_command_buffer(PrgQueue *queue, BOOL concurrent) {
    if (queue == NULL) return NULL;

    PrgCmdBuf *cb = (PrgCmdBuf *)malloc(sizeof(PrgCmdBuf));
    if (cb == NULL) return NULL;

    id<MTLCommandBuffer> command = [(id<MTLCommandQueue>)queue commandBuffer];
    if (command == nil) {
        free(cb);
        return NULL;
    }

    id<MTLComputeCommandEncoder> encoder = concurrent
        ? [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
        : [command computeCommandEncoder];
    if (encoder == nil) {
        free(cb);
        return NULL;
    }

    cb->command = [command retain];
    cb->encoder = [encoder retain];
    return cb;
}

PrgCmdBuf *prg_cmdbuf_create(PrgQueue *queue) {
    @autoreleasepool {
        return create_command_buffer(queue, NO);
    }
}

PrgCmdBuf *prg_cmdbuf_create_concurrent(PrgQueue *queue) {
    @autoreleasepool {
        return create_command_buffer(queue, YES);
    }
}

void prg_cmdbuf_barrier(PrgCmdBuf *cb) {
    if (cb == NULL || cb->encoder == nil) return;
    [cb->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void prg_cmdbuf_encode(PrgCmdBuf *cb, PrgPipeline *pipeline,
                       PrgBuffer *const *buffers, size_t buffer_count, size_t grid) {
    encode_dispatch(cb->encoder, (id<MTLComputePipelineState>)pipeline, buffers, buffer_count, grid);
}

void prg_cmdbuf_encode_offsets(PrgCmdBuf *cb, PrgPipeline *pipeline,
                               PrgBuffer *const *buffers, const size_t *offsets,
                               size_t buffer_count, size_t grid) {
    encode_dispatch_tg_offsets(cb->encoder, (id<MTLComputePipelineState>)pipeline,
                               buffers, offsets, buffer_count, grid, 0);
}

void prg_cmdbuf_encode_tg(PrgCmdBuf *cb, PrgPipeline *pipeline,
                          PrgBuffer *const *buffers, size_t buffer_count, size_t grid,
                          size_t threads_per_threadgroup) {
    encode_dispatch_tg(cb->encoder, (id<MTLComputePipelineState>)pipeline, buffers, buffer_count,
                       grid, threads_per_threadgroup);
}

void prg_cmdbuf_encode_threadgroups_3d(
    PrgCmdBuf *cb, PrgPipeline *pipeline, PrgBuffer *const *buffers,
    const size_t *indices, const size_t *offsets, size_t buffer_count,
    size_t groups_x, size_t groups_y, size_t groups_z,
    size_t threads_x, size_t threads_y, size_t threads_z) {
    encode_dispatch_threadgroups_3d(
        cb->encoder,
        (id<MTLComputePipelineState>)pipeline,
        buffers,
        indices,
        offsets,
        buffer_count,
        groups_x,
        groups_y,
        groups_z,
        threads_x,
        threads_y,
        threads_z);
}

void prg_cmdbuf_update_fence(PrgCmdBuf *cb, PrgFence *fence) {
    if (cb == NULL || cb->encoder == nil || fence == NULL) return;
    [cb->encoder updateFence:(id<MTLFence>)fence];
}

void prg_cmdbuf_wait_fence(PrgCmdBuf *cb, PrgFence *fence) {
    if (cb == NULL || cb->encoder == nil || fence == NULL) return;
    [cb->encoder waitForFence:(id<MTLFence>)fence];
}

int prg_cmdbuf_commit_wait(PrgCmdBuf *cb) {
    prg_cmdbuf_commit(cb);
    return prg_cmdbuf_wait_destroy(cb);
}

void prg_cmdbuf_commit(PrgCmdBuf *cb) {
    if (cb->encoder != nil) {
        [cb->encoder endEncoding];
        [cb->encoder release];
        cb->encoder = nil;
    }
    [cb->command commit];
}

int prg_cmdbuf_wait_destroy(PrgCmdBuf *cb) {
    return prg_cmdbuf_wait_destroy_profile(cb, NULL);
}

int prg_cmdbuf_wait_destroy_profile(PrgCmdBuf *cb, uint64_t *gpu_time_ns) {
    [cb->command waitUntilCompleted];
    if (gpu_time_ns != NULL) {
        CFTimeInterval start = [cb->command GPUStartTime];
        CFTimeInterval end = [cb->command GPUEndTime];
        *gpu_time_ns = (end > start) ? prg_metal_time_interval_to_ns(end - start) : 0;
    }
    int rc = ([cb->command status] == MTLCommandBufferStatusError) ? 1 : 0;
    [cb->command release];
    free(cb);
    return rc;
}

void prg_cmdbuf_destroy(PrgCmdBuf *cb) {
    if (cb->encoder != nil) {
        [cb->encoder endEncoding];
        [cb->encoder release];
    }
    [cb->command release];
    free(cb);
}

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#import <os/lock.h>

// Installed only in the benchmark process. Completion records contain no tensor data.
enum { MaxRecords = 1024, MaxMethods = 3 };

typedef struct {
    double submitted;
    double gpu_start;
    double gpu_end;
} BufferTime;

static IMP original_commit;
static os_unfair_lock gate = OS_UNFAIR_LOCK_INIT;
static BufferTime records[MaxRecords];
static size_t count, pending;
static BOOL recording, overflow;

static void profiled_commit(id<MTLCommandBuffer> buffer, SEL action) {
    size_t index = MaxRecords;
    os_unfair_lock_lock(&gate);
    if (recording && count < MaxRecords) {
        index = count++;
        ++pending;
        records[index] = (BufferTime){.submitted = CACurrentMediaTime()};
    } else if (recording) {
        overflow = YES;
    }
    os_unfair_lock_unlock(&gate);

    if (index != MaxRecords) {
        [buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
            os_unfair_lock_lock(&gate);
            if (completed.status == MTLCommandBufferStatusCompleted) {
                records[index].gpu_start = completed.GPUStartTime;
                records[index].gpu_end = completed.GPUEndTime;
            }
            --pending;
            os_unfair_lock_unlock(&gate);
        }];
    }
    ((void (*)(id, SEL))original_commit)(buffer, action);
}

size_t local_timing_install(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return 0;
        }
        MTLCommandBufferDescriptor *descriptor = [[MTLCommandBufferDescriptor alloc] init];
        descriptor.retainedReferences = NO;
        id<MTLCommandBuffer> buffers[] = {
            [queue commandBuffer], [queue commandBufferWithUnretainedReferences],
            [queue commandBufferWithDescriptor:descriptor],
        };
        Method target = NULL;
        for (size_t i = 0; i < MaxMethods; ++i) {
            if (buffers[i] == nil) {
                continue;
            }
            Method method = class_getInstanceMethod(object_getClass(buffers[i]), @selector(commit));
            if (method == NULL || (target != NULL && target != method)) {
                return 0;
            }
            target = method;
        }
        if (target == NULL) {
            return 0;
        }
        if (original_commit == NULL) {
            original_commit = method_setImplementation(target, (IMP)profiled_commit);
        }
        return 1;
    }
}

double local_timing_begin(void) {
    os_unfair_lock_lock(&gate);
    if (pending || recording) {
        os_unfair_lock_unlock(&gate);
        return 0;
    }
    count = 0;
    overflow = NO;
    double now = CACurrentMediaTime();
    recording = YES;
    os_unfair_lock_unlock(&gate);
    return now;
}

double local_timing_end(void) {
    os_unfair_lock_lock(&gate);
    recording = NO;
    double now = CACurrentMediaTime();
    os_unfair_lock_unlock(&gate);
    return now;
}

// -1 means completion handlers are still running. -2 means the bounded trace was too small.
int local_timing_read(BufferTime *destination, size_t capacity) {
    os_unfair_lock_lock(&gate);
    int result;
    if (recording || pending) {
        result = -1;
    } else if (overflow || count > capacity) {
        result = -2;
    } else {
        memcpy(destination, records, count * sizeof(*destination));
        result = (int)count;
    }
    os_unfair_lock_unlock(&gate);
    return result;
}

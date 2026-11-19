#include <mlx/backend/metal/device.h>
#include <mlx/device.h>

// Benchmark-only access, built against the exact MLX wheel's shipped headers.
// The caller synchronizes the GPU before every update.
namespace {
using MetalDevice = mlx::core::metal::Device;

struct OpsLimit {
    using Member = int MetalDevice::*;
    friend Member limit_member(OpsLimit);
};

struct ByteLimit {
    using Member = int MetalDevice::*;
    friend Member limit_member(ByteLimit);
};

template <typename Tag, typename Tag::Member Field>
struct LimitAccess {
    friend typename Tag::Member limit_member(Tag) { return Field; }
};

template struct LimitAccess<OpsLimit, &MetalDevice::max_ops_per_buffer_>;
template struct LimitAccess<ByteLimit, &MetalDevice::max_mb_per_buffer_>;

MetalDevice& device() {
    return mlx::core::metal::device(mlx::core::Device::gpu);
}
}  // namespace

extern "C" void local_buffer_limits(int* operations, int* mb) {
    auto [ops, bytes] = device().get_max_ops_mb_per_buffer();
    *operations = ops;
    *mb = bytes;
}

extern "C" int local_set_buffer_limits(int operations, int mb) {
    if (operations < 1 || operations > 128 || mb < 1 || mb > 1024) {
        return 0;
    }
    auto& gpu = device();
    gpu.*limit_member(OpsLimit{}) = operations;
    gpu.*limit_member(ByteLimit{}) = mb;
    return 1;
}

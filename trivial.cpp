#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto platform = platforms[1];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices[0];

    cl::Context context { device };
    cl::CommandQueue queue { context, device };

    cl::Program add { context,
R"CLC(
kernel void add( const global int *vector_a,
                 const global int *vector_b,
                 global int *vector_c ) {
    const int id = get_global_id(0);
    vector_c[ id ] = vector_a[ id ] + vector_b[ id ];
}
)CLC", true};

    auto addKernel = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>{ add, "add" };

    typedef int Type;
    const int N = 1 << 23;

    std::vector<Type> vA (N);
    std::iota(vA.begin(), vA.end(), 0);
    std::vector<Type> vB (N, 1);
    std::vector<Type> vC (N);

    cl::Buffer vADevice { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Type) * vA.size(), vA.data() };
    cl::Buffer vBDevice { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Type) * vB.size(), vB.data() };
    cl::Buffer vCDevice { context, CL_MEM_WRITE_ONLY, sizeof(Type) * vC.size(), nullptr };

    addKernel(
            cl::EnqueueArgs( queue, cl::NDRange(N) ),
            vADevice, vBDevice, vCDevice
    );

    queue.enqueueReadBuffer(vCDevice, CL_TRUE, 0, vC.size() * sizeof(Type), vC.data());

    for( int i = 0; i < N; ++i ) {
        assert( vC[i] == vA[i] + vB[i] );
    }
}
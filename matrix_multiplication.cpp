#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <sstream>

#include <string_view>

using namespace std::literals::string_view_literals;

static constexpr std::string_view kernelMultiplySrc { R"CLC(
kernel void matrixMultiply(
                const global float* A,
			    const int numARows, const int numAColumns,
                const global float* B,
			    const int numBRows, const int numBColumns,
		        global float* C,
			    const int numCRows, const int numCColumns,
                // local
			    local float* ds_M,
			    local float* ds_N
) {
    const int tx  = get_local_id(0);
    const int ty  = get_local_id(1);
    const int col = get_group_id(0) * TILE_WIDTH + tx;
    const int row = get_group_id(1) * TILE_WIDTH + ty;

    float Pvalue = 0;

    for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {

        const int mtwtx = m * TILE_WIDTH + tx;
        if (row < numARows && mtwtx < numAColumns) {
          ds_M[ty * TILE_WIDTH + tx] = A[row * numAColumns + mtwtx];
        } else {
          ds_M[ty * TILE_WIDTH + tx] = 0;
        }

        const int mtwty =  m * TILE_WIDTH+ty ;
        if (col < numBColumns && mtwty < numBRows) {
          ds_N[ty * TILE_WIDTH + tx] = B[mtwty * numBColumns + col];
        } else {
          ds_N[ty * TILE_WIDTH + tx] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_WIDTH; ++k) {
           Pvalue += ds_M[ty * TILE_WIDTH + k] * ds_N[k * TILE_WIDTH + tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < numCRows && col < numCColumns) {
       C[row * numCColumns + col] = Pvalue;
    }
}
)CLC"sv };

void printMatrix(const std::vector<float>& a, int x, int y) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            std::cout << a[i * y + j] << ' ';
        }
        std::cout << std::endl;
    }
}

void multiplyMatrices(const std::vector<float>& a, int rowsA, int colsA,
              const std::vector<float>& b, int bX, int colsB,
              std::vector<float>& out) {
    for (int row = 0; row < rowsA; row++) {
        for (int col = 0; col < colsA; col++) {
            float sum = 0;
            for (int k = 0; k < colsB; k++) {
                sum += a[row * colsA + k] * b[k * colsB + col];
            }
            out[row * colsA + col] = sum;
        }
    }
}

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto platform = platforms[1];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices[0];

    cl::Context context { device };
    cl::CommandQueue queue { context, device };

    const int TILE_WIDTH = 16;
    cl::Program multiply { context, kernelMultiplySrc.data() };
    std::ostringstream ss;
    ss << "-D TILE_WIDTH=" << TILE_WIDTH;
    try {
        multiply.build(ss.str().data());
    } catch (const cl::Error& e) {
        std::cout << multiply.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        throw e;
    }
    auto multiplyKernel = cl::make_kernel<
            cl::Buffer&, int, int,
            cl::Buffer&, int, int,
            cl::Buffer&, int, int,
            cl::LocalSpaceArg, cl::LocalSpaceArg
    >{ multiply, "matrixMultiply" };

#define M 10

    const int numAColumns = M;
    const int numARows = M;
    const int numBColumns = M;
    const int numBRows = M;
    const int numCColumns = M;
    const int numCRows = M;

    std::vector<float> matrixAHost ( numAColumns * numARows );
    std::iota( matrixAHost.begin(), matrixAHost.end(), 0);
    std::vector<float> matrixBHost ( numBColumns * numBRows, 0.5f );

    std::vector<float> matrixCHost ( numCColumns * numCRows );
    std::vector<float> matC = matrixCHost;

    cl::Buffer matrixA { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float) * matrixAHost.size(), matrixAHost.data() };
    cl::Buffer matrixB { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float) * matrixBHost.size(), matrixBHost.data() };
    cl::Buffer matrixC { context, CL_MEM_WRITE_ONLY,
                         sizeof(float) * matrixCHost.size() };

    multiplyKernel(
            cl::EnqueueArgs{queue,
                            cl::NDRange {(numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1},
                            cl::NDRange {TILE_WIDTH, TILE_WIDTH},

            },
            matrixA, numARows, numAColumns,
            matrixB, numBRows, numBColumns,
            matrixC, numCRows, numCColumns,
            cl::Local(TILE_WIDTH * TILE_WIDTH), cl::Local(TILE_WIDTH * TILE_WIDTH)
    );

    queue.enqueueReadBuffer(matrixC, CL_TRUE, 0, matrixAHost.size() * sizeof(float), matrixCHost.data());

    printMatrix( matrixAHost, numARows, numAColumns );
    printMatrix( matrixBHost, numBRows, numBColumns );
    printMatrix( matrixCHost, numCRows, numCColumns );

    // do same on cpu
    multiplyMatrices(matrixAHost, numAColumns, numARows, matrixBHost, numBColumns, numBRows, matC);

    for (size_t i = 0; i < matrixCHost.size(); ++i) {
        assert( matrixCHost[i] == matC[i] );
    }

}
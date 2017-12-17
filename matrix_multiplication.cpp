#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include <sstream>

#include <string_view>

using namespace std::literals::string_view_literals;

/**! \todo test for different matrices! */

static constexpr std::string_view kernelMultiplySrc { R"CLC(
// -D TS=
kernel void matrixMultiply(const int M, const int N, const int K,
                    const global float* A,
                    const global float* B,
                    global float* C) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;
    local float Asub[TS][TS];
    local float Bsub[TS][TS];

    float sum = 0.0f;
    const int numTiles = K / TS;
    for (int t=0; t < numTiles; ++t) {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        Asub[col][row] = A[tiledCol * M + globalRow];
        Bsub[col][row] = B[globalCol * K + tiledRow];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k < TS; ++k) {
            sum += Asub[k][row] * Bsub[col][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[globalCol * M + globalRow] = sum;
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

    const int TILE_WIDTH = 3;
    cl::Program multiply { context, kernelMultiplySrc.data() };
    std::ostringstream ss;
    ss << "-D TS=" << TILE_WIDTH;
    try {
        multiply.build(ss.str().data());
    } catch (const cl::Error& e) {
        std::cout << multiply.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        throw e;
    }
    auto multiplyKernel = cl::make_kernel<
            int, int, int,
            cl::Buffer&, cl::Buffer&, cl::Buffer&
    >{ multiply, "matrixMultiply" };

    const int M = 3;
    const int N = 3;
    const int K = 3;

    std::vector<float> matrixAHost ( M * N, 0.5f );
    std::vector<float> matrixBHost ( N * K, 1.5f );

    std::vector<float> matrixCHost ( M * K );
    std::vector<float> matC = matrixCHost;

    cl::Buffer matrixA { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float) * matrixAHost.size(), matrixAHost.data() };
    cl::Buffer matrixB { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float) * matrixBHost.size(), matrixBHost.data() };
    cl::Buffer matrixC { context, CL_MEM_WRITE_ONLY,
                         sizeof(float) * matrixCHost.size() };

    multiplyKernel(
            cl::EnqueueArgs{queue,
                            cl::NDRange {M, N},
                            cl::NDRange {TILE_WIDTH, TILE_WIDTH},
            },
            M, N, K,
            matrixA, matrixB, matrixC
    );

    queue.enqueueReadBuffer(matrixC, CL_TRUE, 0, matrixCHost.size() * sizeof(float), matrixCHost.data());

//    printMatrix( matrixAHost, M, N);
//    printMatrix( matrixBHost, N, K);
//    printMatrix( matrixCHost, M, K);

    // do same on cpu
    multiplyMatrices(matrixAHost, M, N, matrixBHost, N, K, matC);

//    printMatrix(matC, M, K);

    for (size_t i = 0; i < matrixCHost.size(); ++i) {
        assert( matrixCHost[i] == matC[i] );
    }

}
/*!
  * \defgroup trivial_cpp_group Складывание векторов С++
  *
  * Тривиальный пример складывания векторов с использованием технологии openCL. Реализация на языке программирования C++
  * Подробное описание всех действий можно посмотреть в разделе \link{trivial_c_group}
  * @{
  */

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

int main() {
    std::vector<cl::Platform> platforms;
    // Получаем список доступных платформ
    cl::Platform::get(&platforms);
    // Получаем платформу
    auto platform = platforms[0];

    std::vector<cl::Device> devices;
    // Получаем список устройств на платформе
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    // Получаем устройство
    auto device = devices[0];

    // Создаем контекст
    cl::Context context { device };
    // Создаем очередь выполнения
    cl::CommandQueue queue { context, device };

    // Создаем программу для kernel'я
    cl::Program add { context,
                R"CLC(
                kernel void add( const global int *vector_a,
                                 const global int *vector_b,
                                 global int *vector_c ) {
                    const int id = get_global_id(0);
                    vector_c[ id ] = vector_a[ id ] + vector_b[ id ];
                }
                )CLC", true};

    // Создаем kernel
    auto addKernel = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>{ add, "add" };

    typedef int Type;
    // Определяем количество итераций
    const int N = 1 << 23;

    // Создаем массивы данных
    std::vector<Type> vA (N);
    std::iota(vA.begin(), vA.end(), 0);
    std::vector<Type> vB (N, 1);
    std::vector<Type> vC (N);

    // Создаем и ассоциируем openCL буферы с созданными массивами данных
    cl::Buffer vADevice { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Type) * vA.size(), vA.data() };
    cl::Buffer vBDevice { context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Type) * vB.size(), vB.data() };
    cl::Buffer vCDevice { context, CL_MEM_WRITE_ONLY, sizeof(Type) * vC.size(), nullptr };

    // Добавляем kernel в очередь выполнения
    addKernel(
            cl::EnqueueArgs( queue, cl::NDRange(N) ),
            vADevice, vBDevice, vCDevice
    );

    // Добавляем в очередь выполнения задание на считывание данных буфера vC
    queue.enqueueReadBuffer(vCDevice, CL_TRUE, 0, vC.size() * sizeof(Type), vC.data());

    // Проверка правильности выполнения
    for( int i = 0; i < N; ++i ) {
        assert( vC[i] == vA[i] + vB[i] );
    }
}

/*!
 * @}
 */

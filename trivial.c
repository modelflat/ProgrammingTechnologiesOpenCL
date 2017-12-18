/*!
  * \defgroup trivial_c_group Складывание векторов С
  *
  * Тривиальный пример складывания векторов с использованием технологии openCL. Реализация на языке программирования C
  * @{
  */

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <assert.h>
#include <time.h>

/*!
 * \brief printPlatform Печатает информацию о платформе, указанной в plid
 * \param [in] plid Платформа для которой необходимо вывести информацию
 */
void printPlatform(cl_platform_id plid) {
    size_t size;
    // Узнаем размер строки (name)
    clGetPlatformInfo(plid, CL_PLATFORM_NAME, 0, NULL, &size);
    char name[size];
    // Узнаем наименование платформы
    clGetPlatformInfo(plid, CL_PLATFORM_NAME, size, &name, NULL);
    // Узнаем размер строки (vendor)
    clGetPlatformInfo(plid, CL_PLATFORM_VENDOR, 0, NULL, &size);
    char vendor[size];
    // Узнаем наименование вендора
    clGetPlatformInfo(plid, CL_PLATFORM_VENDOR, size, &vendor, NULL);
    // Выводим информацию на экран
    printf("CL Platform: %s (%s)\n", name, vendor);
}

/*!
 * \brief printDevice Печатает информацию об устройстве, указанном в \param did
 * \param in did Устройство для которой необходимо вывести информацию
 */
void printDevice(cl_device_id did) {
    size_t size;
    // Узнаем размер строки (name)
    clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &size);
    char* name = (char*)malloc( sizeof(char)*size );
    char* orig = name;
    //Узнаем наименование устройства
    clGetDeviceInfo(did, CL_DEVICE_NAME, size, name, NULL);
    // strip spaces
    while (*name != '\0' && *(name++) == ' '); --name;

    cl_device_type type;
    // Узнаем тип устройства
    clGetDeviceInfo(did, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);

    cl_uint bits;
    // Узнаем битность устройства
    clGetDeviceInfo(did, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &bits, NULL);
    cl_uint clock;
    // Узнаем максимальную частоту устройства
    clGetDeviceInfo(did, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock, NULL);
    cl_uint units;
    // Узнаем количество физических ядер на устройстве
    clGetDeviceInfo(did, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL);
    size_t virtualProcessors;
    // Узнаем количество виртуальных ядер на одном физическом ядре
    clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &virtualProcessors, NULL);
    cl_ulong localMemSize;
    // Узнаем объем локальной памяти устройства
    clGetDeviceInfo(did, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
    cl_ulong globalMem;
    // Узнаем объем глобальной памяти устройства
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMem, NULL);
    cl_ulong globalCache;
    // Узнаем объем глобальной кеш - памяти
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &globalCache, NULL);
    cl_uint globalCacheLine;
    // Узнаем длинну строки кеша
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &globalCacheLine, NULL);

    // Переводим тип устройства в строку
    const char* strtype = type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU or other";

    // Выводим информацию на экран
    printf("%s / %dbit, type: %s;\n\t%lld virtual processors X %d physical processors X %d MHz = %lld GFLOPS\n\t"
           "memory:\n\t\t%lld bytes global (cache is %lld bytes, cacheline is %d bytes)\n\t\t%lld bytes local\n",
           name, bits, strtype, virtualProcessors, units, clock, virtualProcessors * clock * units / 1000, globalMem, globalCache,
           globalCacheLine, localMemSize
    );

    free( orig );
}

/*!
 * \brief printConfiguration Выводит информацию об устройствах.
 *
 * Выводит информацию, такую как производительность и память, об устройствах в системе, поддерживающих openCL.
 */
void printConfiguration() {
    cl_uint count;
    // Узнаем количество доступных платформ
    clGetPlatformIDs(0, NULL, &count);
    // Получаем список платформ
    cl_platform_id platform[count];
    clGetPlatformIDs(count, platform, NULL);

    // Для каждой платформы:
    for (cl_uint i = 0; i < count; ++i) {
        // Выводим информацию о платформе
        printPlatform(platform[i]);
        printf("Devices:\n");
        cl_uint deviceCount;
        // Узнаем количество устройств
        clGetDeviceIDs( platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        cl_device_id device[deviceCount];
        // Получаем список устройств
        clGetDeviceIDs( platform[i], CL_DEVICE_TYPE_ALL, deviceCount, device, NULL);
        // Для каждого устройства:
        for (cl_uint j = 0; j < deviceCount; ++j) {
            // Выводим информацию об устройстве
            printDevice(device[j]);
        }
    }
}

/*!
 * \brief selectPlatformDevice Находит и возвращает платформу и устройство openCL
 *
 * Функция находит и возвращает платформу и устройство openCL по заданному номеру.\n
 * Информацию о номере платформы и/или устройства можно получить запустив файл ProgrammingTechnologiesOpenCL-trivial-c без параметров.
 * \param [in] platformId Номер платформы
 * \param [in] deviceId Номер устройства
 * \param [out] platform Указатель на платформу
 * \param [out] device Указатель на устройство
 */
void selectPlatformDevice(int platformId, int deviceId, cl_platform_id* platform, cl_device_id* device) {
    cl_uint count;
    // Узнаем количество платформ
    clGetPlatformIDs(0, NULL, &count);
    cl_platform_id platforms[count];
    // Получаем список платформ
    clGetPlatformIDs(count, platforms, NULL);

    cl_uint deviceCount;
    // Узнаем количество устройств для указанной платформы
    clGetDeviceIDs( platforms[platformId], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    cl_device_id devices[deviceCount];
    // Получаем список устройств
    clGetDeviceIDs( platforms[platformId], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

    // Присваеваем out - переменным нужные платформы и устройство
    *platform = platforms[platformId];
    *device = devices[deviceId];
}

/*!
 * \brief getEventTimingMs Считает время выполнения события
 *
 * Событие хранит информацию о том как протекает событие. Данная функция вычисляет чистое время выполнения события.
 * \param [in] event Событие, для которого необходимо вычислить время
 * \return Время выполнения события
 */
double getEventTimingMs(cl_event* event) {
    cl_ulong start, end;
    // Получаем время начала и конца события
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    return (end - start) / 1e6;
}

int main(int argc, char* argv[]) {
    // Проверяем параметры и при отсутствии необходимых параметров
    // выводим список доступных устройств и
    // информацию о запуске
    if (argc < 3) {
        printConfiguration();
        printf("usage: <platformId> <deviceId>\n");
        return 0;
    }

    cl_platform_id platform;
    cl_device_id device;

    // Получаем указанные платформу и устройство
    selectPlatformDevice(atoi( argv[1] ), atoi( argv[2] ), &platform, &device);

    cl_int err;

    // Создаем массив свойств для создания контекста программы
    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) platform,
            NULL
    };
    // Создаем контекст
    // Контекст является окружением для компиляции и выполнения программы на openCL устройстве
    cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS && "Context creation failed");

    // Создаем массив свойств для очереди выполнения
    cl_command_queue_properties queueProperties[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
            NULL
    };
    // Создаем очередь выполнения
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queueProperties, &err);
    assert(err == CL_SUCCESS && "Queue creation failed");

    // Пишем исходный код программы для kernel'я
    // kernel - обособленная специализированная программа на ЯП openCL C
    // предназначенная для компиляции и выполнения на openCL устройстве
    // Содержит все необходимые данные, такие как контекст, для выполнения на конкретном openCL - устройстве
    static const char* programCode = ""
            "kernel void add(const global TYPE* vector_a,"
            "                const global TYPE* vector_b,"
            "                global TYPE* vector_c) {"
            "   int gid = get_global_id(0);"
            "   vector_c[gid] = vector_a[gid] + vector_b[gid];"
            "}";

    size_t len = strlen(programCode);
    // Создаем программу
    cl_program program = clCreateProgramWithSource(context, 1, &programCode, &len, &err);
    assert(err == CL_SUCCESS && "Program creation failed");
    // Компилируем программу
    err = clBuildProgram(program, 1, &device, "-D TYPE=float", NULL, NULL);

    // В случае ошибки выведем лог OpenCL C компилятора
    if (err != CL_SUCCESS) {
        size_t logSize;
        // Узнаем длинну сообщения
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char log[logSize];
        // Получаем сообщение компилятора
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        // выводим сообщение компилятора на экран
        printf("Program build log:\n%s\n", log);
    }

    // Создаем kernel
    cl_kernel kernel = clCreateKernel(program, "add", &err);
    assert(err == CL_SUCCESS && "Kernel creation failed");

    // Определяем количество итераций
    const size_t N = 1 << 23;
    // Создаем массивы с входящими и выходящими данными
    float* vector_a = malloc(sizeof(float) * N);
    float* vector_b = malloc(sizeof(float) * N);
    float* vector_c = malloc(sizeof(float) * N);
    time_t t;
    time(&t);
    srand((unsigned int) t);

    // Заполняем входящие данные
    for (size_t i = 0; i < N; ++i) {
        vector_a[i] = (rand() % 100) / 100.0f;
        vector_b[i] = (rand() % 100) / 100.0f;
    }

    clock_t writeT = clock();       //< Замеряем время начала создания буферов
    // Создаем буферы данных для openCL устройства и связываем их с ранее созданными массивами данных
    cl_mem vector_a_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof( float ),
                                            vector_a, &err);
    assert(err == CL_SUCCESS && "Buffer A creation failed");
    cl_mem vector_b_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof( float ),
                                            vector_b, &err);
    assert(err == CL_SUCCESS && "Buffer B creation failed");
    cl_mem vector_c_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS && "Buffer C creation failed");
    writeT = clock() - writeT;      //< Получаем время, затраченое на создание буферов

    // Соединяем аргументы kernel'я с созданными ранее буферами
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vector_a_device);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vector_b_device);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vector_c_device);

    size_t globalWorkItems = N;
    // Добавляем в очередь выполнения созданный ранее kernel.
    // После добавления он начинает выполняться.
    cl_event eventKernel;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkItems, NULL, 0, NULL, &eventKernel);
    // Добавляем в очередь выполнения задание на считывание данных буфера vector_c_device
    cl_event eventRead;
    clEnqueueReadBuffer(queue, vector_c_device, CL_TRUE, 0, sizeof(float)*N, vector_c, 0, NULL, &eventRead);
    // Завершаем очередь выполнения
    clFinish(queue);

    double writeTime = writeT * 1000.0 / CLOCKS_PER_SEC;
    // Получаем время выполнения расчетов и время чтения данных
    double executeTime = getEventTimingMs( &eventKernel );
    double readTime = getEventTimingMs( &eventRead );
    // Выводим тайминги на экран
    printf( "Total time to add two vectors of length %lld: %f ms\n", N, writeTime + executeTime + readTime);
    printf( "\twrite:\t\t%f ms\n", writeTime);
    printf( "\texecute:\t%f ms\n", executeTime);
    printf( "\tread back:\t%f ms\n", readTime);

    // Проверка данных
    for (size_t i = 0; i < N; ++i) {
        assert( vector_c[i] == (vector_a[i] + vector_b[i]));
    }

    printf("Success!\n");

    // Освобождение памяти
    clReleaseMemObject(vector_a_device);
    clReleaseMemObject(vector_b_device);
    clReleaseMemObject(vector_c_device);

    free(vector_a);
    free(vector_b);
    free(vector_c);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

/*!
 * @}
 */

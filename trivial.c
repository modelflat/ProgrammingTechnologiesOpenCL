#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <assert.h>
#include <time.h>

void printPlatform(cl_platform_id plid) {
    size_t size;
    clGetPlatformInfo(plid, CL_PLATFORM_NAME, 0, NULL, &size);
    char name[size];
    clGetPlatformInfo(plid, CL_PLATFORM_NAME, size, &name, NULL);
    clGetPlatformInfo(plid, CL_PLATFORM_VENDOR, 0, NULL, &size);
    char vendor[size];
    clGetPlatformInfo(plid, CL_PLATFORM_VENDOR, size, &vendor, NULL);
    printf("CL Platform: %s (%s)\n", name, vendor);
}

void printDevice(cl_device_id did) {
    size_t size;
    clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &size);
    char* name = (char*)malloc( sizeof(char)*size );
    char* orig = name;
    clGetDeviceInfo(did, CL_DEVICE_NAME, size, name, NULL);
    // strip spaces
    while (*name != '\0' && *(name++) == ' '); --name;

    cl_device_type type;
    clGetDeviceInfo(did, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);

    cl_uint bits;
    clGetDeviceInfo(did, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &bits, NULL);
    cl_uint clock;
    clGetDeviceInfo(did, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock, NULL);
    cl_uint units;
    clGetDeviceInfo(did, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &units, NULL);
    size_t virtualProcessors;
    clGetDeviceInfo(did, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &virtualProcessors, NULL);
    cl_ulong localMemSize;
    clGetDeviceInfo(did, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
    cl_ulong globalMem;
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMem, NULL);
    cl_ulong globalCache;
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &globalCache, NULL);
    cl_device_mem_cache_type globalCacheType;
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &globalCacheType, NULL);
    cl_uint globalCacheLine;
    clGetDeviceInfo(did, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &globalCacheLine, NULL);

    const char* strtype = type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU or other";

    printf("%s / %dbit, type: %s;\n\t%lld virtual processors X %d physical processors X %d MHz = %lld FLOPS\n\t"
           "memory:\n\t\t%lld bytes global (cache is %lld bytes, cacheline is %d bytes, cache type: %d)\n\t\t%lld bytes local\n",
           name, bits, strtype, virtualProcessors, units, clock, virtualProcessors * clock * units, globalMem, globalCache,
           globalCacheLine, globalCacheType, localMemSize
    );

    free( orig );
}

void printConfiguration() {
    cl_uint count;
    clGetPlatformIDs(0, NULL, &count);
    cl_platform_id platform[count];
    clGetPlatformIDs(count, platform, NULL);

    for (cl_uint i = 0; i < count; ++i) {
        printPlatform(platform[i]);
        printf("Devices:\n");
        cl_uint deviceCount;
        clGetDeviceIDs( platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        cl_device_id device[deviceCount];
        clGetDeviceIDs( platform[i], CL_DEVICE_TYPE_ALL, deviceCount, device, NULL);
        for (cl_uint j = 0; j < deviceCount; ++j) {
            printDevice(device[j]);
        }
    }
}

int print(int err) {
    printf("Error: %d", err);
    return 0;
}

void selectPlatformDevice(int platformId, int deviceId, cl_platform_id* platform, cl_device_id* device) {
    cl_uint count;
    clGetPlatformIDs(0, NULL, &count);
    cl_platform_id platforms[count];
    clGetPlatformIDs(count, platforms, NULL);

    cl_uint deviceCount;
    clGetDeviceIDs( platforms[platformId], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    cl_device_id devices[deviceCount];
    clGetDeviceIDs( platforms[platformId], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

    *platform = platforms[platformId];
    *device = devices[deviceId];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printConfiguration();
        printf("usage: <platformId> <deviceId>\n");
        return 0;
    }

    cl_platform_id platform;
    cl_device_id device;

    selectPlatformDevice(atoi( argv[1] ), atoi( argv[2] ), &platform, &device);

    cl_int err;

    cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
            NULL
    };
    cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS && "Context creation failed");
    cl_command_queue_properties queueProperties[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
            NULL
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, queueProperties, &err);
    assert(err == CL_SUCCESS && "Queue creation failed");

    static const char* programCode = ""
            "kernel void add(const global TYPE* vector_a,"
            "                const global TYPE* vector_b,"
            "                global TYPE* vector_c) {"
            "   int gid = get_global_id(0);"
            "   vector_c[gid] = vector_a[gid] + vector_b[gid];"
            "}";
    size_t len = strlen(programCode);
    cl_program program = clCreateProgramWithSource(context, 1, &programCode, &len, &err);
    assert(err == CL_SUCCESS && "Program creation failed");
    clBuildProgram(program, 1, &device, "-D TYPE=float", NULL, NULL);
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char log[logSize];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

    printf("Program build log:\n%s\n", log);

    cl_kernel kernel = clCreateKernel(program, "add", &err);
    assert(err == CL_SUCCESS && "Kernel creation failed");

    const int N = 1 << 23;
    float* vector_a = malloc(sizeof(float) * N);
    float* vector_b = malloc(sizeof(float) * N);
    float* vector_c = malloc(sizeof(float) * N);
    time_t t;
    time(&t);
    srand((unsigned int) t);

    for (int i = 0; i < N; ++i) {
        vector_a[i] = (rand() % 100) / 100.0f;
        vector_b[i] = (rand() % 100) / 100.0f;
    }

    cl_mem vector_a_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof( float ),
                                            vector_a, &err);
    assert(err == CL_SUCCESS && "Buffer A creation failed");
    cl_mem vector_b_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof( float ),
                                            vector_b, &err);
    assert(err == CL_SUCCESS && "Buffer B creation failed");
    cl_mem vector_c_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS && "Buffer C creation failed");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vector_a_device);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &vector_b_device);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &vector_c_device);

    size_t globalWorkItems = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkItems, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, vector_c_device, CL_TRUE, 0, sizeof(float)*N, vector_c, 0, NULL, NULL);
    clFinish(queue);

    for (int i = 0; i < N; ++i) {
        assert( vector_c[i] == (vector_a[i] + vector_b[i]) || print(i));
    }

    printf("Success!\n");

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

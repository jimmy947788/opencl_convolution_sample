/*
 * https://stackoom.com/question/1lPQj/OpenCL%E5%AE%9E%E7%8E%B0%E7%9A%84%E7%AE%97%E6%B3%95%E6%AF%94%E6%AD%A3%E5%B8%B8%E5%BE%AA%E7%8E%AF%E6%85%A2
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//Constants
#define INPUT_SIGNAL_WIDTH 8
#define INPUT_SIGNAL_HEIGHT 8
#define OUTPUT_SIGNAL_WIDTH 6
#define OUTPUT_SIGNAL_HEIGHT 6
#define MASK_SIGNAL_WIDTH 3
#define MASK_SIGNAL_HEIGHT 3
#define CL_KERNEL_NAME "convolve"

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_DEVICE_SIZE 256

cl_uint inputSignal[INPUT_SIGNAL_WIDTH][INPUT_SIGNAL_HEIGHT] ={
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 },
        {3, 1, 1, 4, 8, 2, 1, 3 }
    };
cl_uint outputSignal[OUTPUT_SIGNAL_WIDTH][OUTPUT_SIGNAL_HEIGHT];
cl_uint mask[MASK_SIGNAL_WIDTH][MASK_SIGNAL_HEIGHT] = {
        {1, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

void checkErr(cl_int err, const char* name)
{
    if(err != CL_SUCCESS)
    {
        printf("ERROR: %s ( %s )\n", name, err);
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(const char * errInfo, const void * private_info, size_t cb, void * user_data)
{
    printf("Error occurred during context user: %s \n", errInfo);
    exit(EXIT_FAILURE);
}

void show_device_information(cl_device_id device)
{
    cl_uint addr_data;
    /* Extension data */
    char name_data[48], ext_data[4096];

    /* Access device name */
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_NAME, 48 * sizeof(char), name_data, NULL);			
    if(err < 0) {		
        perror("Couldn't read extension data");
        exit(1);
    }

    /* Access device address size */
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_data), &addr_data, NULL);			

    /* Access device extensions */
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 4096 * sizeof(char), ext_data, NULL);			
    
    printf("NAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s\n", name_data, addr_data, ext_data);
}


int load_opencl_kernel_code_file(const char* kernel_code_path, char *source_str)
{
    FILE *fp;
    size_t source_size;

    fp = fopen(kernel_code_path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    //printf(source_str);
    fclose( fp );
    
    return source_size;
}

int GetGetPlatforms(cl_platform_id **platforms)
{
    cl_int errNum;
    cl_uint total_platforms;
    char* ext_data;
    size_t ext_size;

    errNum = clGetPlatformIDs(0, NULL, &total_platforms);
    checkErr((errNum != CL_SUCCESS)? errNum : (total_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs for init.");
    printf("get number of Platforms: %d\n", total_platforms);

    *platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * total_platforms);
    errNum = clGetPlatformIDs(total_platforms, *platforms, NULL);
    checkErr((errNum != CL_SUCCESS)? errNum : (total_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatFormIDs for get data.");
    
    int platformId = -1;
    for (platformId = 0; platformId < total_platforms; platformId++)
    {
        errNum = clGetPlatformInfo(*platforms[platformId], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
        checkErr(errNum, "clGetPlatformInfo for init.");

        ext_data = (char*)malloc(ext_size);
        errNum = clGetPlatformInfo(*platforms[platformId], CL_PLATFORM_EXTENSIONS, ext_size, ext_data, NULL);
        checkErr(errNum, "clGetPlatformInfo for get data.");
        printf("Platform ID: %d\nsupports extensions: \n %s\n", platformId, ext_data); 
    }

    free(ext_data);
    return total_platforms;
}


int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint total_devices;
    cl_platform_id* platforms = NULL;
    cl_context context =NULL;
    cl_command_queue* queues;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputSignalBuffer;
    cl_mem outputSignalBuffer;
    cl_mem maskBuffer;
    cl_device_id* devices = NULL;
    char* source_str = NULL;
    clock_t start_time, end_time;
    float total_time = 0;

    char *kernel_code_path = argv[1];
    printf("kernel code path: %s\n", kernel_code_path);

    int platformId = 0; // only AMD GPU so always 1 platform
    GetGetPlatforms(&platforms);

    errNum = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_GPU, 0, NULL, &total_devices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    } 
    else if (total_devices > 0)
    {
        devices = (cl_device_id *) malloc(sizeof(cl_device_id) * total_devices);
        errNum = clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_GPU, total_devices, devices, NULL);
        //checkErr(errNum, "clGetDeviceIDs");
        printf("found number of GPU : %d\n", total_devices);
    }
    else
    {
        printf("No CPU devices found.\n");
        exit(-1);
    }
    
    // show device info
    for(int deviceId = 0; deviceId < total_devices; deviceId++)
    {
        printf("==================== GPU%d ====================\n", deviceId);
        show_device_information(devices[deviceId]);
        printf("==============================================\n");
    }
    

    // Create an OpenCL context
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, 
        (cl_context_properties) platforms[platformId], 
        0
    };
    context = clCreateContext(contextProperties, total_devices, devices, &contextCallback, NULL, &errNum);
    checkErr(errNum, "clCreateContext");
    printf("create OpenCL context for all GPU ........... successful!!\n");

    // Load the kernel source code into the array source_str
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = load_opencl_kernel_code_file(kernel_code_path, source_str);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
    printf("create OpenCL program from %s ........... successful!!\n", kernel_code_path);

    // Build the program
    errNum = clBuildProgram(program, total_devices, devices, NULL, NULL, NULL);
    checkErr(errNum, "clBuildProgram");
    // 產出Build cl檔案的log不然cl程式碼寫錯也編譯不出來
    if(errNum < 0){
        // Shows the log
        char* build_log;
        size_t log_size;
        // First call to know the proper size
        clGetProgramBuildInfo(program, *devices, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        build_log = malloc(log_size+1);
        // Second call to get the log
        clGetProgramBuildInfo(program, *devices, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf(build_log);
        free(build_log);
    }
    printf("build OpenCL program from %s ........... successful!!\n", "./bin/Convolution.cl");

    // Create OpenCL Kernel program
    kernel = clCreateKernel(program, CL_KERNEL_NAME, &errNum);
    checkErr(errNum, "clCreateKernel");
    printf("Create OpenCL Kernel program :%s", CL_KERNEL_NAME);

    // create buffer for pass parameters and output result
    inputSignalBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint) * INPUT_SIGNAL_WIDTH * INPUT_SIGNAL_HEIGHT, inputSignal, &errNum);
    checkErr(errNum, "clCreateBuffer(inputSignal)");
    printf("Create input Signal Buffer ........... successful!!\n");

    maskBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, sizeof(cl_uint)* MASK_SIGNAL_WIDTH * MASK_SIGNAL_HEIGHT, mask, &errNum);
    checkErr(errNum, "clCreateBuffer(mask)");
    printf("Create mask Buffer ........... successful!!\n");

    outputSignalBuffer= clCreateBuffer(context,CL_MEM_WRITE_ONLY ,sizeof(cl_uint)* OUTPUT_SIGNAL_WIDTH * OUTPUT_SIGNAL_HEIGHT, NULL, &errNum);
    checkErr(errNum, "clCreateBuffer(outputSignal)");
    printf("Create output Signal Buffer ........... successful!!\n");

    queues =(cl_command_queue*) malloc(sizeof(cl_command_queue) * total_devices);
    for(int deviceId = 0; deviceId < total_devices; deviceId++)
    {
        queues[deviceId] = clCreateCommandQueue(context, devices[deviceId], 0, &errNum);
        checkErr(errNum, "clCreateCommandQueue");
        printf("create command queue in GPU%d\n", deviceId);
    }

    start_time = clock(); /* mircosecond */

    cl_uint inputWidth = INPUT_SIGNAL_WIDTH;
    cl_uint maskWidth = MASK_SIGNAL_WIDTH;
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
    errNum |=clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |=clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
    errNum |=clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputWidth);
    errNum |=clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
    checkErr(errNum, "clSetKernelArg");
    printf("send input arguments memory to GPU ........... successful!!\n");

    /*
     * 黑虎鯊A8 RX470 參數，用clinfo來查詢
     * Max work item dimensions 3
     * Max work item sizes      1024x1024x1024
     * Max work group size      256
    */
    const size_t globalWorkSize[1] = { OUTPUT_SIGNAL_WIDTH * OUTPUT_SIGNAL_HEIGHT };
    const size_t localWorkSize[1] = {1};
    errNum = clEnqueueNDRangeKernel(queues[0], kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");
    printf("pass kernel code to GPU%d\n", 0);

    /*
    // 分工給其他GPU運算
    for(int deviceId = 0; deviceId < total_devices; deviceId++)
    {
        errNum = clEnqueueNDRangeKernel(queues[deviceId], kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        checkErr(errNum, "clEnqueueNDRangeKernel");
        printf("pass kernel code to GPU%d\n", deviceId);
    }
    */
    
    errNum = clEnqueueReadBuffer(queues[0], outputSignalBuffer, CL_TRUE, 0, sizeof(cl_uint)* OUTPUT_SIGNAL_WIDTH * OUTPUT_SIGNAL_HEIGHT, outputSignal, 0, NULL, NULL);
    checkErr(errNum, "clEnqueueReadBuffer");
    printf("get result from GPU%d\n", 0);
    for(int j = 0; j < OUTPUT_SIGNAL_HEIGHT; j++)
    {
        for(int i = 0; i < OUTPUT_SIGNAL_WIDTH; i++)
        {
            printf("[%d]", (int)outputSignal[j][i]);
        }
        printf("\n");
    }

    /*
    // 從各個GPU取回運算結果
    for(int deviceId = 0; deviceId < total_devices; deviceId++)
    {
        errNum = clEnqueueReadBuffer(queues[deviceId], outputSignalBuffer, CL_TRUE, 0, sizeof(cl_uint)* OUTPUT_SIGNAL_WIDTH * OUTPUT_SIGNAL_HEIGHT, outputSignal, 0, NULL, NULL);
        checkErr(errNum, "clEnqueueReadBuffer");
        printf("get result from GPU%d\n", deviceId);
        for(int j = 0; j < OUTPUT_SIGNAL_HEIGHT; j++)
        {
            for(int i = 0; i < OUTPUT_SIGNAL_WIDTH; i++)
            {
                printf("[%d]", (int)outputSignal[j][i]);
            }
            printf("\n");
        }
    } */
    end_time = clock();
    
    /* CLOCKS_PER_SEC is defined at time.h */
    total_time = (float)(end_time - start_time)/CLOCKS_PER_SEC;
    printf("porcess kernel program time : %f sec \n", total_time);

release_memory:
    free(platforms);
    free(devices);
    free(source_str);
    //free(queues);

    clReleaseProgram(program);
    clReleaseMemObject(inputSignalBuffer);
    clReleaseMemObject(maskBuffer);
    clReleaseMemObject(outputSignalBuffer);
    for(int deviceId = 0; deviceId < total_devices; deviceId++){
        clReleaseCommandQueue(queues[deviceId]);
    }
    clReleaseContext(context);
    return 0;
}
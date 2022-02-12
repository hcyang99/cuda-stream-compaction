#include "prefixScanCpu.hpp"
#include "prefixScanGpu.cuh"
#include <stdio.h>

void printArray(uint32_t* a, size_t n, char* name)
{
    printf("%s[%llu] = \n", name, n);
    for (size_t i = 0; i < n; ++i)
    {
        printf("%10u", a[i]);
        if (i % 16 == 15)
            printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

void test(size_t size)
{
    uint32_t* h_in; 
    uint32_t* h_out_cpu;
    uint32_t* h_out_gpu;
    
    h_in = new uint32_t[1024]();
    h_out_cpu = new uint32_t[1024];
    h_out_gpu = new uint32_t[1024];
    if (h_in == nullptr || h_out_cpu == nullptr || h_out_gpu == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    genInput(h_in, size);
    prefixScanCpu(h_in, h_out_cpu, size);

    fprintf(stderr, "Testing size %llu ... ", size);

    prefixScanGpuSingleBlock(h_in, h_out_gpu, size);

    bool flag = true;
    for (size_t i = 0; i < size; ++i)
    {
        if (h_out_cpu[i] != h_out_gpu[i])
        {
            flag = false;
            fprintf(stderr, "FAILED\n");
            break;
        }
    }

    if (flag)
        fprintf(stderr, "SUCCESS\n");
    else 
    {
        printArray(h_in, size, "h_in");
        printArray(h_out_cpu, size, "h_out_cpu");
        printArray(h_out_gpu, size, "h_out_gpu");
    }
    
    

    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
}

int main()
{
    test(3);
    test(13);
    test(16);
    test(133);
    test(1000);
    test(1024);
}
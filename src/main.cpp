#include "prefixScanCpu.hpp"
#include "prefixScanGpu.cuh"
#include <stdio.h>

void printArray(uint32_t* a, size_t n, char* name, FILE* file)
{
    fprintf(file, "%s[%llu] = \n", name, n);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(file, "%10u", a[i]);
        if (i % 16 == 15)
            fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

void test(size_t size)
{
    uint32_t* h_in; 
    uint32_t* h_out_cpu;
    uint32_t* h_out_gpu;

    FILE* f_in = fopen("../h_in.txt", "w");
    FILE* f_out_cpu = fopen("../h_out_cpu.txt", "w");
    FILE* f_out_gpu = fopen("../h_out_gpu.txt", "w");
    
    h_in = new uint32_t[size]();
    h_out_cpu = new uint32_t[size];
    h_out_gpu = new uint32_t[size];
    if (h_in == nullptr || h_out_cpu == nullptr || h_out_gpu == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    genInput(h_in, size);
    prefixScanCpuPartial(h_in, h_out_cpu, size);

    fprintf(stderr, "Testing size %llu ... ", size);

    prefixScanGpuPartial(h_in, h_out_gpu, size);

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
        printArray(h_in, size, "h_in", f_in);
        printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
        printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    }
    
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    fclose(f_in);
    fclose(f_out_cpu);
    fclose(f_out_gpu);
}

int main()
{
    test(1024);
    test(16*1024);
    test(384*1024);
    test(512*1024);
}
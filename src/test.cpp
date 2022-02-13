#include "test.hpp"
#include "prefixScanCpu.hpp"
#include "prefixScanGpu.cuh"
#include "utils.hpp"

void test_stream_compaction(size_t size)
{
    uint64_t* h_in; 
    uint64_t* h_out_cpu;
    uint64_t* h_out_gpu;

    FILE* f_in = fopen("../h_in.txt", "w");
    FILE* f_out_cpu = fopen("../h_out_cpu.txt", "w");
    FILE* f_out_gpu = fopen("../h_out_gpu.txt", "w");
    if (f_in == nullptr || f_out_cpu == nullptr || f_out_gpu == nullptr)
    {
        fprintf(stderr, "Cannot Open File!\n");
        exit(1);
    }
    
    h_in = new uint64_t[size];
    h_out_cpu = new uint64_t[size]();
    h_out_gpu = new uint64_t[size]();
    if (h_in == nullptr || h_out_cpu == nullptr || h_out_gpu == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    gen_input_stream_compaction(h_in, size);
    streamCompactionCpu(h_in, h_out_cpu, size);

    fprintf(stderr, "Testing size %llu ... ", size);

    testStreamCompactionGpu(h_in, h_out_gpu, size);

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
        // printArray(h_in, size, "h_in", f_in);
        // printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
        // printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    }

    // printArray(h_in + 1, size, "h_in", f_in);
    // printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
    // printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    fclose(f_in);
    fclose(f_out_cpu);
    fclose(f_out_gpu);
}

void test_prefix_scan(size_t size)
{
    uint32_t* h_in; 
    uint32_t* h_out_cpu;
    uint32_t* h_out_gpu;

    FILE* f_in = fopen("../h_in.txt", "w");
    FILE* f_out_cpu = fopen("../h_out_cpu.txt", "w");
    FILE* f_out_gpu = fopen("../h_out_gpu.txt", "w");

    size_t aligned;
    if (size % 1024 != 0)
        aligned = (size / 1024 + 1) * 1024;
    else 
        aligned = size;
    
    h_in = new uint32_t[aligned]();
    h_out_cpu = new uint32_t[aligned];
    h_out_gpu = new uint32_t[aligned];
    if (h_in == nullptr || h_out_cpu == nullptr || h_out_gpu == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    gen_input_prefix_scan(h_in, size);
    prefixScanCpu(h_in, h_out_cpu, size);

    fprintf(stderr, "Testing size %llu ... ", size);

    testPrefixScanGpu(h_in, h_out_gpu, size);

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
        // printArray(h_in, size, "h_in", f_in);
        // printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
        // printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    }

    printArray(h_in, size, "h_in", f_in);
    printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
    printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    fclose(f_in);
    fclose(f_out_cpu);
    fclose(f_out_gpu);
}

void test_mask_gen(size_t size)
{
    uint64_t* h_in; 
    uint32_t* h_out_cpu;
    uint32_t* h_out_gpu;

    FILE* f_in = fopen("../h_in.txt", "w");
    FILE* f_out_cpu = fopen("../h_out_cpu.txt", "w");
    FILE* f_out_gpu = fopen("../h_out_gpu.txt", "w");
    if (f_in == nullptr || f_out_cpu == nullptr || f_out_gpu == nullptr)
    {
        fprintf(stderr, "Cannot Open File!\n");
        exit(1);
    }
    
    h_in = new uint64_t[size]();
    h_out_cpu = new uint32_t[size]();
    h_out_gpu = new uint32_t[size]();
    if (h_in == nullptr || h_out_cpu == nullptr || h_out_gpu == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    gen_input_stream_compaction(h_in, size);    // borrowed
    maskGenCpu(h_in, h_out_cpu, size);

    fprintf(stderr, "Testing size %llu ... ", size);

    testMaskGenGpu(h_in, h_out_gpu, size);

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
        // printArray(h_in, size, "h_in", f_in);
        // printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
        // printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    }

    // printArray(h_in + 1, size, "h_in", f_in);
    // printArray(h_out_cpu, size, "h_out_cpu", f_out_cpu);
    // printArray(h_out_gpu, size, "h_out_gpu", f_out_gpu);
    
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    fclose(f_in);
    fclose(f_out_cpu);
    fclose(f_out_gpu);
}

void test_masked_scatter(size_t size)
{
    uint64_t* h_in; 
    uint32_t* h_mask;
    uint32_t* h_addr;
    uint64_t* h_out_cpu;
    uint64_t* h_out_gpu;

    FILE* f_in = fopen("../h_in.txt", "w");
    FILE* f_out_cpu = fopen("../h_out_cpu.txt", "w");
    FILE* f_out_gpu = fopen("../h_out_gpu.txt", "w");
    if (f_in == nullptr || f_out_cpu == nullptr || f_out_gpu == nullptr)
    {
        fprintf(stderr, "Cannot Open File!\n");
        exit(1);
    }
    
    h_in = new uint64_t[size]();
    h_mask = new uint32_t[size]();
    h_addr = new uint32_t[size]();

    if (h_in == nullptr || h_mask == nullptr || h_addr == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    gen_input_stream_compaction(h_in, size);    // borrowed
    maskGenCpu(h_in, h_mask, size);
    prefixScanCpu(h_mask, h_addr, size);

    size_t out_size = h_addr[size - 1] + h_mask[size - 1];
    h_out_cpu = new uint64_t[out_size]();
    h_out_gpu = new uint64_t[out_size]();

    if (h_out_cpu == nullptr || h_out_gpu == nullptr)
    {
        fprintf(stderr, "Malloc failed!\n");
        exit(1);
    }

    maskedScatterCpu(h_in, h_mask, h_addr, h_out_cpu, size);

    fprintf(stderr, "Testing size %llu ... ", size);

    testMaskedScatterGpu(h_in, h_out_gpu, h_mask, h_addr, size);

    bool flag = true;
    for (size_t i = 0; i < out_size; ++i)
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
        // printArray(h_in + 1, size, "h_in", f_in);
        // printArray(h_out_cpu, out_size, "h_out_cpu", f_out_cpu);
        // printArray(h_out_gpu, out_size, "h_out_gpu", f_out_gpu);
    }

    // printArray(h_in + 1, size, "h_in", f_in);
    // printArray(h_out_cpu, out_size, "h_out_cpu", f_out_cpu);
    // printArray(h_out_gpu, out_size, "h_out_gpu", f_out_gpu);
    
    delete[] h_in;
    delete[] h_mask;
    delete[] h_addr;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    fclose(f_in);
    fclose(f_out_cpu);
    fclose(f_out_gpu);
}
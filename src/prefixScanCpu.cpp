#include "prefixScanCpu.hpp"

void prefixScanCpu(uint32_t* data, uint32_t* out, size_t size)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < size; ++i)
    {
        out[i] = sum;
        sum += data[i];
    }
}

void prefixScanCpuPartial(uint32_t* data, uint32_t* out, size_t size)
{
    uint32_t sum;
    for (size_t i = 0; i < size; ++i)
    {
        if (i % 1024 == 0) sum = 0;
        out[i] = sum;
        sum += data[i];
    }
}

void streamCompactionCpu(uint64_t* data, uint64_t* out, size_t size)
{
    uint64_t last = data[0];
    size_t out_addr = 1;

    out[0] = data[0];

    for (size_t i = 1; i < size; ++i)
    {
        if (data[i] != last)
        {
            last = data[i];
            out[out_addr] = data[i];
            ++out_addr;
        }
    }
}

void maskGenCpu(uint64_t* data, uint32_t* mask, size_t size)
{
    mask[0] = 1;
    for (size_t i = 1; i < size; ++i)
    {
        if (data[i-1] != data[i])
            mask[i] = 1;
        else 
            mask[i] = 0;
    }
}

void maskedScatterCpu(uint64_t* data, uint32_t* mask, uint32_t* addr, uint64_t* out, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        if (mask[i])
            out[addr[i]] = data[i];
    }
}
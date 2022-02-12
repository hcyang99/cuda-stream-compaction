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

void genInput(uint32_t* data, size_t size)
{
    srand(7);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = rand() % 256;
    }
}


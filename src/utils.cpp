#include "utils.hpp"

void gen_input_prefix_scan(uint32_t* data, size_t size)
{
    srand(7);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = rand() % 256;
    }
}

void gen_input_stream_compaction(uint64_t* data, size_t size)
{
    srand(7);
    size_t counter = 0;
    while (true)
    {
        size_t repeat = rand() % 16 + 1;
        uint64_t current = ((uint64_t)rand() << 32) | ((uint64_t)rand() << 16) | (uint64_t)rand();
        while (repeat != 0)
        {
            if (counter >= size)
                return;
            data[counter] = current;
            ++counter;
            --repeat;
        }
    }
}

void printArray(uint32_t* a, size_t n, char* name, FILE* file)
{
    fprintf(file, "%s[%llu] = \n", name, n);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(file, "%10x", a[i]);
        if (i % 16 == 15)
            fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

void printArray(uint64_t* a, size_t n, char* name, FILE* file)
{
    fprintf(file, "%s[%llu] = \n", name, n);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(file, "%24llx", a[i]);
        if (i % 16 == 15)
            fprintf(file, "\n");
    }
    fprintf(file, "\n");
}
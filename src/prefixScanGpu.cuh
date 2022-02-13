#pragma once
#include <stddef.h>
#include <stdint.h>
void printGpuProperties();
void prefixScanGpuSingleBlock(uint32_t* h_in, uint32_t* h_out, size_t length);
void prefixScanGpuPartial(uint32_t* h_in, uint32_t* h_out, size_t length);
void prefixScanGpu(uint32_t* d_in, uint32_t* d_out, size_t length);
void testPrefixScanGpu(uint32_t* h_in, uint32_t* h_out, size_t length);
void stream_compaction(uint64_t** d_out_ptr, size_t* out_length_ptr, uint64_t* d_in, size_t length);
void testStreamCompactionGpu(uint64_t* h_in, uint64_t* h_out, size_t size);
void testMaskGenGpu(uint64_t* h_in, uint32_t* h_out, size_t size);
void testMaskedScatterGpu(uint64_t* h_in, uint64_t* h_out, uint32_t* h_mask, uint32_t* h_addr, size_t length);
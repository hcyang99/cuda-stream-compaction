#pragma once
#include <stdint.h>

// void prefixScanGpu(uint32_t d_in, uint32_t d_out, size_t length);

void prefixScanGpuSingleBlock(uint32_t* h_in, uint32_t* h_out, size_t length);
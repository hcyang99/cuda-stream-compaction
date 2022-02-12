#pragma once
#include <stdint.h>
#include <stdlib.h>

void prefixScanCpu(uint32_t* data, uint32_t* out, size_t size);
void prefixScanCpuPartial(uint32_t* data, uint32_t* out, size_t size);
void genInput(uint32_t* data, size_t size);
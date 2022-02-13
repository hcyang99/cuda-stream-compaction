#pragma once
#include <stddef.h>
#include <stdint.h>

void prefixScanCpu(uint32_t* data, uint32_t* out, size_t size);
void prefixScanCpuPartial(uint32_t* data, uint32_t* out, size_t size);
void streamCompactionCpu(uint64_t* data, uint64_t* out, size_t size);
void maskGenCpu(uint64_t* data, uint32_t* mask, size_t size);
void maskedScatterCpu(uint64_t* data, uint32_t* mask, uint32_t* addr, uint64_t* out, size_t size);
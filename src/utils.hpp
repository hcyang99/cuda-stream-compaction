#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
void printArray(uint32_t* a, size_t n, char* name, FILE* file);
void printArray(uint64_t* a, size_t n, char* name, FILE* file);
void gen_input_prefix_scan(uint32_t* data, size_t size);
void gen_input_stream_compaction(uint64_t* data, size_t size);
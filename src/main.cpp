#include "test.hpp"
#include "prefixScanGpu.cuh"

int main()
{
    printGpuProperties();
    test_stream_compaction(1022);
    test_stream_compaction(1024);
    test_stream_compaction(1025);
    test_stream_compaction(16*1024+724);
    test_stream_compaction(384*1024+74);
    test_stream_compaction(512*1024-46);
    test_stream_compaction(4*1024*1024 + 16);
}
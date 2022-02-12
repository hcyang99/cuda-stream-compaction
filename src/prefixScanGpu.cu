#include "prefixScanGpu.cuh"
#include <stdio.h>

#define LOG_NUM_BANKS 5 
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 


__global__ void 
prescan_single_block(uint32_t *g_odata, uint32_t *g_idata, uint32_t n) 
{ 
    int bx = blockIdx.x;
    if (bx != 0)
        return;
    
    n = 1024;

    __shared__ uint32_t temp[1024 + CONFLICT_FREE_OFFSET(1024)];  // allocated on invocation 

    int tx = threadIdx.x; 
    int offset = 1;

    int ai = 2 * tx;
    int bi = 2 * tx + 1;
    ai += CONFLICT_FREE_OFFSET(ai); 
    bi += CONFLICT_FREE_OFFSET(bi);
    temp[ai] = g_idata[2*tx]; 
    temp[bi] = g_idata[2*tx + 1]; 

    for (uint32_t d = n>>1; d > 0; d >>= 1)  // build sum in place up the tree 
    { 
        __syncthreads();
        int ai = offset*(2*tx+1)-1; 
        int bi = offset*(2*tx+2)-1;     
        if (bi < n) 
        {
            ai += CONFLICT_FREE_OFFSET(ai); 
            bi += CONFLICT_FREE_OFFSET(bi);  
            temp[bi] += temp[ai];    
        }    
        offset *= 2; 
    } 

    if (tx == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n-1)] = 0; } // clear the last element
    for (uint32_t d = 1; d < n; d *= 2) // traverse down tree & build scan 
    {      
        offset >>= 1;      
        __syncthreads();
        int ai = offset*(2*tx+1)-1; 
        int bi = offset*(2*tx+2)-1;      
        if (bi < n) 
        {
            ai += CONFLICT_FREE_OFFSET(ai); 
            bi += CONFLICT_FREE_OFFSET(bi);   
            uint32_t t = temp[ai]; 
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }  
    __syncthreads(); 

    // write results to device memory
    g_odata[2*tx] = temp[ai];
    g_odata[2*tx + 1] = temp[bi];
}

__global__ void 
prescan_partial(uint32_t *g_odata, uint32_t *g_idata, uint32_t n) 
{ 
    __shared__ uint32_t temp[1024 + CONFLICT_FREE_OFFSET(1024)];  // allocated on invocation 

    int bx = blockIdx.x;
    int tx = threadIdx.x; 
    int gs = gridDim.x;
    
    int batch_size = n / 1024 / gs + 1;
    int batch_start = bx * batch_size;
    int batch_end = (bx + 1) * batch_size;
    if (batch_end > n / 1024) batch_end = n / 1024;

    for (int batch = batch_start; batch < batch_end; ++batch)
    {
        int offset = 1;

        int ai = 2 * tx;
        int bi = 2 * tx + 1;
        ai += CONFLICT_FREE_OFFSET(ai); 
        bi += CONFLICT_FREE_OFFSET(bi);
        temp[ai] = g_idata[2*tx + batch * 1024]; 
        temp[bi] = g_idata[2*tx + 1 + batch * 1024]; 

        for (uint32_t d = 1024>>1; d > 0; d >>= 1)  // build sum in place up the tree 
        { 
            __syncthreads();
            int ai = offset*(2*tx+1)-1; 
            int bi = offset*(2*tx+2)-1;     
            if (bi < 1024) 
            {
                ai += CONFLICT_FREE_OFFSET(ai); 
                bi += CONFLICT_FREE_OFFSET(bi);  
                temp[bi] += temp[ai];    
            }    
            offset *= 2; 
        } 

        if (tx == 0) { temp[1024 - 1 + CONFLICT_FREE_OFFSET(1024-1)] = 0; } // clear the last element
        for (uint32_t d = 1; d < 1024; d *= 2) // traverse down tree & build scan 
        {      
            offset >>= 1;      
            __syncthreads();
            int ai = offset*(2*tx+1)-1; 
            int bi = offset*(2*tx+2)-1;      
            if (bi < 1024) 
            {
                ai += CONFLICT_FREE_OFFSET(ai); 
                bi += CONFLICT_FREE_OFFSET(bi);   
                uint32_t t = temp[ai]; 
                temp[ai] = temp[bi]; 
                temp[bi] += t;       
            } 
        }  
        __syncthreads(); 

        // write results to device memory
        g_odata[2*tx + batch * 1024] = temp[ai];
        g_odata[2*tx + 1 + batch * 1024] = temp[bi];
    }
}

__global__ void
prescan_add(uint32_t* d_out, uint32_t* d_in, uint32_t num_batches)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x; 
    int gs = gridDim.x;
    int bs = blockDim.x;

    int batch_size = num_batches / gs + 1;
    int batch_start = bx * batch_size;
    int batch_end = (bx + 1) * batch_size;
    if (batch_end > num_batches) batch_end = num_batches;

    for (uint32_t i = batch_start * 1024 + tx; i < batch_end * 1024; i += bs)
    {
        d_out[i] += d_in[i / 1024];
    }
}

__global__ void
prescan_sum(uint32_t* d_out, uint32_t* d_in, uint32_t num_batches)
{
    __shared__ uint32_t s[32];
    int bx = blockIdx.x;
    int tx = threadIdx.x; 
    int gs = gridDim.x;
    int bs = blockDim.x;

    int batch_size = num_batches / gs + 1;
    int batch_start = bx * batch_size;
    int batch_end = (bx + 1) * batch_size;
    if (batch_end > num_batches) batch_end = num_batches;

    for (int batch = batch_start; batch < batch_end; ++batch)
    {
        uint32_t local_sum = 0;
        for (int i = tx; i < 1024; i += bs)
        {
            local_sum += d_in[i + batch * 1024];
        }
        s[tx] = local_sum;
        __syncthreads();
        if (tx == 0)
        {
            uint32_t master_sum = 0;
            for (int i = 0; i < bs; ++i)
            {
                master_sum += s[i];
            }
            d_out[batch] = master_sum;
        }
        __syncthreads();
    }

    if (bx == gs - 1)
    {
        uint32_t num_batch_aligned = num_batches;
        if (num_batches % 1024 != 0) num_batch_aligned = (num_batches / 1024 + 1) * 1024;
        for (int i = num_batches + tx; i < num_batch_aligned; i += 32)
            d_out[i] = 0;
    }
}

void prefixScanGpu(uint32_t* d_in, uint32_t* d_out, size_t length)
{
    int numBlocks = 128; // i.e. number of thread blocks on the GPU
    int blockSize = 512; // i.e. number of GPU threads per thread block

    if (length <= 1024)
    {
        // fprintf(stderr, "Call: prescan_single_block\n");
        prescan_single_block<<<1, 512>>>(d_out, d_in, 1024);
    } 
    else 
    {
        // call partial scan
        size_t aligned_length;
        if (length % 1024 == 0) aligned_length = length; else aligned_length = (length / 1024 + 1) * 1024;
        prescan_partial<<<128,512>>>(d_out, d_in, aligned_length);

        // call sum
        size_t sum_length, sum_aligned;
        uint32_t* sum_out;
        sum_length = aligned_length / 1024;
        if (sum_length % 1024 == 0) sum_aligned = sum_length; else sum_aligned = (sum_length / 1024 + 1) * 1024;
        auto err = cudaMalloc(&sum_out, sum_aligned*sizeof(uint32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
            exit(1);
        }
        prescan_sum<<<512,32>>>(sum_out, d_in, sum_length);

        // call recursive scan
        uint32_t* sum_scanned;
        err = cudaMalloc(&sum_scanned, sum_aligned*sizeof(uint32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
            exit(1);
        }
        prefixScanGpu(sum_out, sum_scanned, sum_length);
 

        // call scatter add
        prescan_add<<<128, 512>>>(d_out, sum_scanned, sum_length);

        // free memory
        cudaFree(sum_out);
        cudaFree(sum_scanned);
    }
}

void testPrefixScanGpu(uint32_t* h_in, uint32_t* h_out, size_t size)
{
    uint32_t* d_in;
    uint32_t* d_out;
    uint32_t aligned;
    if (size % 1024 != 0)
        aligned = (size / 1024 + 1) * 1024;
    else 
        aligned = size;

    auto err = cudaMalloc(&d_in, aligned*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    err = cudaMalloc(&d_out, aligned*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    err = cudaMemcpy(d_in, h_in, aligned*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    prefixScanGpu(d_in, d_out, size);

    err = cudaMemcpy(h_out, d_out, aligned*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    cudaFree(d_in);
    cudaFree(d_out);
}


void prefixScanGpuSingleBlock(uint32_t* h_in, uint32_t* h_out, size_t length)
{
    if (length > 1024)
    {
        fprintf(stderr, "Invalid Parameter: length out of bound\n");
        exit(1);
    }

    uint32_t* d_in;
    uint32_t* d_out;

    auto err = cudaMalloc(&d_in, length*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_out, length*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_in, h_in, length*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy 1 failed!\n");
        exit(1);
    }

    int numBlocks = 128; // i.e. number of thread blocks on the GPU
    int blockSize = 512; // i.e. number of GPU threads per thread block

    prescan_single_block<<<numBlocks, blockSize>>>(d_out, d_in, 1024);

    err = cudaMemcpy(h_out, d_out, length*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy 2 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorName(err));
        fprintf(stderr, "h_out: %p, d_out: %p\n", h_out, d_out);
        exit(1);
    }

    cudaFree(d_in);
    cudaFree(d_out);
}

void prefixScanGpuPartial(uint32_t* h_in, uint32_t* h_out, size_t length)
{
    if (length % 1024 != 0)
    {
        fprintf(stderr, "Invalid Parameter: length not aligned to 1024\n");
        exit(1);
    }

    uint32_t* d_in;
    uint32_t* d_out;

    auto err = cudaMalloc(&d_in, length*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_out, length*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_in, h_in, length*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy 1 failed!\n");
        exit(1);
    }

    int numBlocks = 128; // i.e. number of thread blocks on the GPU
    int blockSize = 512; // i.e. number of GPU threads per thread block

    prescan_partial<<<numBlocks, blockSize>>>(d_out, d_in, length);

    err = cudaMemcpy(h_out, d_out, length*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy 2 failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorName(err));
        fprintf(stderr, "h_out: %p, d_out: %p\n", h_out, d_out);
        exit(1);
    }

    cudaFree(d_in);
    cudaFree(d_out);
}
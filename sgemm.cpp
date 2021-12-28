#include "hip/hip_runtime.h"
/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <hip/hip_runtime.h>

#define DBG_CHECK_STEP (1)
#define BLOCK_SIZE (16)

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
          }\
}

using namespace std;

/*
 * Square each element in the array A and write to array C.
 */
__global__ void
vector_sgemmNN(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    if (ix < M && iy < N) {
        float result = 0;
        for (size_t j = 0; j < K; j++) {
            result += A_d[j*M + ix] * B_d[iy*K + j];
        }
        C_d[iy*M+ix] = result;
    }
}

__global__ void
vector_sgemmNT(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);
    if (ix < M && iy < N) {
        float result = 0;
        for (size_t j = 0; j < K; j++) {
            result += A_d[j*M + ix] * B_d[j*N + iy];
        }
        C_d[iy*M+ix] = result;
    }
}

__global__ void
vector_sgemmTT(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);
    if (ix < M && iy < N) {
        float result = 0;
        for (size_t j = 0; j < K; j++) {
            result += A_d[ix*K + j] * B_d[j*N + iy];
        }
        C_d[iy*M+ix] = result;
    }
}

__global__ void
vector_sgemmTN(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);
    if (ix < M && iy < N) {
        float result = 0;
        for (size_t j = 0; j < K; j++) {
            result += A_d[ix*K + j] * B_d[iy*K + j];
        }
        C_d[iy*M+ix] = result;
    }
}

__global__ void
vector_sgemmNN_fast(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    __shared__ float sDataA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sDataB[BLOCK_SIZE * BLOCK_SIZE];
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    size_t idxs = threadIdx.y * blockDim.x + threadIdx.x;
    
    //Calculate the iterations of loop
    int iter = (K+BLOCK_SIZE-1) / BLOCK_SIZE;
    int residual = K % BLOCK_SIZE;
    if (residual == 0)
        residual = BLOCK_SIZE;

    size_t ax = blockIdx.x * blockDim.x;
    size_t ay = 0;
    size_t bx = 0;
    size_t by = blockIdx.y * blockDim.y;
    
    float result = 0;
    for (size_t j = 0; j < (iter - 1); j++, ay += BLOCK_SIZE, bx += BLOCK_SIZE) {
        
        //fetch 2 16x16 submatrix of A and B
        sDataA[threadIdx.y*BLOCK_SIZE + threadIdx.x] = A_d[(threadIdx.y + ay)*M + (threadIdx.x + ax)];
        sDataB[threadIdx.y*BLOCK_SIZE + threadIdx.x] = B_d[(threadIdx.y + by)*K + (threadIdx.x + bx)];
        
        //wait until all data is loaded
        __syncthreads();
        
        //submatrix multiply
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
        }

        //wait all calculation done to load the next data
        __syncthreads();
    }
    //fetch 2 16x16 submatrix of A and B
    sDataA[threadIdx.y*BLOCK_SIZE + threadIdx.x] = A_d[(threadIdx.y + ay)*M + (threadIdx.x + ax)];
    sDataB[threadIdx.y*BLOCK_SIZE + threadIdx.x] = B_d[(threadIdx.y + by)*K + (threadIdx.x + bx)];
    
    //wait until all data is loaded
    __syncthreads();
    
    //submatrix multiply
    for (int k = 0; k < residual; ++k) {
        result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
    }

    //wait all calculation done to load the next data
    __syncthreads();
        
    if (ix < M && iy < N) {
        C_d[iy*M+ix] = result;
    }

}

__global__ void
vector_sgemmNT_fast(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    __shared__ float sDataA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sDataB[BLOCK_SIZE * BLOCK_SIZE];
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    size_t idxs = threadIdx.y * blockDim.x + threadIdx.x;
    
    //Calculate the iterations of loop
    int iter = (K+BLOCK_SIZE-1) / BLOCK_SIZE;
    int residual = K % BLOCK_SIZE;
    if (residual == 0)
        residual = BLOCK_SIZE;
    
    size_t ax = blockIdx.x * blockDim.x;
    size_t ay = 0;
    size_t bx = blockIdx.y * blockDim.y;
    size_t by = 0;
    
    float result = 0;
    for (size_t j = 0; j < (iter - 1); j++, ay += BLOCK_SIZE, by += BLOCK_SIZE) {
        
        //fetch 2 16x16 submatrix of A and B
        sDataA[threadIdx.y*BLOCK_SIZE + threadIdx.x] = A_d[(threadIdx.y + ay)*M + (threadIdx.x + ax)];
        sDataB[threadIdx.x*BLOCK_SIZE + threadIdx.y] = B_d[(threadIdx.y + by)*N + (threadIdx.x + bx)];
        
        //wait until all data is loaded
        __syncthreads();
        
        //submatrix multiply
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
        }

        //wait all calculation done to load the next data
        __syncthreads();
    }
    //fetch 2 16x16 submatrix of A and B
    sDataA[threadIdx.y*BLOCK_SIZE + threadIdx.x] = A_d[(threadIdx.y + ay)*M + (threadIdx.x + ax)];
    sDataB[threadIdx.x*BLOCK_SIZE + threadIdx.y] = B_d[(threadIdx.y + by)*N + (threadIdx.x + bx)];
    
    //wait until all data is loaded
    __syncthreads();
    
    //submatrix multiply
    for (int k = 0; k < residual; ++k) {
        result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
    }

    //wait all calculation done to load the next data
    __syncthreads();
        
    if (ix < M && iy < N) {
        C_d[iy*M+ix] = result;
    }
}

__global__ void
vector_sgemmTN_fast(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    __shared__ float sDataA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sDataB[BLOCK_SIZE * BLOCK_SIZE];
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    size_t idxs = threadIdx.y * blockDim.x + threadIdx.x;
    
    //Calculate the iterations of loop
    int iter = (K+BLOCK_SIZE-1) / BLOCK_SIZE;
    int residual = K % BLOCK_SIZE;
    if (residual == 0)
        residual = BLOCK_SIZE;
    
    size_t ax = 0;
    size_t ay = blockIdx.x * blockDim.x;
    size_t bx = 0;
    size_t by = blockIdx.y * blockDim.y;
    
    float result = 0;
    for (size_t j = 0; j < (iter - 1); j++, ax += BLOCK_SIZE, bx += BLOCK_SIZE) {
        
        //fetch 2 16x16 submatrix of A and B
        sDataA[threadIdx.x*BLOCK_SIZE + threadIdx.y] = A_d[(threadIdx.y + ay)*K + (threadIdx.x + ax)];
        sDataB[threadIdx.y*BLOCK_SIZE + threadIdx.x] = B_d[(threadIdx.y + by)*K + (threadIdx.x + bx)];
        
        //wait until all data is loaded
        __syncthreads();
        
        //submatrix multiply
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
        }

        //wait all calculation done to load the next data
        __syncthreads();
    }
    //fetch 2 16x16 submatrix of A and B
    sDataA[threadIdx.x*BLOCK_SIZE + threadIdx.y] = A_d[(threadIdx.y + ay)*K + (threadIdx.x + ax)];
    sDataB[threadIdx.y*BLOCK_SIZE + threadIdx.x] = B_d[(threadIdx.y + by)*K + (threadIdx.x + bx)];
    
    //wait until all data is loaded
    __syncthreads();
    
    //submatrix multiply
    for (int k = 0; k < residual; ++k) {
        result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
    }

    //wait all calculation done to load the next data
    __syncthreads();
        
    if (ix < M && iy < N) {
        C_d[iy*M+ix] = result;
    }
}

__global__ void
vector_sgemmTT_fast(float *C_d, float *A_d, float *B_d, size_t M, size_t K, size_t N)
{
    __shared__ float sDataA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sDataB[BLOCK_SIZE * BLOCK_SIZE];
    size_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    size_t idxs = threadIdx.y * blockDim.x + threadIdx.x;
    
    //Calculate the iterations of loop
    int iter = (K+BLOCK_SIZE-1) / BLOCK_SIZE;
    int residual = K % BLOCK_SIZE;
    if (residual == 0)
        residual = BLOCK_SIZE;
    
    size_t ax = 0;
    size_t ay = blockIdx.x * blockDim.x;
    size_t bx = blockIdx.y * blockDim.y;
    size_t by = 0;
    
    float result = 0;
    for (size_t j = 0; j < (iter - 1); j++, ax += BLOCK_SIZE, by += BLOCK_SIZE) {
        
        //fetch 2 16x16 submatrix of A and B
        sDataA[threadIdx.x*BLOCK_SIZE + threadIdx.y] = A_d[(threadIdx.y + ay)*K + (threadIdx.x + ax)];
        sDataB[threadIdx.x*BLOCK_SIZE + threadIdx.y] = B_d[(threadIdx.y + by)*N + (threadIdx.x + bx)];
        
        //wait until all data is loaded
        __syncthreads();
        
        //submatrix multiply
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
        }

        //wait all calculation done to load the next data
        __syncthreads();
    }
    //fetch 2 16x16 submatrix of A and B
    sDataA[threadIdx.x*BLOCK_SIZE + threadIdx.y] = A_d[(threadIdx.y + ay)*K + (threadIdx.x + ax)];
    sDataB[threadIdx.x*BLOCK_SIZE + threadIdx.y] = B_d[(threadIdx.y + by)*N + (threadIdx.x + bx)];
    
    //wait until all data is loaded
    __syncthreads();
    
    //submatrix multiply
    for (int k = 0; k < residual; ++k) {
        result += sDataA[k*BLOCK_SIZE + threadIdx.x] * sDataB[threadIdx.y*BLOCK_SIZE + k];
    }

    //wait all calculation done to load the next data
    __syncthreads();
        
    if (ix < M && iy < N) {
        C_d[iy*M+ix] = result;
    }
}

int main(int argc, char *argv[]) 
{
    float *A_d, *B_d, *C_d, *C_Gold;
    float *A_h, *B_h, *C_h;
    size_t M = 256;
    size_t N = 256;
    size_t K = 256;
    string transA = "n";
    string transB = "n";
    size_t LDA;
    size_t LDB;
    size_t LDC;

    if(argc == 4){
        M =(size_t)atoi(argv[1]);
        N =(size_t)atoi(argv[2]);
        K =(size_t)atoi(argv[3]);
        if(transA == "n" || transA == "N"){
            LDA = M;
        } else{
            LDA = K;
        }
        
        if(transB == "n" || transB == "N"){
            LDB = K;
        } else{
            LDB = N;
        }
        
    } else if (argc == 6){
        M =(size_t)atoi(argv[1]);
        N =(size_t)atoi(argv[2]);
        K =(size_t)atoi(argv[3]);
        transA = argv[4];
        transB = argv[5];
        if(transA == "n" || transA == "N"){
            LDA = M;
        } else{
            LDA = K;
        }
        
        if(transB == "n" || transB == "N"){
            LDB = K;
        } else{
            LDB = N;
        }
    } else if (argc == 9){
        M =(size_t)atoi(argv[1]);
        N =(size_t)atoi(argv[2]);
        K =(size_t)atoi(argv[3]);
        transA = argv[4];
        transB = argv[5];
        LDA =(size_t)atoi(argv[6]);
        LDB =(size_t)atoi(argv[7]);
        LDC =(size_t)atoi(argv[8]);
        if(transA == "n" || transA == "N"){
            if(LDA < M){
                printf("ERROR: LDA < M\n");
            }
        } else{
            if(LDA < K){
                printf("ERROR: LDA < K\n");
            }
        }
        
        if(transB == "n" || transB == "N"){
            if(LDB < K){
                printf("ERROR: LDB < K\n");
            }
        } else{
            if(LDB < N){
                printf("ERROR: LDB < N\n");
            }
        }
        if(LDC < M){
            printf("ERROR: LDC < M\n");
        }
    } else if(argc != 0){
        printf("Usage: \n");
        printf("M,N,K \n");
        printf("or \n");
        printf("M,N,K,transA,transB,LDA,LDB,LDC \n");
        printf("Run default setting M,N,K = 256\n");
        if(transA == "n" || transA == "N"){
            LDA = M;
        } else{
            LDA = K;
        }
        
        if(transB == "n" || transB == "N"){
            LDB = K;
        } else{
            LDB = N;
        }
    }
    LDC = M;
    size_t heightA;
    size_t heightB;
    size_t widthA;
    size_t widthB;
    bool b_transA, b_transB;
    if(transA == "n" || transA == "N"){
        heightA = K;
        widthA = M;
        b_transA = true;
    } else{
        heightA = M;
        widthA = K;
        b_transA = false;
    }
    
    if(transB == "n" || transB == "N"){
        heightB = N;
        widthB = K;
        b_transB = true;
    } else{
        heightB = K;
        widthB = N;
        b_transB = false;
    }
    size_t heightC = N;
    
    size_t Abytes = heightA * LDA * sizeof(float);
    size_t Bbytes = heightB * LDB * sizeof(float);
    size_t Cbytes = heightC * LDC * sizeof(float);
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    
    printf ("info: allocate host mem A %zu x %zu (%6.2f MB)\n", LDA , heightA, (Abytes)/1024.0/1024.0);
    printf ("info: allocate host mem B %zu x %zu (%6.2f MB)\n", LDB , heightB,(Bbytes)/1024.0/1024.0);
    printf ("info: allocate host mem C %zu x %zu (%6.2f MB)\n", LDC , heightC,(Cbytes)/1024.0/1024.0);
    printf ("info: allocate host mem C %zu x %zu golden (%6.2f MB)\n", LDC , heightC,(Cbytes)/1024.0/1024.0);
    A_h = (float*)malloc(Abytes);
    CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess );
    B_h = (float*)malloc(Bbytes);
    CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess );
    C_h = (float*)malloc(Cbytes);
    CHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess );
    C_Gold = (float*)malloc(Cbytes);
    CHECK(C_Gold == 0 ? hipErrorOutOfMemory : hipSuccess );
    memset (C_Gold, 0, Cbytes);
    // Fill A and B
    for (size_t i=0; i<heightA; i++)
        for (size_t j=0; j<widthA; j++)
        {
            //A_h[i*LDA+j] = 1.0f;
            A_h[i*LDA+j] = 0.25f - 0.5*j/widthA + 0.1f*i/heightA;
        }
        
    for (size_t i=0; i<heightB; i++)
        for (size_t j=0; j<widthB; j++)
        {
            //B_h[i*LDB+j] = 1.0f;
            B_h[i*LDB+j] = 0.12f + 0.03*i/heightB- 0.12f*j/widthB;
        }
    printf ("info: calculate result in Host\n");
    if (b_transA & b_transB){
        for (size_t i=0; i<N; i+=DBG_CHECK_STEP){
            for (size_t k=0; k<K; k++){
                for (size_t j=0; j<M; j+=DBG_CHECK_STEP)
                {
                    C_Gold[i*LDC + j] += A_h[k*LDA + j] * B_h[i*LDB + k];
                }
            }
        }
    } else if (b_transA & !b_transB){
        for (size_t i=0; i<N; i+=DBG_CHECK_STEP){
            for (size_t j=0; j<M; j+=DBG_CHECK_STEP){
                for (size_t k=0; k<K; k++)
                {
                    C_Gold[i*LDC + j] += A_h[k*LDA + j] * B_h[k*LDB + i];
                }
            }
        }
    } else if (!b_transA & b_transB){
        for (size_t i=0; i<N; i+=DBG_CHECK_STEP){
            for (size_t k=0; k<K; k++){
                for (size_t j=0; j<M; j+=DBG_CHECK_STEP)
                {
                    C_Gold[i*LDC + j] += A_h[j*LDA + k] * B_h[i*LDB + k];
                }
            }
        }
    } else if (!b_transA & !b_transB){
        for (size_t i=0; i<N; i+=DBG_CHECK_STEP){
            for (size_t k=0; k<K; k++){
                for (size_t j=0; j<M; j+=DBG_CHECK_STEP)
                {
                    C_Gold[i*LDC + j] += A_h[j*LDA + k] * B_h[k*LDB + i];
                }
            }
        }
    }
    //for (size_t i=0; i<N; i++){
    //    for (size_t j=0; j<M; j++){
    //        printf("C_Gold[i*LDC + j] = %f\n",C_Gold[i*LDC + j]);
    //    }
    //}
    printf("M,N,K=%zu,%zu,%zu\n",M,N,K);
    printf ("info: allocate A device mem (%6.2f MB)\n", (M*K*sizeof(float))/1024.0/1024.0);
    CHECK(hipMalloc(&A_d, (M*K*sizeof(float))));
    printf ("info: allocate B device mem (%6.2f MB)\n", (N*K*sizeof(float))/1024.0/1024.0);
    CHECK(hipMalloc(&B_d, (N*K*sizeof(float))));
    printf ("info: allocate C device mem (%6.2f MB)\n", (M*N*sizeof(float))/1024.0/1024.0);
    CHECK(hipMalloc(&C_d, (M*N*sizeof(float))));
    hipStream_t stream;
        
    CHECK (hipStreamCreate(&stream));
    
    printf ("info: copy Host2Device with hipMemcpy2D\n");
    CHECK ( hipMemcpy2DAsync(A_d, widthA * sizeof(float), A_h, LDA * sizeof(float), widthA * sizeof(float), heightA, hipMemcpyHostToDevice,stream));
    CHECK ( hipMemcpy2DAsync(B_d, widthB * sizeof(float), B_h, LDB * sizeof(float), widthB * sizeof(float), heightB, hipMemcpyHostToDevice,stream));

    //debug
    //CHECK ( hipMemcpy2DAsync(A_h, LDA * sizeof(float), A_d, widthA * sizeof(float), widthA * sizeof(float), heightA, hipMemcpyDeviceToHost,stream));
    //CHECK ( hipMemcpy2DAsync(A_h, LDB * sizeof(float), B_d, widthB * sizeof(float), widthB * sizeof(float), heightB, hipMemcpyDeviceToHost,stream));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    dim3 gridDim((M+15)/16,(N+15)/16,1);
    dim3 blockDim(16,16,1);

    float mS = 0;
    hipEvent_t start, stop;

    CHECK (hipEventCreate(&start));
    CHECK (hipEventCreate(&stop));
    
    if (b_transA & b_transB){
        printf ("info: launch 'vector_sgemmNN' kernel\n");
    } else if (b_transA & !b_transB){
        printf ("info: launch 'vector_sgemmNT' kernel\n");
    } else if (!b_transA & b_transB){
        printf ("info: launch 'vector_sgemmTN' kernel\n");
    } else if (!b_transA & !b_transB){
        printf ("info: launch 'vector_sgemmTT' kernel\n");
    }
    
    hipEventRecord(start);
    if (b_transA & b_transB){
        hipLaunchKernelGGL(vector_sgemmNN, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } else if (b_transA & !b_transB){
        hipLaunchKernelGGL(vector_sgemmNT, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } else if (!b_transA & b_transB){
        hipLaunchKernelGGL(vector_sgemmTN, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } else if (!b_transA & !b_transB){
        hipLaunchKernelGGL(vector_sgemmTT, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } 
    hipEventRecord(stop);
    CHECK (hipStreamSynchronize(stream));
    CHECK (hipEventElapsedTime(&mS, start, stop));
    
    printf ("info: launch cost %f ms\n", mS);
    
    printf ("info: copy Device2Host\n");
    //CHECK ( hipMemcpy(C_h, C_d, Cbytes, hipMemcpyDeviceToHost));
    //debug
    //CHECK ( hipMemcpy2DAsync(C_d, M * sizeof(float), C_Gold, LDC * sizeof(float), M * sizeof(float) ,N, hipMemcpyHostToDevice, stream));
    
    CHECK ( hipMemcpy2DAsync(C_h, LDC * sizeof(float), C_d, M * sizeof(float), M * sizeof(float) ,N, hipMemcpyDeviceToHost, stream));
    CHECK (hipStreamSynchronize(stream));
    
    printf ("info: compare result\n");
    for (size_t i=0; i<N; i+=DBG_CHECK_STEP){
        bool b_error = false;
        for (size_t j=0; j<M; j+=DBG_CHECK_STEP){
            //printf("[%u,%u] gold = %f, result = %f\n",i,j,C_Gold[i*N+j],C_h[i*N+j]);
            if(abs(C_h[i*LDC + j] - C_Gold[i*LDC + j]) > (0.005*abs(C_h[i*LDC + j]))){
                printf("[%zu,%zu] mismatch, gold = %f, result = %f\n",j,i,C_Gold[i*LDC + j],C_h[i*LDC + j]);
                b_error = true;
                break;
            }
        }
        if(b_error)
            break;
    }    

    //reset memory
    hipMemset(C_d,0, (M*N*sizeof(float)));
    //reset block size
    //gridDim = dim3((M+255)/256,N,1);
    //blockDim = dim3(256,1,1);
    
    if (b_transA & b_transB){
        printf ("info: launch 'vector_sgemmNN_fast' kernel with shared memory\n");
    } else if (b_transA & !b_transB){
        printf ("info: launch 'vector_sgemmNT_fast' kernel with shared memory\n");
    } else if (!b_transA & b_transB){
        printf ("info: launch 'vector_sgemmTN_fast' kernel with shared memory\n");
    } else if (!b_transA & !b_transB){
        printf ("info: launch 'vector_sgemmTT_fast' kernel with shared memory\n");
    }

    hipEventRecord(start);
    if (b_transA & b_transB){
        hipLaunchKernelGGL(vector_sgemmNN_fast, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } else if (b_transA & !b_transB){
        hipLaunchKernelGGL(vector_sgemmNT_fast, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } else if (!b_transA & b_transB){
        hipLaunchKernelGGL(vector_sgemmTN_fast, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } else if (!b_transA & !b_transB){
        hipLaunchKernelGGL(vector_sgemmTT_fast, gridDim, blockDim, 0, stream, C_d, A_d, B_d, M, K, N);
    } 
    hipEventRecord(stop);
    CHECK (hipStreamSynchronize(stream));
    CHECK (hipEventElapsedTime(&mS, start, stop));
    
    printf ("info: launch with shared memory cost %f ms\n", mS);
    
    CHECK ( hipMemcpy2DAsync(C_h, LDC * sizeof(float), C_d, M * sizeof(float), M * sizeof(float) ,N, hipMemcpyDeviceToHost, stream));
    CHECK (hipStreamSynchronize(stream));
    
    printf ("info: compare result\n");
    for (size_t i=0; i<N; i+=DBG_CHECK_STEP){
        bool b_error = false;
        for (size_t j=0; j<M; j+=DBG_CHECK_STEP){
            //printf("[%u,%u] gold = %f, result = %f\n",i,j,C_Gold[i*N+j],C_h[i*N+j]);
            if(abs(C_h[i*LDC + j] - C_Gold[i*LDC + j]) > (0.005*abs(C_h[i*LDC + j]))){
                printf("[%zu,%zu] mismatch, gold = %f, result = %f\n",j,i,C_Gold[i*LDC + j],C_h[i*LDC + j]);
                b_error = true;
                break;
            }
        }
        if(b_error)
            break;
    }   
    
    CHECK(hipFree(A_d));
    CHECK(hipFree(B_d));
    CHECK(hipFree(C_d));

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_Gold);


    printf ("PASSED!\n");
}

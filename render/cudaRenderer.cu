#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "cycleTimer.h"

#define sq(x) (x) * (x)
#define TPB 1024
#define SQRT_TPB 32  // TODO: play with this
#define CHUNK 64

__constant__ int cuda_wc;
__constant__ int cuda_hc;
__constant__ float cuda_inv_w;
__constant__ float cuda_inv_h;
__constant__ int* cuda_chunks;
__constant__ int* cuda_prefix;
__constant__ int* cuda_idxs;

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixelSnowflake(int circleIndex, float norm_dist, float z, float4& cur_rgba) {
    float3 rgb;
    float alpha;

    // simple: each circle has an assigned color
    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    rgb = lookupColor(norm_dist);

    float maxAlpha = .6f + .4f * (1.f-z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * norm_dist * norm_dist);

    float oneMinusAlpha = 1.f - alpha;

    cur_rgba.x = alpha * rgb.x + oneMinusAlpha * cur_rgba.x;
    cur_rgba.y = alpha * rgb.y + oneMinusAlpha * cur_rgba.y;
    cur_rgba.z = alpha * rgb.z + oneMinusAlpha * cur_rgba.z;
    cur_rgba.w = alpha + cur_rgba.w;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float4& cur_rgba) {
    float3 rgb;
    float alpha;

    int index3 = 3 * circleIndex;
    rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    alpha = .5f;

    float oneMinusAlpha = 1.f - alpha;

    cur_rgba.x = alpha * rgb.x + oneMinusAlpha * cur_rgba.x;
    cur_rgba.y = alpha * rgb.y + oneMinusAlpha * cur_rgba.y;
    cur_rgba.z = alpha * rgb.z + oneMinusAlpha * cur_rgba.z;
    cur_rgba.w = alpha + cur_rgba.w;
}

__global__ void fillChunks() {
    int circle = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = cuConstRendererParams.numCircles;
    if (circle >= nc || chunk >= cuda_wc * cuda_hc)
        return;
    float3 p = *(float3*)(&cuConstRendererParams.position[3*circle]);
    float r = cuConstRendererParams.radius[circle];
    int x = (chunk % cuda_wc) * CHUNK;
    int y = (chunk / cuda_wc) * CHUNK;
    float min_x = cuda_inv_w * (static_cast<float>(x) + 0.5f);
    float min_y = cuda_inv_h * (static_cast<float>(y) + 0.5f);
    float max_x = cuda_inv_w * (static_cast<float>(x + CHUNK - 1) + 0.5f);
    float max_y = cuda_inv_h * (static_cast<float>(y + CHUNK - 1) + 0.5f);
    // TODO: write better versions of these (maybe use bbox instead of circle?) or think more abt geo
    if (circleInBoxConservative(p.x, p.y, r, min_x, max_x, max_y, min_y) &&
        circleInBox(p.x, p.y, r, min_x, max_x, max_y, min_y))
        cuda_chunks[chunk*nc+circle] = 1;
}

__global__ void getIdxs() {
    int circle = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = cuConstRendererParams.numCircles;
    if (circle >= nc || chunk >= cuda_wc * cuda_hc)
        return;
    int i = chunk * nc + circle;
    if (cuda_chunks[i])
        cuda_idxs[chunk*nc+cuda_prefix[i]] = circle;
}

__global__ void renderPixelsSnowflakes() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = cuConstRendererParams.imageWidth
    if (x >= w || y >= cuConstRendererParams.imageHeight)
        return;
    int chunk = y / CHUNK * cuda_wc + x / CHUNK;
    int i = y * w + x;
    float4 rgba = *(float4*)(&cuConstRendererParams.imageData[4*i]);
    float2 center = make_float2(cuda_inv_w * (static_cast<float>(x) + 0.5f),
                                cuda_inv_h * (static_cast<float>(y) + 0.5f));
    int nc = cuConstRendererParams.numCircles;
    int chunk_circles = cuda_prefix[chunk*nc+nc-1] + cuda_chunks[chunk*nc+nc-1];
    int k = 3 - (chunk * nc + 3) % 4; // address alignment
    for (int j = 0; j < chunk_circles; j++) {
        if (k <= j && j <= chunk_circles - 4) {
            int4 circles = *(int4*)(&cuda_idxs[chunk*nc+j]);
            for (int circle : {circles.x, circles.y, circles.z, circles.w}) {
                float3 p = *(float3*)(&cuConstRendererParams.position[3*circle]);
                float r = cuConstRendererParams.radius[circle];
                float sq_dist = sq(p.x - center.x) + sq(p.y - center.y);
                if (sq_dist <= sq(r))
                    shadePixelSnowflake(circle, sqrt(sq_dist) / r, p.z, rgba);
            }
            j += 3;
        }
        else {
            int circle = cuda_idxs[chunk*nc+j];
            float3 p = *(float3*)(&cuConstRendererParams.position[3*circle]);
            float r = cuConstRendererParams.radius[circle];
            float sq_dist = sq(p.x - center.x) + sq(p.y - center.y);
            if (sq_dist <= sq(r))
                shadePixelSnowflake(circle, sqrt(sq_dist) / r, p.z, rgba);
        }
    }
    *(float4*)(&cuConstRendererParams.imageData[4*i]) = rgba;
}

__global__ void renderPixel() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = cuConstRendererParams.imageWidth
    if (x >= w || y >= cuConstRendererParams.imageHeight)
        return;
    int chunk = y / CHUNK * cuda_wc + x / CHUNK;
    int i = y * w + x;
    float4 rgba = *(float4*)(&cuConstRendererParams.imageData[4*i]);
    float2 center = make_float2(cuda_inv_w * (static_cast<float>(x) + 0.5f),
                                cuda_inv_h * (static_cast<float>(y) + 0.5f));
    int nc = cuConstRendererParams.numCircles;
    int chunk_circles = cuda_prefix[chunk*nc+nc-1] + cuda_chunks[chunk*nc+nc-1];
    int k = 3 - (chunk * nc + 3) % 4; // address alignment
    for (int j = 0; j < chunk_circles; j++) {
        if (k <= j && j <= chunk_circles - 4) {
            int4 circles = *(int4*)(&cuda_idxs[chunk*nc+j]);
            for (int circle : {circles.x, circles.y, circles.z, circles.w}) {
                float3 p = *(float3*)(&cuConstRendererParams.position[3*circle]);
                float r = cuConstRendererParams.radius[circle];
                if (sq(p.x - center.x) + sq(p.y - center.y) <= sq(r))
                    shadePixel(circle, rgba);
            }
            j += 3;
        }
        else {
            int circle = cuda_idxs[chunk*nc+j];
            float3 p = *(float3*)(&cuConstRendererParams.position[3*circle]);
            float r = cuConstRendererParams.radius[circle];
            if (sq(p.x - center.x) + sq(p.y - center.y) <= sq(r))
                shadePixel(circle, rgba);
        }
    }
    *(float4*)(&cuConstRendererParams.imageData[4*i]) = rgba;
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;

    wc = 0;
    hc = 0;
    chunks = NULL;
    prefix = NULL;
    idxs = NULL;
    frame = 0;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (chunks) {
        cudaFree(chunks);
        cudaFree(prefix);
        cudaFree(idxs);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    wc = (image->width + CHUNK - 1) / CHUNK;
    hc = (image->height + CHUNK - 1) / CHUNK;
    cudaMemcpyToSymbol(cuda_wc, &wc, sizeof(int));
    cudaMemcpyToSymbol(cuda_hc, &hc, sizeof(int));

    float inv_w = 1.f / image->width;
    float inv_h = 1.f / image->height;
    cudaMemcpyToSymbol(cuda_inv_w, &inv_w, sizeof(float));
    cudaMemcpyToSymbol(cuda_inv_h, &inv_h, sizeof(float));

    cudaMalloc(&chunks, wc * hc * numCircles * sizeof(int));
    cudaMemset(chunks, 0, wc * hc * numCircles * sizeof(int));
    cudaMalloc(&prefix, wc * hc * numCircles * sizeof(int));
    cudaMalloc(&idxs, wc * hc * numCircles * sizeof(int));
    for (int i = 0; i < wc * hc; i++) {
        cudaMemset(idxs + i * numCircles, i & 1, numCircles * sizeof(int));
    }
    cudaMemcpyToSymbol(cuda_chunks, &chunks, sizeof(int*));
    cudaMemcpyToSymbol(cuda_prefix, &prefix, sizeof(int*));
    cudaMemcpyToSymbol(cuda_idxs, &idxs, sizeof(int*));

    frame = 0;
    dim3 block_dim(SQRT_TPB, SQRT_TPB);
    dim3 chunk_grid_dim((numCircles + SQRT_TPB - 1) / SQRT_TPB, (wc * hc + SQRT_TPB - 1) / SQRT_TPB);
    fillChunks<<<chunk_grid_dim, block_dim>>>();
    thrust::exclusive_scan_by_key(thrust::device, idxs, idxs + wc * hc * numCircles, chunks, prefix);
    getIdxs<<<chunk_grid_dim, block_dim>>>();
    cudaDeviceSynchronize();
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    if (frame)
        return;
    dim3 block_dim(SQRT_TPB, SQRT_TPB);
    dim3 pixel_grid_dim((image->width + SQRT_TPB - 1) / SQRT_TPB, (image->height + SQRT_TPB - 1) / SQRT_TPB);
    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME)
        renderPixelsSnowflakes<<<pixel_grid_dim, block_dim>>>();
    else
        renderPixel<<<pixel_grid_dim, block_dim>>>();
    frame++;
}

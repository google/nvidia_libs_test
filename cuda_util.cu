/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda/include/cuda_fp16.h"
#include "cuda/include/cuda_runtime.h"
#include "cuda/include/curand_kernel.h"
// cuda_util.h is intentionally not included to simplify compiling this file.

namespace nvidia_libs_test {
namespace {
const int kGridDim = 16;
const int kBlockDim = 128;

template <typename DstT, typename SrcT>
struct ValueConverter {
  __device__ DstT operator()(const SrcT& value) const {
    return static_cast<DstT>(scale * value);
  }
  double scale;
};

template <typename SrcT>
struct ValueConverter<__half, SrcT> {
  __device__ __half operator()(const SrcT& value) const {
    return __float2half(static_cast<float>(scale * value));
  }
  double scale;
};

template <typename DstT>
struct ValueConverter<DstT, __half> {
  __device__ DstT operator()(const __half& value) const {
    return static_cast<DstT>(scale * __half2float(value));
  }
  double scale;
};

template <typename DstT, typename SrcT>
__global__ void ConvertDeviceDataKernel(double scale, DstT* dst,
                                        const SrcT* src, int num_elements) {
  size_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  ValueConverter<DstT, SrcT> convert = {scale};
  for (size_t i = thread_idx; i < num_elements; i += gridDim.x * blockDim.x) {
    dst[i] = convert(src[i]);
  }
}

template <typename DstT, typename SrcT>
void ConvertDeviceDataImpl(double scale, DstT* dst, const SrcT* src,
                           size_t num_elements) {
  ConvertDeviceDataKernel<<<kGridDim, kBlockDim>>>(scale, dst, src,
                                                   num_elements);
}

__global__ void InitializeCurandStateKernel(curandState* states, size_t seed) {
  size_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, thread_idx, /*offset=*/0, states + thread_idx);
}

__device__ void GenerateUniform(__half* dst, float scale, float bias,
                                curandState* state) {
  *dst = __float2half(curand_uniform(state) * scale + bias);
}
__device__ void GenerateUniform(float* dst, float scale, float bias,
                                curandState* state) {
  *dst = curand_uniform(state) * scale + bias;
}
__device__ void GenerateUniform(double* dst, double scale, double bias,
                                curandState* state) {
  *dst = curand_uniform_double(state) * scale + bias;
}

template <typename T>
__global__ void InitializeDeviceDataKernel(T* ptr, size_t num_elements,
                                           double scale, double bias,
                                           curandState* states) {
  size_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  curandState state = states[thread_idx];
  for (size_t i = thread_idx; i < num_elements; i += gridDim.x * blockDim.x) {
    GenerateUniform(ptr + i, scale, bias, &state);
  }
  states[thread_idx] = state;
}

template <typename T>
void InitializeDeviceDataImpl(T* ptr, size_t num_elements, double lower,
                              double upper, void* state) {
  InitializeDeviceDataKernel<<<kGridDim, kBlockDim>>>(
      ptr, num_elements, upper - lower, lower,
      static_cast<curandState*>(state));
}

}  // namespace

void ConvertDeviceData(double scale, double* dst, const float* src,
                       size_t num_elements) {
  ConvertDeviceDataImpl(scale, dst, src, num_elements);
}
void ConvertDeviceData(double scale, float* dst, const double* src,
                       size_t num_elements) {
  ConvertDeviceDataImpl(scale, dst, src, num_elements);
}
void ConvertDeviceData(double scale, __half* dst, const float* src,
                       size_t num_elements) {
  ConvertDeviceDataImpl(scale, dst, src, num_elements);
}
void ConvertDeviceData(double scale, float* dst, const __half* src,
                       size_t num_elements) {
  ConvertDeviceDataImpl(scale, dst, src, num_elements);
}
void ConvertDeviceData(double scale, __half* dst, const double* src,
                       size_t num_elements) {
  ConvertDeviceDataImpl(scale, dst, src, num_elements);
}
void ConvertDeviceData(double scale, double* dst, const __half* src,
                       size_t num_elements) {
  ConvertDeviceDataImpl(scale, dst, src, num_elements);
}

namespace detail {
size_t GetCurandStateSize() {
  return kBlockDim * kGridDim * sizeof(curandState);
}

void InitializeCurandState(void* state, size_t seed) {
  InitializeCurandStateKernel<<<kGridDim, kBlockDim>>>(
      static_cast<curandState*>(state), seed);
}

void InitializeDeviceData(float* ptr, size_t num_elements, double lower,
                          double upper, void* state) {
  InitializeDeviceDataImpl(ptr, num_elements, lower, upper, state);
}

void InitializeDeviceData(double* ptr, size_t num_elements, double lower,
                          double upper, void* state) {
  InitializeDeviceDataImpl(ptr, num_elements, lower, upper, state);
}

void InitializeDeviceData(__half* ptr, size_t num_elements, double lower,
                          double upper, void* state) {
  InitializeDeviceDataImpl(ptr, num_elements, lower, upper, state);
}
}  // namespace detail

}  // namespace nvidia_libs_test

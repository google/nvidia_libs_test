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

#ifndef NVIDIA_LIBS_TEST_CUDA_UTIL_H_
#define NVIDIA_LIBS_TEST_CUDA_UTIL_H_

#include <cstddef>
#include <memory>

// host_defines.h with __align__ macro has to be included before cuda_fp16.h.
#include "cuda/include/host_defines.h"

#include "cuda/extras/CUPTI/include/cupti_result.h"
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_fp16.h"
#include "cuda/include/driver_types.h"
#include "status.h"

// Function overloads converting CUDA API status codes to Status so that CUDA
// API calls can be wrapped in IsError or CheckOk.
//
// Functions to create and initialize device data.

namespace nvidia_libs_test {

// Returns Status from CUDA Device API status.
Status GetStatus(CUresult result);
// Returns Status from CUDA Runtime API status.
Status GetStatus(cudaError_t);
// Returns Status from CUPTI status.
Status GetStatus(CUptiResult);

namespace detail {
struct HostMemoryDeleter {
  void operator()(void*) const;
};
}  // namespace detail

// RAII wrapper for host memory.
using HostMemory = std::unique_ptr<void, detail::HostMemoryDeleter>;

// Represents memory allocated on device. Interface is similar to unique_ptr.
class DeviceMemory {
 public:
  // Construct from nullptr. Use AllocateDeviceMemory for non-null instance.
  explicit DeviceMemory(std::nullptr_t);
  ~DeviceMemory();
  DeviceMemory(DeviceMemory&&) noexcept;
  DeviceMemory& operator=(DeviceMemory&&);

  void* get() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  friend StatusOr<DeviceMemory> AllocateDeviceMemory(size_t);
  void* ptr_;
  size_t size_;
};

namespace detail {
size_t GetCurandStateSize();
void InitializeCurandState(void* state, size_t seed);
void InitializeDeviceData(float* ptr, size_t num_elements, double lower,
                          double upper, void* state);
void InitializeDeviceData(double* ptr, size_t num_elements, double lower,
                          double upper, void* state);
void InitializeDeviceData(__half* ptr, size_t num_elements, double lower,
                          double upper, void* state);
}  // namespace detail

// Random number generator for device data.
class RandomGenerator {
 public:
  RandomGenerator(size_t seed);

 private:
  template <typename T>
  friend StatusOr<DeviceMemory> CreateDeviceData(
      size_t num_elements, double lower, double upper,
      const RandomGenerator& rand_gen);

  DeviceMemory state_;
};

// Allocates size bytes of host memory.
StatusOr<HostMemory> AllocateHostMemory(size_t size);

// Returns device pointer equivalent of host_ptr.
void* GetDevicePointer(const HostMemory& host_ptr);

// Allocates size bytes of device memory.
StatusOr<DeviceMemory> AllocateDeviceMemory(size_t size);

// Fill the device memory with garbage data.
void FillWithNaNs(const DeviceMemory& mem);

// Returns amount of bytes allocated through AllocateDeviceMemory.
size_t GetAllocatedDeviceMemoryBytes();

// Copies size bytes from src to dst.
Status CopyDeviceMemory(const DeviceMemory& dst, const DeviceMemory& src,
                        size_t size);

// Creates array of num_elements of type T containing random data.
template <typename T>
StatusOr<DeviceMemory> CreateDeviceData(size_t num_elements, double lower,
                                        double upper,
                                        const RandomGenerator& rand_gen) {
  CHECK_LE(lower, upper);
  auto result = AllocateDeviceMemory(num_elements * sizeof(T));
  RETURN_IF_ERROR_STATUS(result.status());
  T* ptr = static_cast<T*>(result.ValueOrDie().get());
  detail::InitializeDeviceData(ptr, num_elements, lower, upper,
                               rand_gen.state_.get());
  return result;
}

// Scales and converts a device array of num_elements from one type to another.
void ConvertDeviceData(double scale, double* dst, const float* src,
                       size_t num_elements);
void ConvertDeviceData(double scale, float* dst, const double* src,
                       size_t num_elements);
void ConvertDeviceData(double scale, __half* dst, const float* src,
                       size_t num_elements);
void ConvertDeviceData(double scale, float* dst, const __half* src,
                       size_t num_elements);
void ConvertDeviceData(double scale, __half* dst, const double* src,
                       size_t num_elements);
void ConvertDeviceData(double scale, double* dst, const __half* src,
                       size_t num_elements);

// Returns whether the device is at least 'sm_<major>.<minor>'.
bool DeviceHasAtLeastComputeCapability(int major, int minor);

// Returns whether the device supports reduced precision math (fp16). Returns
// true for sm_61, even though it has very low reduced precision throughput.
bool DeviceSupportsReducedPrecision();

// Returns whether the GPU has tensor cores for fast mixed precision math.
bool DeviceSupportsTensorOpMath();

// Resets the device and the runtime error status.
void ResetDevice();

}  // namespace nvidia_libs_test
#endif  // NVIDIA_LIBS_TEST_CUDA_UTIL_H_

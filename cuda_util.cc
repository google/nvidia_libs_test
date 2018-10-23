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

#include "cuda_util.h"
#include "cuda/include/cuda_runtime.h"

#include <atomic>

namespace nvidia_libs_test {

Status GetStatus(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return OkStatus();
  }
  const char* str = nullptr;
  CHECK(cuGetErrorString(result, &str) == CUDA_SUCCESS);
  return ErrorStatus("CUDA Driver API error '") << str << "'";
}

Status GetStatus(cudaError_t error) {
  if (error == cudaSuccess) {
    return OkStatus();
  }
  // Reset CUDA runtime status because we can expect the user to handle the
  // returned error.
  cudaGetLastError();
  const char* str = cudaGetErrorString(error);
  return ErrorStatus("CUDA Runtime API error '") << str << "'";
}

Status GetStatus(CUptiResult result) {
  if (result == CUPTI_SUCCESS) {
    return OkStatus();
  }
  const char* str = nullptr;
  CHECK(cuptiGetResultString(result, &str) == CUPTI_SUCCESS);
  return ErrorStatus("CUPTI error '") << str << "'";
}

RandomGenerator::RandomGenerator(size_t seed)
    : state_(std::move(
          AllocateDeviceMemory(detail::GetCurandStateSize()).ValueOrDie())) {
  detail::InitializeCurandState(state_.get(), seed);
}

namespace detail {
void HostMemoryDeleter::operator()(void* ptr) const {
  CHECK_OK_STATUS(GetStatus(cudaFreeHost(ptr)));
}
}  // namespace detail

DeviceMemory::DeviceMemory(std::nullptr_t) : ptr_(nullptr), size_(0) {}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
  other.ptr_ = nullptr;
  other.size_ = 0;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) {
  if (this != &other) {
    CHECK_EQ(ptr_ == nullptr || ptr_ != other.ptr_, true);
    if(ptr_) {
	CHECK_OK_STATUS(GetStatus(cudaFree(ptr_)));
    }
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

namespace {
std::atomic<std::size_t> allocated_device_memory_bytes{0};
}  // namespace

DeviceMemory::~DeviceMemory() {
  CHECK_GE(allocated_device_memory_bytes, size_);
  allocated_device_memory_bytes -= size_;
  if(ptr_) {
      CHECK_OK_STATUS(GetStatus(cudaFree(ptr_)));
  }
}

StatusOr<HostMemory> AllocateHostMemory(size_t size) {
  void* result;
  RETURN_IF_ERROR_STATUS(GetStatus(cudaMallocHost(&result, size)));
  return HostMemory(result);
}

void* GetDevicePointer(const HostMemory& host_ptr) {
  void* dev_ptr;
  CHECK_OK_STATUS(
      GetStatus(cudaHostGetDevicePointer(&dev_ptr, host_ptr.get(), 0)));
  return dev_ptr;
}

StatusOr<DeviceMemory> AllocateDeviceMemory(size_t size) {
  DeviceMemory result(nullptr);
  cudaError_t error = cudaMalloc(&result.ptr_, size);
  auto status = GetStatus(error);
  if (error == cudaErrorMemoryAllocation) {
    size_t free = 0;
    size_t total = 0;
    CHECK_OK_STATUS(GetStatus(cudaMemGetInfo(&free, &total)));
    status << "\nbytes requested: " << size
           << "\nbytes allocated: " << allocated_device_memory_bytes
           << "\nbytes free: " << free << "\nbytes total: " << total;
  }
  RETURN_IF_ERROR_STATUS(status);
  result.size_ = size;
  allocated_device_memory_bytes += size;
  return std::move(result);
}

void FillWithNaNs(const DeviceMemory& mem) {
  CHECK_OK_STATUS(GetStatus(cudaMemset(mem.get(), 0xff, mem.size())));
}

size_t GetAllocatedDeviceMemoryBytes() { return allocated_device_memory_bytes; }

Status CopyDeviceMemory(const DeviceMemory& dst, const DeviceMemory& src,
                        size_t size) {
  RETURN_IF_ERROR_STATUS(GetStatus(
      cudaMemcpy(dst.get(), src.get(), size, cudaMemcpyDeviceToDevice)));
  return OkStatus();
}

void ResetDevice() {
  cudaGetLastError();  // Reset CUDA runtime status.
  CHECK_OK_STATUS(GetStatus(cudaDeviceReset()));
}

namespace {
cudaDeviceProp GetDeviceProperties() {
  int device = 0;
  CHECK_OK_STATUS(GetStatus(cudaGetDevice(&device)));
  cudaDeviceProp props;
  CHECK_OK_STATUS(GetStatus(cudaGetDeviceProperties(&props, device)));
  return props;
}
}  // namespace

bool DeviceHasAtLeastComputeCapability(int major, int minor) {
  static cudaDeviceProp props = GetDeviceProperties();
  return props.major > major || (props.major == major && props.minor >= minor);
}

bool DeviceSupportsReducedPrecision() {
  return DeviceHasAtLeastComputeCapability(5, 3);
}

bool DeviceSupportsTensorOpMath() {
  return DeviceHasAtLeastComputeCapability(7, 0);
}
}  // namespace nvidia_libs_test

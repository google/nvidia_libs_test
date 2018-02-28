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

#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "cuda/include/cuda_runtime.h"

namespace nvidia_libs_test {

uint_fast32_t GetRandomSeed() {
  return testing::FLAGS_gtest_random_seed;
}

namespace {

template <typename T>
struct DeviceTypeTraits {
  using HostType = T;
};

// __half data is converted to float when copying to host.
template <>
struct DeviceTypeTraits<__half> {
  using HostType = float;
};

template <typename T>
Status CopyToHost(T* dst, const T* src, size_t num_elements) {
  return GetStatus(cudaMemcpyAsync(dst, src, num_elements * sizeof(T),
                                   cudaMemcpyDeviceToHost));
}

Status CopyToHost(float* dst, const __half* src, size_t num_elements) {
  float* dev_dst = nullptr;
  RETURN_IF_ERROR_STATUS(GetStatus(cudaHostGetDevicePointer(&dev_dst, dst, 0)));
  ConvertDeviceData(1.0, dev_dst, src, num_elements);
  return OkStatus();
}

template <typename DeviceType>
Status DeviceDataEqual(const DeviceType* dev_first,
                       const DeviceType* dev_second, size_t num_elements,
                       double tolerance) {
  using HostType = typename DeviceTypeTraits<DeviceType>::HostType;

  unsigned num_diffs_to_report = 8;
  unsigned buffer_size_in_bytes = 1u << 27;  // 128MB
  size_t num_buffer_elements =
      std::min(num_elements, buffer_size_in_bytes / sizeof(DeviceType));

  if ((!dev_first || !dev_second) && num_elements) {
    return ErrorStatus("nullptr argument");
  }

  struct HostPointerDeleter {
    void operator()(HostType* ptr) {
      CHECK_OK_STATUS(GetStatus(cudaFreeHost(ptr)));
    }
  };
  using HostPointer = std::unique_ptr<HostType, HostPointerDeleter>;

  auto allocate_host_data = [](size_t num_elements) -> StatusOr<HostPointer> {
    HostType* ptr = nullptr;
    RETURN_IF_ERROR_STATUS(
        GetStatus(cudaMallocHost(&ptr, num_elements * sizeof(HostType))));
    return HostPointer{ptr};
  };

  auto buffer_or = allocate_host_data(2 * num_buffer_elements);
  RETURN_IF_ERROR_STATUS(buffer_or.status());
  HostType* buf_first = buffer_or.ValueOrDie().get();
  HostType* buf_second = buf_first + num_buffer_elements;

  struct Diff {
    size_t index;
    HostType first, second;
    double error;
  };
  auto greater_error = [](const Diff& left, const Diff& right) {
    if (std::isunordered(left.error, right.error)) {
      return std::isnan(left.error);
    }
    return left.error > right.error;
  };
  std::vector<Diff> heap;
  heap.reserve(num_diffs_to_report + 1);

  size_t num_failures = 0;
  for (size_t i = 0; i < num_elements; i += num_buffer_elements) {
    size_t n = std::min(num_buffer_elements, num_elements - i);
    RETURN_IF_ERROR_STATUS(CopyToHost(buf_first, dev_first, n));
    RETURN_IF_ERROR_STATUS(CopyToHost(buf_second, dev_second, n));
    RETURN_IF_ERROR_STATUS(GetStatus(cudaDeviceSynchronize()));

    for (size_t j = 0; j < n; ++j) {
      HostType first = buf_first[j];
      HostType second = buf_second[j];
      double difference = std::abs(0.0 + first - second);
      // Relative difference for huge values, absolute difference for tiny
      // values.
      double denominator = std::max(std::abs(first), std::abs(second)) + 1.0;
      if (difference <= tolerance * denominator) {
        continue;
      }
      Diff diff{i + j, first, second, difference / denominator};
      if (heap.size() < num_diffs_to_report ||
          greater_error(diff, heap.front())) {
        heap.push_back(diff);
        std::push_heap(heap.begin(), heap.end(), greater_error);
      }
      while (heap.size() > num_diffs_to_report) {
        std::pop_heap(heap.begin(), heap.end(), greater_error);
        heap.pop_back();
      }
      ++num_failures;
    }
  }

  if (num_failures == 0) {
    return OkStatus();
  }

  std::ostringstream oss;
  oss << num_failures << " elements differ more than " << tolerance
      << ". Largest differences:";
  std::sort_heap(heap.begin(), heap.end(), greater_error);
  for (const Diff& diff : heap) {
    oss << "\n[" << diff.index << "]: " << diff.first << " vs " << diff.second
        << ", error = " << diff.error;
  }
  return ErrorStatus(oss.str());
}
}  // namespace

Status DeviceDataEqual(const float* first, const float* second,
                       size_t num_elements, double tolerance) {
  return DeviceDataEqual<float>(first, second, num_elements, tolerance);
}

Status DeviceDataEqual(const double* first, const double* second,
                       size_t num_elements, double tolerance) {
  return DeviceDataEqual<double>(first, second, num_elements, tolerance);
}

Status DeviceDataEqual(const __half* first, const __half* second,
                       size_t num_elements, double tolerance) {
  return DeviceDataEqual<__half>(first, second, num_elements, tolerance);
}

::testing::AssertionResult IsOk(const Status& status) {
  if (status.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << status;
}

}  // namespace nvidia_libs_test

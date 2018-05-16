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

#ifndef NVIDIA_LIBS_TEST_CUDNN_UTIL_H_
#define NVIDIA_LIBS_TEST_CUDNN_UTIL_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/types/variant.h"
#include "cudnn/include/cudnn.h"
#include "cuda_util.h"
#include "cudnn.pb.h"

// Provides wrappers to perform cuDNN convolution ops defined by proto messages.

namespace nvidia_libs_test {

// Returns Status from cuDNN status.
Status GetStatus(cudnnStatus_t);

namespace detail {
struct CudnnHandleDeleter {
  void operator()(cudnnHandle_t) const;
};
struct TensorDescriptorDeleter {
  void operator()(cudnnTensorDescriptor_t) const;
};
struct FilterDescriptorDeleter {
  void operator()(cudnnFilterDescriptor_t) const;
};
struct ConvolutionDescriptorDeleter {
  void operator()(cudnnConvolutionDescriptor_t) const;
};
}  // namespace detail

// RAII wrappers for cuDNN handles.
using CudnnHandle = std::unique_ptr<cudnnContext, detail::CudnnHandleDeleter>;
using TensorDescriptor =
    std::unique_ptr<cudnnTensorStruct, detail::TensorDescriptorDeleter>;
using FilterDescriptor =
    std::unique_ptr<cudnnFilterStruct, detail::FilterDescriptorDeleter>;
using ConvolutionDescriptor =
    std::unique_ptr<cudnnConvolutionStruct,
                    detail::ConvolutionDescriptorDeleter>;

// Specifies one convolution algorithm.
using ConvolutionAlgo =
    absl::variant<cudnnConvolutionFwdAlgo_t, cudnnConvolutionBwdDataAlgo_t,
                  cudnnConvolutionBwdFilterAlgo_t>;

// Creates a cuDNN handle.
CudnnHandle CreateCudnnHandle();

// Creates cuDNN tensor descriptor from proto.
TensorDescriptor CreateTensorDescriptor(proto::TensorDescriptor proto);

// Returns true iff left and right are equal.
bool TensorDescriptorEqual(const TensorDescriptor& left,
                           const TensorDescriptor& right);

// Returns the number of elements in tensor, including strided elements.
size_t GetTensorNumElements(const TensorDescriptor& tensor);

// Returns the size of the tensor in bytes.
size_t GetTensorSizeInBytes(const TensorDescriptor& tensor);

// Returns the data type of tensor.
cudnnDataType_t GetTensorDataType(const TensorDescriptor& tensor);

// Allocates device memory for tensor and initializes it with random values.
StatusOr<DeviceMemory> CreateTensorData(const TensorDescriptor& tensor,
                                        const RandomGenerator& rand_gen);

// Blends the data from src tensor with dst tensor.
Status TransformTensor(const CudnnHandle& handle, double alpha, double beta,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data);

// Copies the data from src tensor to dst tensor.
Status TransformTensor(const CudnnHandle& handle,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data);

// Copies the data from src tensor to dst tensor. Also converts src data type
// to dst data type if required.
Status ConvertAndTransformTensor(const CudnnHandle& handle, double alpha,
                                 double beta, const TensorDescriptor& src_desc,
                                 const DeviceMemory& src_data,
                                 const TensorDescriptor& dst_desc,
                                 const DeviceMemory& dst_data);

string GetTensorDebugString(const TensorDescriptor& desc,
                            const DeviceMemory& data, bool print_values);

// Creates cuDNN filter descriptor from proto.
FilterDescriptor CreateFilterDescriptor(const proto::FilterDescriptor& proto);

// Returns true iff left and right are equal.
bool FilterDescriptorEqual(const FilterDescriptor& left,
                           const FilterDescriptor& right);

// Returns the number of elements in filter, including strided elements.
size_t GetFilterNumElements(const FilterDescriptor& filter);

// Returns the size of the filter in bytes.
size_t GetFilterSizeInBytes(const FilterDescriptor& filter);

// Returns the data type of filter.
cudnnDataType_t GetFilterDataType(const FilterDescriptor& filter);

// Allocates device memory for filter and initializes it with random values.
StatusOr<DeviceMemory> CreateFilterData(const FilterDescriptor& filter,
                                        const RandomGenerator& rand_gen);

// Creates cuDNN convolution descriptor from proto.
ConvolutionDescriptor CreateConvolutionDescriptor(
    proto::ConvolutionDescriptor proto);

// Returns true iff left and right are equal.
bool ConvolutionDescriptorEqual(const ConvolutionDescriptor& left,
                                const ConvolutionDescriptor& right);

// Creates an 4D output tensor desciptor for the given parameters.
StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const proto::TensorFormat& format, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ConvolutionDescriptor& convolution);

// Returns the proto's output tensor if one is present, otherwise forwards to
// the function above.
template <typename T>
StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const T& proto, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ConvolutionDescriptor& convolution) {
  if (proto.has_output()) {
    return CreateTensorDescriptor(proto.output());
  }
  return CreateOutputDescriptor(proto.input().format(), input, filter,
                                convolution);
}

// Returns the number of bytes in device_memory_limit_mb flag that have not yet
// been allocated through AllocateDeviceMemory().
size_t GetAvailableDeviceMemoryBytes();

// Returns the workspace limit in bytes specified by the proto, or otherwise
// the device memory limit specified by the corresponding flag minus the number
// of bytes that have already been allocated.
StatusOr<size_t> GetWorkspaceLimit(const proto::ConvolutionConfig& proto);

// Returns the workspace size in bytes required to perform a convolution with
// the given parameters.
StatusOr<size_t> GetWorkspaceSize(const CudnnHandle& handle,
                                  const TensorDescriptor& input,
                                  const FilterDescriptor& filter,
                                  const ConvolutionDescriptor& convolution,
                                  const TensorDescriptor& output,
                                  const ConvolutionAlgo& algo);

// Returns all algorithms that successfully return a workspace size no larger
// than the workspace_limit.
std::vector<ConvolutionAlgo> GetSupportedConvolutionAlgos(
    const CudnnHandle& handle, const proto::ConvolutionDirection& direction,
    const TensorDescriptor& input, const FilterDescriptor& filter,
    const ConvolutionDescriptor& convolution, const TensorDescriptor& output,
    size_t workspace_limit);

// Returns the fastest algorithm.
StatusOr<ConvolutionAlgo> FindConvolutionAlgo(
    const CudnnHandle& handle, const proto::ConvolutionDirection& direction,
    const TensorDescriptor& input_desc, const DeviceMemory& input_data,
    const FilterDescriptor& filter_desc, const DeviceMemory& filter_data,
    const ConvolutionDescriptor& convolution_desc,
    const TensorDescriptor& output_desc, const DeviceMemory& output_data,
    size_t workspace_limit);

// Performs convolution.
Status RunConvolution(const CudnnHandle& handle, const ConvolutionAlgo& algo,
                      double alpha, double beta,
                      const TensorDescriptor& input_desc,
                      const DeviceMemory& input_data,
                      const FilterDescriptor& filter_desc,
                      const DeviceMemory& filter_data,
                      const ConvolutionDescriptor& convolution_desc,
                      const TensorDescriptor& output_desc,
                      const DeviceMemory& output_data,
                      const DeviceMemory& workspace, size_t workspace_size);

// Same as above, but allocates workspace.
Status RunConvolution(const CudnnHandle& handle, const ConvolutionAlgo& algo,
                      double alpha, double beta,
                      const TensorDescriptor& input_desc,
                      const DeviceMemory& input_data,
                      const FilterDescriptor& filter_desc,
                      const DeviceMemory& filter_data,
                      const ConvolutionDescriptor& convolution_desc,
                      const TensorDescriptor& output_desc,
                      const DeviceMemory& output_data);

struct Convolution {
  TensorDescriptor input_desc;
  FilterDescriptor filter_desc;
  TensorDescriptor output_desc;
  ConvolutionDescriptor conv_desc;
  DeviceMemory input_data;
  DeviceMemory filter_data;
  DeviceMemory output_data;
};

StatusOr<Convolution> CreateConvolution(const proto::ConvolutionConfig& proto,
                                        const RandomGenerator& rand_gen);

}  // namespace nvidia_libs_test

// This operator<< is in the global namespace in order to be found by ADL.
// That's important so it can be called from any namespace, not just the
// namespace it was declared in.
//
// ConvolutionAlgo is an instantiation of the absl::variant template class with
// types from the global namespace (::cudnnConvolutionFwdAlgo_t etc.). The fact
// that the typedef is in the nvidia_libs_test namespace is irrelevant for ADL.
// It's undefined behavior to overload functions in the std namespace, and the
// same should apply for the namespace of the C++17 std::variant implementation
// used here. We are therefore left with implementing this function in the
// global namespace (the namespace of the template arguments).
std::ostream& operator<<(std::ostream& str,
                         const nvidia_libs_test::ConvolutionAlgo& algo);

#endif  // NVIDIA_LIBS_TEST_CUDNN_UTIL_H_

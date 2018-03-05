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

#include "cudnn_util.h"

#include <sstream>

#include "base/commandlineflags.h"
#include "base/googleinit.h"
#include "cuda_util.h"

DEFINE_int32(cuda_device, 0, "The CUDA device id to use");
DEFINE_int32(device_memory_limit_mb, 4096,
             "Maximum device memory to use for workspace after tensors have "
             "been allocated, in megabytes. Negative values specify an offset "
             "from the memory available at startup. Defaults to 4096.");

namespace nvidia_libs_test {
namespace {
size_t device_memory_limit_bytes = 0;
}  // namespace

REGISTER_MODULE_INITIALIZER(cudnn_util, {
  int device_id = FLAGS_cuda_device;
  int device_count = 0;
  CHECK_OK_STATUS(GetStatus(cudaGetDeviceCount(&device_count)));
  CHECK_LT(device_id, device_count) << "Invalid CUDA device";
  CHECK_OK_STATUS(GetStatus(cudaSetDevice(device_id)));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CHECK_OK_STATUS(GetStatus(cudaMemGetInfo(&free_bytes, &total_bytes)));
  auto limit_bytes = static_cast<ptrdiff_t>(FLAGS_device_memory_limit_mb) << 20;
  CHECK_GE(free_bytes, std::abs(limit_bytes))
      << "Available device memory is smaller than specified limit.";
  if (limit_bytes < 0) {
    // Use available device memory less flag value.
    device_memory_limit_bytes = free_bytes + limit_bytes;
  } else {
    // Use flag value.
    device_memory_limit_bytes = limit_bytes;
  }
  cudaDeviceProp device_prop;
  CHECK_OK_STATUS(GetStatus(cudaGetDeviceProperties(&device_prop, device_id)));
  auto get_version_string = [](size_t version) {
    std::ostringstream oss;
    oss << version / 1000;
    version %= 1000;
    oss << "." << version / 100;
    version %= 100;
    oss << "." << version;
    return oss.str();
  };
  LOG(INFO) << "Running cuDNN v" << get_version_string(cudnnGetVersion())
            << " for CUDA " << get_version_string(cudnnGetCudartVersion())
            << " on " << device_prop.name;
});

Status GetStatus(cudnnStatus_t status) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return OkStatus();
  }
  const char* str = cudnnGetErrorString(status);
  return ErrorStatus("cuDNN error '") << str << "'";
}

namespace detail {
void CudnnHandleDeleter::operator()(cudnnHandle_t handle) const {
  CHECK_OK_STATUS(GetStatus(cudnnDestroy(handle)));
}

void TensorDescriptorDeleter::operator()(
    cudnnTensorDescriptor_t descriptor) const {
  CHECK_OK_STATUS(GetStatus(cudnnDestroyTensorDescriptor(descriptor)));
}

void FilterDescriptorDeleter::operator()(
    cudnnFilterDescriptor_t descriptor) const {
  CHECK_OK_STATUS(GetStatus(cudnnDestroyFilterDescriptor(descriptor)));
}

void ConvolutionDescriptorDeleter::operator()(
    cudnnConvolutionDescriptor_t descriptor) const {
  CHECK_OK_STATUS(GetStatus(cudnnDestroyConvolutionDescriptor(descriptor)));
}
}  // namespace detail

CudnnHandle CreateCudnnHandle() {
  cudnnHandle_t result;
  CHECK_OK_STATUS(GetStatus(cudnnCreate(&result)));
  return CudnnHandle{result};
}

namespace {
TensorDescriptor CreateTensorDescriptor() {
  cudnnTensorDescriptor_t result;
  CHECK_OK_STATUS(GetStatus(cudnnCreateTensorDescriptor(&result)));
  return TensorDescriptor{result};
}
}  // namespace

TensorDescriptor CreateTensorDescriptor(proto::TensorDescriptor proto) {
  CHECK_EQ(proto.data_type_oneof_case(), proto::TensorDescriptor::kDataType);
  CHECK_EQ(!proto.stride_size(),
           proto.format_oneof_case() == proto::TensorDescriptor::kFormat);
  int rank = proto.dimension_size();
  auto result = CreateTensorDescriptor();
  if (proto.stride_size()) {
    CHECK_EQ(rank, proto.stride_size());
    CHECK_OK_STATUS(GetStatus(cudnnSetTensorNdDescriptor(
        result.get(), static_cast<cudnnDataType_t>(proto.data_type()), rank,
        proto.dimension().data(), proto.stride().data())));
  } else {
    CHECK_EQ(rank, 4);
    CHECK_OK_STATUS(GetStatus(cudnnSetTensor4dDescriptor(
        result.get(), static_cast<cudnnTensorFormat_t>(proto.format()),
        static_cast<cudnnDataType_t>(proto.data_type()), proto.dimension(0),
        proto.dimension(1), proto.dimension(2), proto.dimension(3))));
  }
  return result;
}

struct TensorDescriptorData {
  cudnnDataType_t data_type;
  int rank;
  int dimensions[CUDNN_DIM_MAX];
  int strides[CUDNN_DIM_MAX];
};

bool operator==(const TensorDescriptorData& left,
                const TensorDescriptorData& right) {
  return left.data_type == right.data_type && left.rank == right.rank &&
         std::equal(left.dimensions, left.dimensions + left.rank,
                    right.dimensions) &&
         std::equal(left.strides, left.strides + left.rank, right.strides);
}

TensorDescriptorData GetTensorDescriptorData(
    const cudnnTensorDescriptor_t& tensor) {
  TensorDescriptorData data;
  CHECK_OK_STATUS(GetStatus(
      cudnnGetTensorNdDescriptor(tensor, CUDNN_DIM_MAX, &data.data_type,
                                 &data.rank, data.dimensions, data.strides)));
  return data;
}

bool TensorDescriptorEqual(const TensorDescriptor& left,
                           const TensorDescriptor& right) {
  return GetTensorDescriptorData(left.get()) ==
         GetTensorDescriptorData(right.get());
}

size_t GetTensorNumElements(const TensorDescriptor& tensor) {
  auto data = GetTensorDescriptorData(tensor.get());
  size_t result = 1;
  for (int i = 0; i < data.rank; ++i) {
    result += static_cast<size_t>(data.dimensions[i] - 1) * data.strides[i];
  }
  return result;
}

size_t GetTensorSizeInBytes(const TensorDescriptor& tensor) {
  size_t result = 0;
  CHECK_OK_STATUS(GetStatus(cudnnGetTensorSizeInBytes(tensor.get(), &result)));
  return result;
}

cudnnDataType_t GetTensorDataType(const TensorDescriptor& tensor) {
  return GetTensorDescriptorData(tensor.get()).data_type;
}

namespace {
StatusOr<DeviceMemory> CreateDeviceDataHelper(cudnnDataType_t data_type,
                                              size_t num_elements,
                                              const RandomGenerator& rand_gen) {
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
      return CreateDeviceData<float>(num_elements, rand_gen);
    case CUDNN_DATA_DOUBLE:
      return CreateDeviceData<double>(num_elements, rand_gen);
    case CUDNN_DATA_HALF:
      return CreateDeviceData<__half>(num_elements, rand_gen);
    default:
      LOG(FATAL) << "Not yet supported";
  }
}
}  // namespace

StatusOr<DeviceMemory> CreateTensorData(const TensorDescriptor& tensor,
                                        const RandomGenerator& rand_gen) {
  return CreateDeviceDataHelper(GetTensorDataType(tensor),
                                GetTensorNumElements(tensor), rand_gen);
}

FilterDescriptor CreateFilterDescriptor(const proto::FilterDescriptor& proto) {
  CHECK_EQ(proto.data_type_oneof_case(), proto::FilterDescriptor::kDataType);
  CHECK_EQ(proto.format_oneof_case(), proto::FilterDescriptor::kFormat);
  int rank = proto.dimension_size();
  cudnnFilterDescriptor_t result;
  cudnnCreateFilterDescriptor(&result);
  CHECK_OK_STATUS(GetStatus(cudnnSetFilterNdDescriptor(
      result, static_cast<cudnnDataType_t>(proto.data_type()),
      static_cast<cudnnTensorFormat_t>(proto.format()), rank,
      proto.dimension().data())));
  return FilterDescriptor{result};
}

struct FilterDescriptorData {
  cudnnDataType_t data_type;
  cudnnTensorFormat_t format;
  int rank;
  int dimensions[CUDNN_DIM_MAX];
};

bool operator==(const FilterDescriptorData& left,
                const FilterDescriptorData& right) {
  return left.data_type == right.data_type && left.format == right.format &&
         left.rank == right.rank &&
         std::equal(left.dimensions, left.dimensions + left.rank,
                    right.dimensions);
}

FilterDescriptorData GetFilterDescriptorData(cudnnFilterDescriptor_t filter) {
  FilterDescriptorData data{};
  CHECK_OK_STATUS(GetStatus(
      cudnnGetFilterNdDescriptor(filter, CUDNN_DIM_MAX, &data.data_type,
                                 &data.format, &data.rank, data.dimensions)));
  return data;
}

bool FilterDescriptorEqual(const FilterDescriptor& left,
                           const FilterDescriptor& right) {
  return GetFilterDescriptorData(left.get()) ==
         GetFilterDescriptorData(right.get());
}

size_t GetFilterNumElements(const FilterDescriptor& filter) {
  auto data = GetFilterDescriptorData(filter.get());
  size_t result = 1;
  for (int i = 0; i < data.rank; ++i) {
    result *= data.dimensions[i];
  }
  return result;
}

cudnnDataType_t GetFilterDataType(const FilterDescriptor& filter) {
  return GetFilterDescriptorData(filter.get()).data_type;
}

StatusOr<DeviceMemory> CreateFilterData(const FilterDescriptor& filter,
                                        const RandomGenerator& rand_gen) {
  return CreateDeviceDataHelper(GetFilterDataType(filter),
                                GetFilterNumElements(filter), rand_gen);
}

ConvolutionDescriptor CreateConvolutionDescriptor(
    proto::ConvolutionDescriptor proto) {
  CHECK_EQ(proto.compute_mode_oneof_case(),
           proto::ConvolutionDescriptor::kComputeMode);
  int rank = std::max(
      {proto.pad_size(), proto.filter_stride_size(), proto.dilation_size()});
  while (proto.pad_size() < rank) {
    proto.add_pad(0);
  }
  while (proto.filter_stride_size() < rank) {
    proto.add_filter_stride(1);
  }
  while (proto.dilation_size() < rank) {
    proto.add_dilation(1);
  }
  cudnnConvolutionDescriptor_t result;
  cudnnCreateConvolutionDescriptor(&result);
  CHECK_OK_STATUS(GetStatus(cudnnSetConvolutionNdDescriptor(
      result, rank, proto.pad().data(), proto.filter_stride().data(),
      proto.dilation().data(),
      static_cast<cudnnConvolutionMode_t>(proto.mode()),
      static_cast<cudnnDataType_t>(proto.compute_mode()))));
#if CUDNN_MAJOR >= 7
  if (proto.group_count() > 0) {
    CHECK_OK_STATUS(
        GetStatus(cudnnSetConvolutionGroupCount(result, proto.group_count())));
  }
  CHECK_OK_STATUS(GetStatus(cudnnSetConvolutionMathType(
      result, static_cast<cudnnMathType_t>(proto.math_type()))));
#else
  CHECK_LE(proto.group_count(), 1) << "Grouped convolution requires cuDNN 7";
  LOG_IF(WARNING, proto.math_type() != proto::DEFAULT_MATH)
      << proto::MathType_Name(proto.math_type()) << " requires cuDNN 7";
#endif
  return ConvolutionDescriptor{result};
}

struct ConvolutionDescriptorData {
  int rank;
  int pad[CUDNN_DIM_MAX];
  int stride[CUDNN_DIM_MAX];
  int dilation[CUDNN_DIM_MAX];
  cudnnConvolutionMode_t convolution_mode;
  cudnnDataType_t compute_type;
};

bool operator==(const ConvolutionDescriptorData& left,
                const ConvolutionDescriptorData& right) {
  return left.convolution_mode == right.convolution_mode &&
         left.compute_type == right.compute_type && left.rank == right.rank &&
         std::equal(left.pad, left.pad + left.rank, right.pad) &&
         std::equal(left.stride, left.stride + left.rank, right.stride) &&
         std::equal(left.dilation, left.dilation + left.rank, right.dilation);
}

ConvolutionDescriptorData GetConvolutionDescriptorData(
    const ConvolutionDescriptor& convolution) {
  ConvolutionDescriptorData data{};
  // array_length should be no larger than CUDNN_DIM_MAX according to the
  // documentation, but at least cuDNN 7 reports CUDNN_STATUS_NOT_SUPPORTED
  // for anything larger than 6.
  int array_length = 6;
  CHECK_OK_STATUS(GetStatus(cudnnGetConvolutionNdDescriptor(
      convolution.get(), array_length, &data.rank, data.pad, data.stride,
      data.dilation, &data.convolution_mode, &data.compute_type)));
  return data;
}

bool ConvolutionDescriptorEqual(const ConvolutionDescriptor& left,
                                const ConvolutionDescriptor& right) {
  return GetConvolutionDescriptorData(left) ==
         GetConvolutionDescriptorData(right);
}

StatusOr<TensorDescriptor> CreateOutputDescriptor(
    const proto::TensorFormat& format, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ConvolutionDescriptor& convolution) {
  int n, c, h, w;
  RETURN_IF_ERROR_STATUS(GetStatus(cudnnGetConvolution2dForwardOutputDim(
      convolution.get(), input.get(), filter.get(), &n, &c, &h, &w)));
  auto output = CreateTensorDescriptor();
  RETURN_IF_ERROR_STATUS(GetStatus(cudnnSetTensor4dDescriptor(
      output.get(), static_cast<cudnnTensorFormat_t>(format),
      GetTensorDataType(input), n, c, h, w)));
  return {std::move(output)};
}

size_t GetAvailableDeviceMemoryBytes() {
  size_t allocated = GetAllocatedDeviceMemoryBytes();
  return std::max(device_memory_limit_bytes, allocated) - allocated;
}

StatusOr<size_t> GetWorkspaceSize(const CudnnHandle& handle,
                                  const TensorDescriptor& input,
                                  const FilterDescriptor& filter,
                                  const ConvolutionDescriptor& convolution,
                                  const TensorDescriptor& output,
                                  const ConvolutionAlgo& algo) {
  struct Visitor {
    cudnnStatus_t operator()(cudnnConvolutionFwdAlgo_t algo) {
      return cudnnGetConvolutionForwardWorkspaceSize(
          handle, input, filter, convolution, output, algo, &workspace_size);
    }
    cudnnStatus_t operator()(cudnnConvolutionBwdDataAlgo_t algo) {
      return cudnnGetConvolutionBackwardDataWorkspaceSize(
          handle, filter, output, convolution, input, algo, &workspace_size);
    }
    cudnnStatus_t operator()(cudnnConvolutionBwdFilterAlgo_t algo) {
      return cudnnGetConvolutionBackwardFilterWorkspaceSize(
          handle, input, output, convolution, filter, algo, &workspace_size);
    }
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input;
    cudnnFilterDescriptor_t filter;
    cudnnConvolutionDescriptor_t convolution;
    cudnnTensorDescriptor_t output;
    size_t workspace_size;
  };
  Visitor visitor{handle.get(), input.get(), filter.get(), convolution.get(),
                  output.get()};
  RETURN_IF_ERROR_STATUS(GetStatus(visit(visitor, algo)));
  return visitor.workspace_size;
}

StatusOr<size_t> GetWorkspaceLimit(const proto::ConvolutionConfig& proto) {
  size_t available = GetAvailableDeviceMemoryBytes();
  if (proto.workspace_oneof_case() !=
      proto::ConvolutionConfig::kWorkspaceLimit) {
    return available;
  }
  size_t limit = proto.workspace_limit();
  if (limit > available) {
    return ErrorStatus("Workspace limit (")
           << limit << " bytes) is larger than available memory (" << available
           << " bytes)";
  }
  return limit;
}

namespace {
template <typename T>
std::vector<ConvolutionAlgo> GetSupportedConvolutionAlgosImpl(
    const CudnnHandle& handle, const TensorDescriptor& input,
    const FilterDescriptor& filter, const ConvolutionDescriptor& convolution,
    const TensorDescriptor& output, size_t workspace_limit, int num_elements) {
  // Use backwards-compatible way to get the list of algorithms which
  // cudnnGetConvolution*Algorithm_v7 would return.
  std::vector<ConvolutionAlgo> result;
  for (int i = 0; i < num_elements; ++i) {
    auto algo = static_cast<T>(i);
    auto size_or =
        GetWorkspaceSize(handle, input, filter, convolution, output, algo);
    if (size_or.status().ok() && size_or.ValueOrDie() <= workspace_limit) {
      result.push_back(algo);
    }
  }
  return result;
}
}  // namespace

std::vector<ConvolutionAlgo> GetSupportedConvolutionAlgos(
    const CudnnHandle& handle, const proto::ConvolutionDirection& direction,
    const TensorDescriptor& input, const FilterDescriptor& filter,
    const ConvolutionDescriptor& convolution, const TensorDescriptor& output,
    size_t workspace_limit) {
  switch (direction) {
    case proto::CONVOLUTION_FWD:
      return GetSupportedConvolutionAlgosImpl<cudnnConvolutionFwdAlgo_t>(
          handle, input, filter, convolution, output, workspace_limit,
          CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
    case proto::CONVOLUTION_BWD_DATA:
      return GetSupportedConvolutionAlgosImpl<cudnnConvolutionBwdDataAlgo_t>(
          handle, input, filter, convolution, output, workspace_limit,
          CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);
    case proto::CONVOLUTION_BWD_FILTER:
      return GetSupportedConvolutionAlgosImpl<cudnnConvolutionBwdFilterAlgo_t>(
          handle, input, filter, convolution, output, workspace_limit,
          CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
    default:
      LOG(FATAL) << "Unsupported: " << direction;
  }
}

namespace {
template <typename T>
StatusOr<ConvolutionAlgo> ToConvolutionAlgo(const T& algo_perf,
                                            int num_algorithms) {
  if (!num_algorithms || algo_perf.status != CUDNN_STATUS_SUCCESS) {
    return ErrorStatus("No supported algorithm");
  }
  return ConvolutionAlgo{algo_perf.algo};
}
}  // namespace

StatusOr<ConvolutionAlgo> FindConvolutionAlgo(
    const CudnnHandle& handle, const proto::ConvolutionDirection& direction,
    const TensorDescriptor& input_desc, const DeviceMemory& input_data,
    const FilterDescriptor& filter_desc, const DeviceMemory& filter_data,
    const ConvolutionDescriptor& convolution_desc,
    const TensorDescriptor& output_desc, const DeviceMemory& output_data,
    size_t workspace_limit) {
  ASSIGN_OR_RETURN_STATUS(auto workspace,
                          AllocateDeviceMemory(workspace_limit));
  int num_algorithms = 0;
  switch (direction) {
    case proto::CONVOLUTION_FWD: {
      cudnnConvolutionFwdAlgoPerf_t algo_perf;
      RETURN_IF_ERROR_STATUS(GetStatus(cudnnFindConvolutionForwardAlgorithmEx(
          handle.get(), input_desc.get(), input_data.get(), filter_desc.get(),
          filter_data.get(), convolution_desc.get(), output_desc.get(),
          output_data.get(), 1, &num_algorithms, &algo_perf, workspace.get(),
          workspace_limit)));
      return ToConvolutionAlgo(algo_perf, num_algorithms);
    }
    case proto::CONVOLUTION_BWD_DATA: {
      cudnnConvolutionBwdDataAlgoPerf_t algo_perf;
      RETURN_IF_ERROR_STATUS(
          GetStatus(cudnnFindConvolutionBackwardDataAlgorithmEx(
              handle.get(), filter_desc.get(), filter_data.get(),
              output_desc.get(), output_data.get(), convolution_desc.get(),
              input_desc.get(), input_data.get(), 1, &num_algorithms,
              &algo_perf, workspace.get(), workspace_limit)));
      return ToConvolutionAlgo(algo_perf, num_algorithms);
    }
    case proto::CONVOLUTION_BWD_FILTER: {
      cudnnConvolutionBwdFilterAlgoPerf_t algo_perf;
      RETURN_IF_ERROR_STATUS(
          GetStatus(cudnnFindConvolutionBackwardFilterAlgorithmEx(
              handle.get(), input_desc.get(), input_data.get(),
              output_desc.get(), output_data.get(), convolution_desc.get(),
              filter_desc.get(), filter_data.get(), 1, &num_algorithms,
              &algo_perf, workspace.get(), workspace_limit)));
      return ToConvolutionAlgo(algo_perf, num_algorithms);
    }
    default:
      return ErrorStatus("Unsupported: ") << direction;
  }
}

namespace {
// The scaling factor parameters 'alpha' and 'beta' of the cudnnTransform* and
// cudnnConvolution* functions are type punned pointers. The storage type is
// double for double output tensors, and float otherwise.
union ScalingFactor {
  ScalingFactor(double value, cudnnTensorDescriptor_t descriptor)
      : ScalingFactor(value, GetTensorDescriptorData(descriptor).data_type) {}
  ScalingFactor(double value, cudnnFilterDescriptor_t descriptor)
      : ScalingFactor(value, GetFilterDescriptorData(descriptor).data_type) {}

 private:
  ScalingFactor(double value, cudnnDataType_t data_type) {
    if (data_type == CUDNN_DATA_DOUBLE) {
      double_value = value;
    } else {
      float_value = static_cast<float>(value);
    }
  }

  float float_value;
  double double_value;
};
}  // namespace

Status TransformTensor(const CudnnHandle& handle, double alpha, double beta,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data) {
  ScalingFactor alpha_scale(alpha, dst_desc.get());
  ScalingFactor beta_scale(beta, dst_desc.get());
  RETURN_IF_ERROR_STATUS(GetStatus(cudnnTransformTensor(
      handle.get(), &alpha_scale, src_desc.get(), src_data.get(), &beta_scale,
      dst_desc.get(), dst_data.get())));
  return OkStatus();
}

Status TransformTensor(const CudnnHandle& handle,
                       const TensorDescriptor& src_desc,
                       const DeviceMemory& src_data,
                       const TensorDescriptor& dst_desc,
                       const DeviceMemory& dst_data) {
  return TransformTensor(handle, 1.0, 0.0, src_desc, src_data, dst_desc,
                         dst_data);
}

namespace {
struct ConvertDeviceDataVisitor {
  template <typename DstT, typename SrcT>
  void operator()(DstT* dst, const SrcT* src) {
    ConvertDeviceData(scale, dst, src, dst_size_in_bytes / sizeof(DstT));
  }
  template <typename T>
  void operator()(T*, const T* src) {
    LOG(FATAL) << "No conversion needed";
  }
  size_t dst_size_in_bytes;
  double scale;
};

mpark::variant<float*, double*, __half*> GetPointerVariant(
    const DeviceMemory& data, cudnnDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
      return static_cast<float*>(data.get());
    case CUDNN_DATA_DOUBLE:
      return static_cast<double*>(data.get());
    case CUDNN_DATA_HALF:
      return static_cast<__half*>(data.get());
    default:
      LOG(FATAL) << "Not yet implemented";
  }
}
}  // namespace

Status ConvertAndTransformTensor(const CudnnHandle& handle, double alpha,
                                 double beta, const TensorDescriptor& src_desc,
                                 const DeviceMemory& src_data,
                                 const TensorDescriptor& dst_desc,
                                 const DeviceMemory& dst_data) {
  auto src_desc_data = GetTensorDescriptorData(src_desc.get());
  auto dst_desc_data = GetTensorDescriptorData(dst_desc.get());
  if (src_desc_data.data_type == dst_desc_data.data_type) {
    return TransformTensor(handle, alpha, beta, src_desc, src_data, dst_desc,
                           dst_data);
  }
  CHECK_EQ(src_desc_data.rank, dst_desc_data.rank);
  auto temp_desc = CreateTensorDescriptor();
  RETURN_IF_ERROR_STATUS(GetStatus(cudnnSetTensorNdDescriptor(
      temp_desc.get(), dst_desc_data.data_type, src_desc_data.rank,
      src_desc_data.dimensions, src_desc_data.strides)));

  size_t temp_size = GetTensorSizeInBytes(temp_desc);
  ASSIGN_OR_RETURN_STATUS(auto temp_data, AllocateDeviceMemory(temp_size));

  visit(ConvertDeviceDataVisitor{temp_size, alpha},
        GetPointerVariant(temp_data, dst_desc_data.data_type),
        GetPointerVariant(src_data, src_desc_data.data_type));

  RETURN_IF_ERROR_STATUS(GetStatus(cudaDeviceSynchronize()));
  return TransformTensor(handle, 1.0, beta, temp_desc, temp_data, dst_desc,
                         dst_data);
}

namespace {
struct PrintVisitor {
  template <typename T>
  void operator()(T* device_ptr) {
    size_t num_elements = size_in_bytes / sizeof(T);
    std::unique_ptr<T[]> host_ptr{new T[num_elements]};
    CHECK_OK_STATUS(GetStatus(cudaMemcpy(
        host_ptr.get(), device_ptr, size_in_bytes, cudaMemcpyDeviceToHost)));
  }

  void operator()(__half* device_ptr) {
    size_t num_elements = size_in_bytes / sizeof(__half);
    std::unique_ptr<float[]> host_ptr{new float[num_elements]};
    ConvertDeviceData(1.0, host_ptr.get(), device_ptr, num_elements);
    Print(host_ptr.get(), num_elements);
  }

  template <typename T>
  void Print(const T* ptr, size_t num_elements) {
    CHECK_OK_STATUS(GetStatus(cudaDeviceSynchronize()));
    for (size_t i = 0; i < num_elements; ++i) {
      *oss << " " << ptr[i];
    }
  }

  std::ostringstream* oss;
  size_t size_in_bytes;
};
}  // namespace

string GetTensorDebugString(const TensorDescriptor& desc,
                            const DeviceMemory& data, bool print_values) {
  std::ostringstream oss;
  auto desc_data = GetTensorDescriptorData(desc.get());
  oss << "data_type: "
      << proto::DataType_Name(
             static_cast<proto::DataType>(desc_data.data_type));
  oss << "\ndimensions:";
  for (int i = 0; i < desc_data.rank; ++i) {
    oss << " " << desc_data.dimensions[i];
  }
  oss << "\nstrides:";
  for (int i = 0; i < desc_data.rank; ++i) {
    oss << " " << desc_data.strides[i];
  }
  if (print_values) {
    oss << "\nvalues:";
    visit(PrintVisitor{&oss, GetTensorSizeInBytes(desc)},
          GetPointerVariant(data, desc_data.data_type));
  }
  return oss.str();
}

Status RunConvolution(const CudnnHandle& handle, const ConvolutionAlgo& algo,
                      double alpha, double beta,
                      const TensorDescriptor& input_desc,
                      const DeviceMemory& input_data,
                      const FilterDescriptor& filter_desc,
                      const DeviceMemory& filter_data,
                      const ConvolutionDescriptor& convolution_desc,
                      const TensorDescriptor& output_desc,
                      const DeviceMemory& output_data,
                      const DeviceMemory& workspace, size_t workspace_size) {
  struct Visitor {
    cudnnStatus_t operator()(cudnnConvolutionFwdAlgo_t algo) {
      ScalingFactor alpha_scale(alpha, output_desc);
      ScalingFactor beta_scale(beta, output_desc);
      return cudnnConvolutionForward(
          handle, &alpha_scale, input_desc, input_data, filter_desc,
          filter_data, convolution_desc, algo, workspace, workspace_size,
          &beta_scale, output_desc, output_data);
    }
    cudnnStatus_t operator()(cudnnConvolutionBwdDataAlgo_t algo) {
      ScalingFactor alpha_scale(alpha, input_desc);
      ScalingFactor beta_scale(beta, input_desc);
      return cudnnConvolutionBackwardData(
          handle, &alpha_scale, filter_desc, filter_data, output_desc,
          output_data, convolution_desc, algo, workspace, workspace_size,
          &beta_scale, input_desc, input_data);
    }
    cudnnStatus_t operator()(cudnnConvolutionBwdFilterAlgo_t algo) {
      ScalingFactor alpha_scale(alpha, filter_desc);
      ScalingFactor beta_scale(beta, filter_desc);
      return cudnnConvolutionBackwardFilter(
          handle, &alpha_scale, input_desc, input_data, output_desc,
          output_data, convolution_desc, algo, workspace, workspace_size,
          &beta_scale, filter_desc, filter_data);
    }
    cudnnHandle_t handle;
    double alpha, beta;
    cudnnTensorDescriptor_t input_desc;
    void* input_data;
    cudnnFilterDescriptor_t filter_desc;
    void* filter_data;
    cudnnConvolutionDescriptor_t convolution_desc;
    cudnnTensorDescriptor_t output_desc;
    void* output_data;
    void* workspace;
    size_t workspace_size;
  };
  Visitor visitor{handle.get(),
                  alpha,
                  beta,
                  input_desc.get(),
                  input_data.get(),
                  filter_desc.get(),
                  filter_data.get(),
                  convolution_desc.get(),
                  output_desc.get(),
                  output_data.get(),
                  workspace.get(),
                  workspace_size};
  return GetStatus(visit(visitor, algo));
}

Status RunConvolution(const CudnnHandle& handle, const ConvolutionAlgo& algo,
                      double alpha, double beta,
                      const TensorDescriptor& input_desc,
                      const DeviceMemory& input_data,
                      const FilterDescriptor& filter_desc,
                      const DeviceMemory& filter_data,
                      const ConvolutionDescriptor& convolution_desc,
                      const TensorDescriptor& output_desc,
                      const DeviceMemory& output_data) {
  ASSIGN_OR_RETURN_STATUS(
      auto workspace_size,
      GetWorkspaceSize(handle, input_desc, filter_desc, convolution_desc,
                       output_desc, algo));

  ASSIGN_OR_RETURN_STATUS(auto workspace, AllocateDeviceMemory(workspace_size));

  return RunConvolution(handle, algo, alpha, beta, input_desc, input_data,
                        filter_desc, filter_data, convolution_desc, output_desc,
                        output_data, workspace, workspace_size);
}

StatusOr<Convolution> CreateConvolution(const proto::ConvolutionConfig& proto,
                                        const RandomGenerator& rand_gen) {
  if (CUDNN_MAJOR < 7 && proto.convolution().group_count() > 1) {
    return ErrorStatus("Grouped convolution requires cuDNN 7");
  }

  auto input_desc = CreateTensorDescriptor(proto.input());
  auto filter_desc = CreateFilterDescriptor(proto.filter());
  auto conv_desc = CreateConvolutionDescriptor(proto.convolution());

  ASSIGN_OR_RETURN_STATUS(
      auto output_desc,
      CreateOutputDescriptor(proto, input_desc, filter_desc, conv_desc));

  ASSIGN_OR_RETURN_STATUS(auto input_data,
                          CreateTensorData(input_desc, rand_gen));

  ASSIGN_OR_RETURN_STATUS(auto filter_data,
                          CreateFilterData(filter_desc, rand_gen));

  ASSIGN_OR_RETURN_STATUS(auto output_data,
                          CreateTensorData(output_desc, rand_gen));

  return Convolution{std::move(input_desc),  std::move(filter_desc),
                     std::move(output_desc), std::move(conv_desc),
                     std::move(input_data),  std::move(filter_data),
                     std::move(output_data)};
}

namespace {
string GetAlgoName(const ConvolutionAlgo& algo) {
  struct Visitor {
    string operator()(cudnnConvolutionFwdAlgo_t algo) const {
      return proto::ConvolutionFwdAlgo_Name(
          static_cast<proto::ConvolutionFwdAlgo>(algo));
    }
    string operator()(cudnnConvolutionBwdDataAlgo_t algo) const {
      return proto::ConvolutionBwdDataAlgo_Name(
          static_cast<proto::ConvolutionBwdDataAlgo>(algo));
    }
    string operator()(cudnnConvolutionBwdFilterAlgo_t algo) const {
      return proto::ConvolutionBwdFilterAlgo_Name(
          static_cast<proto::ConvolutionBwdFilterAlgo>(algo));
    }
  };
  return visit(Visitor(), algo);
}

#define CHECK_ENUMERATOR(enumerator)                      \
  static_assert(static_cast<int>(proto::enumerator) ==    \
                    static_cast<int>(CUDNN_##enumerator), \
                "enum values don't match")

#define CHECK_ENUM_SIZE(proto_enum, cudnn_enum)                   \
  static_assert(proto::proto_enum##_ARRAYSIZE ==                  \
                    static_cast<int>(CUDNN_##cudnn_enum##_COUNT), \
                "size does not match")

CHECK_ENUMERATOR(CONVOLUTION);
CHECK_ENUMERATOR(CROSS_CORRELATION);
CHECK_ENUMERATOR(DATA_FLOAT);
CHECK_ENUMERATOR(DATA_DOUBLE);
CHECK_ENUMERATOR(DATA_HALF);
CHECK_ENUMERATOR(DATA_INT8);
CHECK_ENUMERATOR(DATA_INT32);
CHECK_ENUMERATOR(DATA_INT8x4);
CHECK_ENUMERATOR(TENSOR_NCHW);
CHECK_ENUMERATOR(TENSOR_NHWC);
CHECK_ENUMERATOR(TENSOR_NCHW_VECT_C);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_GEMM);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_DIRECT);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_FFT);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_FFT_TILING);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_WINOGRAD);
CHECK_ENUMERATOR(CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
CHECK_ENUM_SIZE(ConvolutionFwdAlgo, CONVOLUTION_FWD_ALGO);
CHECK_ENUMERATOR(CONVOLUTION_BWD_DATA_ALGO_0);
CHECK_ENUMERATOR(CONVOLUTION_BWD_DATA_ALGO_1);
CHECK_ENUMERATOR(CONVOLUTION_BWD_DATA_ALGO_FFT);
CHECK_ENUMERATOR(CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);
CHECK_ENUMERATOR(CONVOLUTION_BWD_DATA_ALGO_WINOGRAD);
CHECK_ENUMERATOR(CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
CHECK_ENUM_SIZE(ConvolutionBwdDataAlgo, CONVOLUTION_BWD_DATA_ALGO);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_0);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_1);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_FFT);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_3);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
CHECK_ENUMERATOR(CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING);
CHECK_ENUM_SIZE(ConvolutionBwdFilterAlgo, CONVOLUTION_BWD_FILTER_ALGO);

#if CUDNN_MAJOR >= 7
CHECK_ENUMERATOR(DEFAULT_MATH);
CHECK_ENUMERATOR(TENSOR_OP_MATH);
#endif
}  // namespace
}  // namespace nvidia_libs_test

std::ostream& operator<<(std::ostream& str,
                         const nvidia_libs_test::ConvolutionAlgo& algo) {
  return str << nvidia_libs_test::GetAlgoName(algo);
}

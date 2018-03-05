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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "google/protobuf/repeated_field.h"
#include "gtest/gtest.h"
#include "absl/types/optional.h"
#include "cudnn/include/cudnn.h"
#include "all_pairs.h"
#include "cuda_util.h"
#include "cudnn.pb.h"
#include "cudnn_test.h"
#include "cudnn_util.h"
#include "test_util.h"

// Tests that compare the outputs of all supported convolution algorithms for
// specific configuration of input, filter, etc. The configurations are loaded
// from cudnn_tests.textproto and generated using the all-pair testing approach.
//
// For cuDNN 7 and later, also tests that the cudnnGetConvolution*Algorithm_v7
// returns the same set of algorithms as GetSupportedConvolutionAlgos.

namespace nvidia_libs_test {
namespace {
class ConvolutionTest
    : public ::testing::TestWithParam<proto::ConvolutionTest> {
  void TearDown() override {
    if (HasFatalFailure()) {
      ResetDevice();
    }
  }
};

// Returns expected accuracy of the given algorithm. The tolerance is a mix
// between absolute and relative error, see test_util.h.
double GetTolerance(const ConvolutionAlgo& algo) {
  struct Visitor {
    double operator()(cudnnConvolutionFwdAlgo_t algo) {
      switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
          return 1e-4;
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
          return 1e-4;
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
          return 1e-4;
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
          return 0.0;  // Not implemented.
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
          return 1e-5;
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
          return 1e-5;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
          return 1e-5;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
          return 1e-2;
        default:
          LOG(FATAL) << "Unknown algorithm";
      }
    }
    double operator()(cudnnConvolutionBwdDataAlgo_t algo) {
      switch (algo) {
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
          return 1e-5;
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
          return 1e-4;
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
          return 1e-5;
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
          return 1e-4;
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
          return 1e-5;
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
          return 1e-2;
        default:
          LOG(FATAL) << "Unknown algorithm";
      }
    }
    double operator()(cudnnConvolutionBwdFilterAlgo_t algo) {
      switch (algo) {
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
          return 1e-5;
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
          return 1e-4;
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
          return 1e-5;
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
          return 1e-4;
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
          return 0.0;  // Not implemented.
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
          return 1e-2;
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
          return 1e-5;
        default:
          LOG(FATAL) << "Unknown algorithm";
      }
    }
  };
  return visit(Visitor(), algo);
}

// Returns expected accuracy of the convolution based on the data types
// configuration specified in the proto, relative to FLOAT_CONFIG (see cuDNN
// documentation).
double GetToleranceScale(const proto::ConvolutionConfig& proto) {
  switch (proto.filter().data_type()) {
    case proto::DATA_FLOAT:
      return 1.0;
    case proto::DATA_DOUBLE:
      // cuDNN uses different kernels for architectures before Maxwell that have
      // much lower double accuracy.
      return DeviceHasAtLeastComputeCapability(5, 0) ? 1e-8 : 1e-2;
    case proto::DATA_HALF:
      if (proto.convolution().math_type() == proto::TENSOR_OP_MATH) {
        return 1e+4;  // Using tensor ops.
      }
      if (proto.convolution().compute_mode() == proto::DATA_FLOAT) {
        return 1e+5;  // Pseudo half config.
      }
      return 1e+5;  // True half config.
    default:
      LOG(FATAL) << "Not yet supported";
  }
}

// Returns a tensor descriptor with the same layout as the given filter
// descriptor.
proto::TensorDescriptor GetFilterTensorDescriptor(
    const proto::FilterDescriptor& filter) {
  proto::TensorDescriptor result;
  *result.mutable_dimension() = filter.dimension();
  result.set_data_type(filter.data_type());
  if (result.dimension_size() == 4) {
    result.set_format(filter.format());
    return result;
  }
  // If filter rank is not 4, it must be fully packed.
  CHECK_EQ(filter.format(), proto::TENSOR_NCHW);
  int rank = result.dimension_size();
  auto* strides = result.mutable_stride();
  strides->Resize(rank, 0);
  for (int i = rank - 1, stride = 1; i >= 0; --i) {
    strides->Set(i, stride);
    stride *= filter.dimension(i);
  }
  return result;
}

// Returns the algorithms specified in the proto.
std::vector<ConvolutionAlgo> GetAlgorithms(
    const proto::ConvolutionConfig& proto, const CudnnHandle& handle,
    const Convolution& convolution, size_t workspace_limit) {
  switch (proto.algo_oneof_case()) {
    case proto::ConvolutionConfig::kAllAlgos: {
      return GetSupportedConvolutionAlgos(
          handle, proto.all_algos(), convolution.input_desc,
          convolution.filter_desc, convolution.conv_desc,
          convolution.output_desc, workspace_limit);
    }
    case proto::ConvolutionConfig::kFwdAlgo:
      return {static_cast<cudnnConvolutionFwdAlgo_t>(proto.fwd_algo())};
    case proto::ConvolutionConfig::kBwdDataAlgo:
      return {
          static_cast<cudnnConvolutionBwdDataAlgo_t>(proto.bwd_data_algo())};
    case proto::ConvolutionConfig::kBwdFilterAlgo:
      return {static_cast<cudnnConvolutionBwdFilterAlgo_t>(
          proto.bwd_filter_algo())};
    default:
      LOG(FATAL) << "Invalid algo_oneof case";
  }
}

// Runs the convolution.
Status RunConvolution(double alpha, double beta, const CudnnHandle& handle,
                      const Convolution& convolution,
                      const ConvolutionAlgo& algo) {
  return RunConvolution(handle, algo, alpha, beta, convolution.input_desc,
                        convolution.input_data, convolution.filter_desc,
                        convolution.filter_data, convolution.conv_desc,
                        convolution.output_desc, convolution.output_data);
}

// Returns 'from' merged into 'proto', handling repeated and oneof fields.
proto::ConvolutionConfig GetMergedConvolution(
    proto::ConvolutionConfig proto, const proto::ConvolutionConfig& from) {
  auto process_tensor_desc = [](proto::TensorDescriptor* proto_tensor,
                                const proto::TensorDescriptor& from_tensor) {
    if (from_tensor.dimension_size()) {
      proto_tensor->clear_dimension();
    }
    if (from_tensor.data_type_oneof_case()) {
      proto_tensor->clear_data_type();
    }
    if (from_tensor.format_oneof_case()) {
      proto_tensor->clear_format();
    }
    if (from_tensor.stride_size()) {
      proto_tensor->clear_stride();
    }
  };

  // Clear all repeated and oneof fields in 'proto' that are set in 'from'.
  process_tensor_desc(proto.mutable_input(), from.input());
  if (from.has_output()) {
    process_tensor_desc(proto.mutable_output(), from.output());
  }

  if (from.filter().dimension_size()) {
    proto.mutable_filter()->clear_dimension();
  }
  if (from.filter().data_type_oneof_case()) {
    proto.mutable_filter()->clear_data_type();
  }
  if (from.filter().format_oneof_case()) {
    proto.mutable_filter()->clear_format();
  }

  if (from.convolution().pad_size()) {
    proto.mutable_convolution()->clear_pad();
  }
  if (from.convolution().filter_stride_size()) {
    proto.mutable_convolution()->clear_filter_stride();
  }
  if (from.convolution().dilation_size()) {
    proto.mutable_convolution()->clear_dilation();
  }
  if (from.convolution().compute_mode_oneof_case()) {
    proto.mutable_convolution()->clear_compute_mode();
  }

  if (from.algo_oneof_case()) {
    proto.set_all_algos(proto::CONVOLUTION_DIRECTION_UNSPECIFIED);
  }

  proto.MergeFrom(from);

  // Always set the convolution fields not wrapped in oneofs.
  proto.mutable_convolution()->set_mode(from.convolution().mode());
  proto.mutable_convolution()->set_math_type(from.convolution().math_type());
  proto.mutable_convolution()->set_group_count(
      from.convolution().group_count());

  return proto;
}

TEST_P(ConvolutionTest, CompareResults) {
  VLOG(2) << "Running all-test for\n" << GetParam().DebugString();

  CudnnHandle handle = CreateCudnnHandle();
  RandomGenerator rand_gen(/*seed=*/0);

  // Run reference convolution first, then compare others to the reference
  // result.

  auto ref_proto = GetParam().reference();
  ASSERT_OK_AND_ASSIGN(auto reference, CreateConvolution(ref_proto, rand_gen));

  ASSERT_NE(ref_proto.algo_oneof_case(), proto::ConvolutionConfig::kAllAlgos);
  auto ref_algo = GetAlgorithms(ref_proto, handle, reference, 0).front();

  // Run reference convolution with default scaling factors.
  ASSERT_TRUE(IsOk(RunConvolution(1.0, 0.0, handle, reference, ref_algo)))
      << "algo: " << ref_algo;

  auto ref_filter_tensor_desc =
      CreateTensorDescriptor(GetFilterTensorDescriptor(ref_proto.filter()));

  for (auto proto : GetParam().test()) {
    proto = GetMergedConvolution(ref_proto, proto);
    ASSERT_OK_AND_ASSIGN(Convolution test, CreateConvolution(proto, rand_gen));

    // We now have input, filter, and output buffers for the reference and the
    // test. A convolution has two read-only argument buffers and writes the
    // result to the remaining buffer. The convolution direction determines
    // which buffers are arguments and which buffer is the result.
    //
    // convolution direction:    argument buffers:     result buffer:
    // --------------------------------------------------------------
    // forward                   input, filter         output
    // backward data             filter, output        input
    // backward filter           output, input         filter
    //
    // The reference buffers contain the arguments and ground truth result of
    // the convolution. The test buffers are filled with random values at this
    // point. Tensors aren't necessarily packed and we use the random value of
    // buffer elements not referenced by the tensor to detect out-of-bounds
    // access by cuDNN.

    auto direction = [&] {
      switch (proto.algo_oneof_case()) {
        case proto::ConvolutionConfig::kAllAlgos:
          return proto.all_algos();
        case proto::ConvolutionConfig::kFwdAlgo:
          return proto::CONVOLUTION_FWD;
        case proto::ConvolutionConfig::kBwdDataAlgo:
          return proto::CONVOLUTION_BWD_DATA;
        case proto::ConvolutionConfig::kBwdFilterAlgo:
          return proto::CONVOLUTION_BWD_FILTER;
        default:
          LOG(FATAL) << "Invalid algo_oneof case";
      }
    }();

    auto filter_tensor_desc =
        CreateTensorDescriptor(GetFilterTensorDescriptor(proto.filter()));
    double alpha = 1.0 - proto.one_minus_alpha();
    double beta = proto.beta();

    // Get a pointer to the test result buffer and its tensor descriptor.
    TensorDescriptor* result_desc = nullptr;
    DeviceMemory* result_data = nullptr;
    struct {
      double alpha = 1.0, beta = 0.0;
    } input_scaling, filter_scaling, output_scaling;
    switch (direction) {
      case proto::CONVOLUTION_FWD:
        result_desc = &test.output_desc;
        result_data = &test.output_data;
        output_scaling.alpha = alpha;
        output_scaling.beta = beta;
        break;
      case proto::CONVOLUTION_BWD_DATA:
        result_desc = &test.input_desc;
        result_data = &test.input_data;
        input_scaling.alpha = alpha;
        input_scaling.beta = beta;
        break;
      case proto::CONVOLUTION_BWD_FILTER:
        result_desc = &filter_tensor_desc;
        result_data = &test.filter_data;
        filter_scaling.alpha = alpha;
        filter_scaling.beta = beta;
        break;
      default:
        LOG(FATAL) << "Invalid direction: "
                   << proto::ConvolutionDirection_Name(direction);
    }
    size_t result_size = GetTensorSizeInBytes(*result_desc);

    // Make a copy of the test result buffer so that we can reset it to the same
    // random values before running each algorithm.
    ASSERT_OK_AND_ASSIGN(auto init_data, AllocateDeviceMemory(result_size));
    ASSERT_TRUE(IsOk(CopyDeviceMemory(init_data, *result_data, result_size)));

    // Blend the reference buffers into the corresponding test buffers. This
    // may involve converting the data type (e.g. from double to float) and
    // changing the data layout (e.g. from NCHW to NHWC).
    ASSERT_TRUE(IsOk(ConvertAndTransformTensor(
        handle, input_scaling.alpha, input_scaling.beta, reference.input_desc,
        reference.input_data, test.input_desc, test.input_data)));
    ASSERT_TRUE(IsOk(ConvertAndTransformTensor(
        handle, filter_scaling.alpha, filter_scaling.beta,
        ref_filter_tensor_desc, reference.filter_data, filter_tensor_desc,
        test.filter_data)));
    ASSERT_TRUE(IsOk(ConvertAndTransformTensor(
        handle, output_scaling.alpha, output_scaling.beta,
        reference.output_desc, reference.output_data, test.output_desc,
        test.output_data)));

    // The test result buffer (pointee of result_data) now contains the copy of
    // the reference result buffer. Stash that away in ref_result_data and
    // allocate a new buffer for the test's actual result buffer.
    ASSERT_OK_AND_ASSIGN(auto ref_result_data,
                         AllocateDeviceMemory(result_size));
    std::swap(*result_data, ref_result_data);

    // Get available workspace size (after allocating all device memory).
    ASSERT_OK_AND_ASSIGN(auto workspace_limit, GetWorkspaceLimit(proto));

    // Run all supported algorithms for current test.
    double tolerance_scale = GetToleranceScale(proto);
    for (const auto& algo :
         GetAlgorithms(proto, handle, test, workspace_limit)) {
      // Reset the test result buffer and run the convolution.
      ASSERT_TRUE(IsOk(CopyDeviceMemory(*result_data, init_data, result_size)));
      auto get_message = [&] {
        std::ostringstream oss;
        oss << "format: " << proto::TensorFormat_Name(proto.input().format())
            << "\ndata_type: "
            << proto::DataType_Name(proto.input().data_type())
            << "\ncompute_mode: "
            << proto::DataType_Name(proto.convolution().compute_mode())
            << "\nmath_type: "
            << proto::MathType_Name(proto.convolution().math_type())
            << "\nalgo: " << algo;
        return oss.str();
      };
      ASSERT_TRUE(IsOk(RunConvolution(alpha, beta, handle, test, algo)))
          << get_message();
      // Check that the test result matches the reference result. This also
      // compares buffer elements not referenced by the tensor.
      double tolerance = tolerance_scale * GetTolerance(algo);
      EXPECT_TRUE(IsOk(TensorDataEqual(ref_result_data, *result_data,
                                       *result_desc, tolerance)))
          << get_message();
    }
  }
}

#if CUDNN_MAJOR >= 7
template <typename T>
std::vector<ConvolutionAlgo> ToConvolutionAlgos(std::vector<T> algo_perfs,
                                                int num_algorithms) {
  auto end =
      std::remove_if(algo_perfs.begin(), algo_perfs.begin() + num_algorithms,
                     [](const T& algo_perf) {
                       return algo_perf.status != CUDNN_STATUS_SUCCESS;
                     });
  std::vector<ConvolutionAlgo> result;
  std::transform(algo_perfs.begin(), end, std::back_inserter(result),
                 [](const T& algo_perf) { return algo_perf.algo; });
  return result;
}

TEST_P(ConvolutionTest, GetAlgorithm_v7) {
  CudnnHandle handle = CreateCudnnHandle();

  auto ref_proto = GetParam().reference();

  for (auto proto : GetParam().test()) {
    proto = GetMergedConvolution(ref_proto, proto);
    if (proto.algo_oneof_case() != proto::ConvolutionConfig::kAllAlgos) {
      continue;
    }

    auto direction = proto.all_algos();
    auto input = CreateTensorDescriptor(proto.input());
    auto filter = CreateFilterDescriptor(proto.filter());
    auto convolution = CreateConvolutionDescriptor(proto.convolution());

    ASSERT_OK_AND_ASSIGN(
        auto output, CreateOutputDescriptor(proto, input, filter, convolution));

    std::vector<ConvolutionAlgo> get_v7_algos;
    int num_algorithms = 0;
    switch (direction) {
      case proto::CONVOLUTION_FWD: {
        std::vector<cudnnConvolutionFwdAlgoPerf_t> algo_perfs(
            CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
        ASSERT_TRUE(IsOk(GetStatus(cudnnGetConvolutionForwardAlgorithm_v7(
            handle.get(), input.get(), filter.get(), convolution.get(),
            output.get(), algo_perfs.size(), &num_algorithms,
            algo_perfs.data()))));
        get_v7_algos = ToConvolutionAlgos(algo_perfs, num_algorithms);
      } break;
      case proto::CONVOLUTION_BWD_DATA: {
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> algo_perfs(
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);
        ASSERT_TRUE(IsOk(GetStatus(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle.get(), filter.get(), output.get(), convolution.get(),
            input.get(), algo_perfs.size(), &num_algorithms,
            algo_perfs.data()))));
        get_v7_algos = ToConvolutionAlgos(algo_perfs, num_algorithms);
      } break;
      case proto::CONVOLUTION_BWD_FILTER: {
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> algo_perfs(
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
        ASSERT_TRUE(
            IsOk(GetStatus(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle.get(), input.get(), output.get(), convolution.get(),
                filter.get(), algo_perfs.size(), &num_algorithms,
                algo_perfs.data()))));
        get_v7_algos = ToConvolutionAlgos(algo_perfs, num_algorithms);
      } break;
      default:
        LOG(FATAL) << "Invalid direction: "
                   << proto::ConvolutionDirection_Name(direction);
    }

    auto get_supported_algos = GetSupportedConvolutionAlgos(
        handle, direction, input, filter, convolution, output,
        std::numeric_limits<size_t>::max());

    std::sort(get_supported_algos.begin(), get_supported_algos.end());
    std::sort(get_v7_algos.begin(), get_v7_algos.end());
    EXPECT_EQ(get_supported_algos, get_v7_algos);
  }
}
#endif  // CUDNN_MAJOR >= 7

// Creates a ConvolutionConfig from the given parameters for
// proto.ConvolutionTest.reference.
proto::ConvolutionConfig MakeConvolutionConfig(int batch_count, int in_depth,
                                               int in_height, int in_width,
                                               int out_depth, int filter_height,
                                               int filter_width,
                                               Padding padding) {
  proto::TensorDescriptor input;
  for (int dim : {batch_count, in_depth, in_height, in_width}) {
    input.add_dimension(dim);
  }
  input.set_format(proto::TENSOR_NCHW);
  input.set_data_type(proto::DATA_DOUBLE);

  proto::FilterDescriptor filter;
  for (int dim : {out_depth, in_depth, filter_height, filter_width}) {
    filter.add_dimension(dim);
  }
  filter.set_format(proto::TENSOR_NCHW);
  filter.set_data_type(proto::DATA_DOUBLE);

  proto::ConvolutionDescriptor convolution;
  for (auto filter_dim : {filter_height, filter_width}) {
    // SAME padding is only this simple if stride and dilation are 1.
    // TODO: Add function to compute padding for generic stride/dilation.
    convolution.add_pad(padding == Padding::SAME ? filter_dim / 2 : 0);
  }
  convolution.set_compute_mode(proto::DATA_DOUBLE);

  proto::ConvolutionConfig result;
  *result.mutable_input() = std::move(input);
  *result.mutable_filter() = std::move(filter);
  *result.mutable_convolution() = std::move(convolution);

  return result;
}

struct SizeRange {
  int minimum;
  int maximum;
};

// Returns range ['largest 2^k+1 not greater than size', size].
SizeRange MakePowerOfTwoRange(int size) {
  CHECK_GT(size, 0);
  int power_of_two = 1;
  while (power_of_two < size) {
    power_of_two *= 2;
  }
  return {power_of_two / 2 + 1, size};
}

// Creates a randomized ConvolutionConfig from the given parameter ranges for
// proto.ConvolutionTest.reference. For VALID padding, the ranges of input size
// and filter size are clamped so that the filter is no larger then the inputs.
template <typename RandGen>
proto::ConvolutionConfig MakeConvolutionConfig(
    RandGen&& rand_gen, SizeRange batch_count_range, SizeRange in_depth_range,
    SizeRange in_height_range, SizeRange in_width_range,
    SizeRange out_depth_range, SizeRange filter_height_range,
    SizeRange filter_width_range, Padding padding) {
  if (padding == Padding::VALID) {
    in_height_range.minimum =
        std::max(in_height_range.minimum, filter_height_range.minimum);
    in_width_range.minimum =
        std::max(in_width_range.minimum, filter_width_range.minimum);
  }

  auto randomize = [&](const SizeRange& range) {
    CHECK_LE(range.minimum, range.maximum);
    return std::uniform_int_distribution<int>{range.minimum,
                                              range.maximum}(rand_gen);
  };
  int batch_count = randomize(batch_count_range);
  int in_depth = randomize(in_depth_range);
  int in_height = randomize(in_height_range);
  int in_width = randomize(in_width_range);
  int out_depth = randomize(out_depth_range);

  if (padding == Padding::VALID) {
    filter_height_range.maximum =
        std::min(filter_height_range.maximum, in_height);
    filter_width_range.maximum = std::min(filter_width_range.maximum, in_width);
  }
  int filter_height = randomize(filter_height_range);
  int filter_width = randomize(filter_width_range);

  auto result =
      MakeConvolutionConfig(batch_count, in_depth, in_height, in_width,
                            out_depth, filter_height, filter_width, padding);

  std::ostringstream label;
  label << batch_count << 'x' << in_depth << 'x' << in_height << 'x' << in_width
        << '_' << out_depth << 'x' << in_depth << 'x' << filter_height << 'x'
        << filter_width << '_' << (padding == Padding::SAME ? "SAME" : "VALID");
  result.set_label(label.str());

  return result;
}

// Set of format and data type configurations to run. For details, see
// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionForward
enum class Config {
  NCHW_FLOAT,
  NCHW_DOUBLE,
  NCHW_PSEUDO_HALF,
  NCHW_TRUE_HALF,
  NCHW_TENSOR_OP,
  NHWC_FLOAT,
  NHWC_PSEUDO_HALF
};

// Returns whether cuDNN supports the given config on the current device.
bool IsConfigSupported(Config config) {
  switch (config) {
    case Config::NCHW_PSEUDO_HALF:
      return DeviceSupportsReducedPrecision();
    case Config::NCHW_TRUE_HALF:
      return DeviceSupportsReducedPrecision();
    case Config::NCHW_TENSOR_OP:
      return CUDNN_MAJOR >= 7 && DeviceSupportsTensorOpMath();
    case Config::NHWC_FLOAT:
      // Internal error on cudNN 6, see cudnn_tests.textproto.
      return CUDNN_MAJOR >= 7;
    case Config::NHWC_PSEUDO_HALF:
      // Crashes on cuDNN 6, see cudnn_tests.textproto.
      return CUDNN_MAJOR >= 7 && DeviceSupportsReducedPrecision();
    default:
      return true;
  }
}

// Creates a ConvolutionConfig from the given config for
// proto.ConvolutionTest.test.
proto::ConvolutionConfig MakeConvolutionConfig(Config config) {
  proto::TensorFormat format;
  proto::DataType data_type;
  proto::DataType compute_mode;
  proto::MathType math_type;
  string label;
  switch (config) {
    case Config::NCHW_FLOAT:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_FLOAT;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_FLOAT";
      break;
    case Config::NCHW_DOUBLE:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_DOUBLE;
      compute_mode = proto::DATA_DOUBLE;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_DOUBLE";
      break;
    case Config::NCHW_PSEUDO_HALF:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_PSEUDO_HALF";
      break;
    case Config::NCHW_TRUE_HALF:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_HALF;
      math_type = proto::DEFAULT_MATH;
      label = "NCHW_TRUE_HALF";
      break;
    case Config::NCHW_TENSOR_OP:
      format = proto::TENSOR_NCHW;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_HALF;
      math_type = proto::TENSOR_OP_MATH;
      label = "NCHW_TENSOR_OP";
      break;
    case Config::NHWC_FLOAT:
      format = proto::TENSOR_NHWC;
      data_type = proto::DATA_FLOAT;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NHWC_FLOAT";
      break;
    case Config::NHWC_PSEUDO_HALF:
      format = proto::TENSOR_NHWC;
      data_type = proto::DATA_HALF;
      compute_mode = proto::DATA_FLOAT;
      math_type = proto::DEFAULT_MATH;
      label = "NHWC_PSEUDO_HALF";
      break;
    default:
      LOG(FATAL) << "Unknown config";
  }

  proto::ConvolutionConfig result;
  result.mutable_input()->set_format(format);
  result.mutable_input()->set_data_type(data_type);
  result.mutable_filter()->set_format(format);
  result.mutable_filter()->set_data_type(data_type);
  result.mutable_convolution()->set_compute_mode(compute_mode);
  result.mutable_convolution()->set_math_type(math_type);
  result.set_label(label);
  return result;
}

// Returns true if cuDNN should handle the given parameters for at least one
// algorithm. Changes to this function will not just drop some tests, but
// completely alter the set of generated tests!
bool ValidateParams(const absl::optional<SizeRange> opt_batch_count,
                    const absl::optional<SizeRange> opt_in_depth,
                    const absl::optional<SizeRange> opt_in_height,
                    const absl::optional<SizeRange> opt_in_width,
                    const absl::optional<SizeRange> opt_out_depth,
                    const absl::optional<SizeRange> opt_filter_height,
                    const absl::optional<SizeRange> opt_filter_width,
                    const absl::optional<Padding> opt_padding,
                    const absl::optional<Config> opt_config) {
  // Default to medium sizes so we don't paint ourselves in a corner.
  SizeRange image_range{32, 32};
  SizeRange filter_range{5, 5};
  SizeRange batch_count = opt_batch_count.value_or(image_range);
  SizeRange in_depth = opt_in_depth.value_or(image_range);
  SizeRange in_height = opt_in_height.value_or(image_range);
  SizeRange in_width = opt_in_width.value_or(image_range);
  SizeRange out_depth = opt_out_depth.value_or(image_range);
  SizeRange filter_height = opt_filter_height.value_or(filter_range);
  SizeRange filter_width = opt_filter_width.value_or(filter_range);
  Padding padding = opt_padding.value_or(Padding::SAME);
  Config config = opt_config.value_or(Config::NCHW_FLOAT);

  size_t million = 1ull << 20;  // 1 million as in MiB.
  size_t max_tensor_elements = static_cast<size_t>(batch_count.maximum) *
                               std::max(in_depth.maximum, out_depth.maximum) *
                               in_height.maximum * in_width.maximum;

  if (max_tensor_elements >= 2048 * million) {
    // cuDNN has a limit of less (!) than 2G elements per tensor.
    return false;
  }

  if (padding == Padding::VALID && (filter_height.minimum > in_height.maximum ||
                                    filter_width.minimum > in_height.maximum)) {
    // VALID padding requires the filter to be no smaller than the input.
    return false;
  }

  auto proto = MakeConvolutionConfig(batch_count.maximum, in_depth.maximum,
                                     in_height.maximum, in_width.maximum,
                                     out_depth.maximum, filter_height.maximum,
                                     filter_width.maximum, padding);

  auto input = CreateTensorDescriptor(proto.input());
  auto filter = CreateFilterDescriptor(proto.filter());
  auto convolution = CreateConvolutionDescriptor(proto.convolution());
  auto output_or =
      CreateOutputDescriptor(proto::TENSOR_NCHW, input, filter, convolution);
  if (!IsOk(output_or)) {
    return false;
  }

  // Compute the size of device allocations in CompareResults and return whether
  // it fits the specified memory limit.

  size_t data_type_size;
  switch (MakeConvolutionConfig(config).filter().data_type()) {
    case proto::DATA_FLOAT:
      data_type_size = sizeof(float);
      break;
    case proto::DATA_DOUBLE:
      data_type_size = sizeof(double);
      break;
    case proto::DATA_HALF:
      data_type_size = sizeof(__half);
      break;
    default:
      LOG(FATAL) << "Not yet supported";
  }

  size_t input_num_elements = GetTensorNumElements(input);
  size_t filter_num_elements = GetFilterNumElements(filter) * sizeof(double);
  size_t output_num_elements = GetTensorNumElements(output_or.ValueOrDie());

  size_t sum_num_elements =
      input_num_elements + filter_num_elements + output_num_elements;
  size_t max_num_elements =
      std::max({input_num_elements, filter_num_elements, output_num_elements});

  // Number of buffers for CompareResults' init_data, ref_result_data and
  // (if conversion from double is needed) ConvertAndTransformTensor.
  size_t num_extra_buffers = data_type_size == sizeof(double) ? 2 : 3;

  size_t required_bytes = (sizeof(double) + data_type_size) * sum_num_elements +
                          num_extra_buffers * data_type_size * max_num_elements;
  if (required_bytes > GetAvailableDeviceMemoryBytes()) {
    return false;
  }

  return true;
}

// Returns a list of tests, one for every direction per generated parameter
// combination.
google::protobuf::RepeatedPtrField<proto::ConvolutionTest> MakeTests(
    const std::vector<SizeRange>& filter_sizes) {
  // Create consecutive closed ranges [2^(k-1)+1, 2^k].
  std::vector<SizeRange> size_ranges;
  for (int size : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
    size_ranges.push_back(MakePowerOfTwoRange(size));
  }
  std::vector<SizeRange> batch_counts = size_ranges;
  std::vector<SizeRange> depths = size_ranges;
  std::vector<SizeRange> image_sizes = size_ranges;
  std::vector<Padding> paddings{Padding::SAME, Padding::VALID};
  std::vector<Config> configs{Config::NCHW_FLOAT,       Config::NCHW_DOUBLE,
                              Config::NCHW_PSEUDO_HALF, Config::NCHW_TRUE_HALF,
                              Config::NCHW_TENSOR_OP,   Config::NHWC_FLOAT,
                              Config::NHWC_PSEUDO_HALF};

  CudnnHandle handle = CreateCudnnHandle();
  auto validator = MakeCallWithTuple(ValidateParams);

  std::mt19937 rand_gen{GetRandomSeed()};
  auto params_vec = MakeAllPairs(rand_gen, validator, batch_counts, depths,
                                 image_sizes, image_sizes, depths, filter_sizes,
                                 filter_sizes, paddings, configs);

  google::protobuf::RepeatedPtrField<proto::ConvolutionTest> conv_tests;
  for (const auto& params : params_vec) {
    MakeCallWithTuple([&](SizeRange batch_count, SizeRange in_depth,
                          SizeRange in_height, SizeRange in_width,
                          SizeRange out_depth, SizeRange filter_height,
                          SizeRange filter_width, Padding padding,
                          Config config) {
      proto::ConvolutionTest conv_test;

      *conv_test.mutable_reference() = MakeConvolutionConfig(
          rand_gen, batch_count, in_depth, in_height, in_width, out_depth,
          filter_height, filter_width, padding);

      *conv_test.add_test() = MakeConvolutionConfig(config);

      // Disable tests that are not supported in the current configuration.
      // Excluding unsupported configs from the parameter vector would make the
      // supported tests depend on the machine configuration, and that seems
      // worse than leaving holes in the all-pairs test space on older hardware
      // and cuDNN versions.
      string label_prefix = IsConfigSupported(config) ? "" : "DISABLED_";
      string label_postfix =
          "_" + conv_test.test(0).label() + "_" + conv_test.reference().label();

      // Add convolution_test for each direction.
      for (auto direction :
           {proto::CONVOLUTION_FWD, proto::CONVOLUTION_BWD_DATA,
            proto::CONVOLUTION_BWD_FILTER}) {
        string direction_name = proto::ConvolutionDirection_Name(direction);
        conv_test.mutable_reference()->set_label(label_prefix + direction_name +
                                                 label_postfix);

        auto& input_dim = conv_test.reference().input().dimension();
        auto& filter_dim = conv_test.reference().filter().dimension();
        size_t num_summands;
        switch (direction) {
          case proto::CONVOLUTION_FWD:
            conv_test.mutable_reference()->set_fwd_algo(
                proto::CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
            // in_depth * filter_height * filter_width.
            num_summands = filter_dim[1] * filter_dim[2] * filter_dim[3];
            break;
          case proto::CONVOLUTION_BWD_DATA:
            conv_test.mutable_reference()->set_bwd_data_algo(
                proto::CONVOLUTION_BWD_DATA_ALGO_1);
            // out_depth * filter_height * filter_width.
            num_summands = filter_dim[0] * filter_dim[2] * filter_dim[3];
            break;
          case proto::CONVOLUTION_BWD_FILTER:
            conv_test.mutable_reference()->set_bwd_filter_algo(
                proto::CONVOLUTION_BWD_FILTER_ALGO_1);
            // batch_count * in_height * in_width.
            num_summands = input_dim[0] * input_dim[2] * input_dim[3];
            break;
          default:
            LOG(FATAL) << "Invalid direction: " << direction_name;
        }
        // Scale result so that elements have an expected value of one. This
        // prevents overflow for half data.
        conv_test.mutable_reference()->set_one_minus_alpha(1.0 -
                                                           4.0 / num_summands);

        for (auto& test : *conv_test.mutable_test()) {
          test.set_all_algos(direction);
        }

        *conv_tests.Add() = conv_test;
      }
    })(params);
  }

  return conv_tests;
}

string GetTestName(const testing::TestParamInfo<proto::ConvolutionTest>& info) {
  return info.param.reference().label();
}

INSTANTIATE_TEST_CASE_P(
    FromFile, ConvolutionTest,
    ::testing::ValuesIn(GetCudnnTestsFromFile().convolution_test()),
    GetTestName);

INSTANTIATE_TEST_CASE_P(Filter3x3, ConvolutionTest, ::testing::ValuesIn([] {
                          std::vector<SizeRange> filter_sizes{{3, 3}};
                          return MakeTests(filter_sizes);
                        }()),
                        GetTestName);

INSTANTIATE_TEST_CASE_P(Filter5x5, ConvolutionTest, ::testing::ValuesIn([] {
                          std::vector<SizeRange> filter_sizes{{5, 5}};
                          return MakeTests(filter_sizes);
                        }()),
                        GetTestName);

INSTANTIATE_TEST_CASE_P(FilterOther, ConvolutionTest, ::testing::ValuesIn([] {
                          std::vector<SizeRange> filter_sizes;
                          for (int size : {1, 2, 4, 8, 16}) {
                            filter_sizes.push_back(MakePowerOfTwoRange(size));
                          }
                          return MakeTests(filter_sizes);
                        }()),
                        GetTestName);
}  // namespace

// Make gtest print the proto as text (instead of raw data) when a test fails.
namespace proto {
::std::ostream& operator<<(::std::ostream& ostr, const ConvolutionTest& proto) {
  return ostr << "\n" << proto.DebugString();
}
}  // namespace proto
}  // namespace nvidia_libs_test

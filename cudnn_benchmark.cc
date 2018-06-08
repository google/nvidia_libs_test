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

// Runs a suite of cuDNN benchmarks, from cudnn_benchmarks.textproto and from
// instances generated in this file. Benchmark time is measured in GPU kernel
// runtime.
//
// Potentially relevant use cases, but not yet benchmarked:
// - Double precision and half precision (pseudo, fp16 math, and tensor ops).
// - NHWC input to NCHW output, and NCHW input to NHWC output.
// - NHWC tensors with NCHW filter (forward only).
// - Using non-default scaling factors (alpha and beta).
//
#include <algorithm>
#include <functional>

#include "gflags/gflags.h"
#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "google/protobuf/repeated_field.h"
#include "benchmark/benchmark.h"
#include "cuda_util.h"
#include "cudnn_util.h"
#include "kernel_timer.h"
#include "load_textproto.h"

DEFINE_string(timing, "kernel-duration",
              "How to measure benchmark time. One of kernel-cycles, "
              "kernel-duration, or host-duration");
DEFINE_string(proto_path, "cudnn_benchmarks.textproto",
              "Path to text proto file containing benchmarks to run.");

namespace nvidia_libs_test {
namespace {
bool has_errors = false;

// If timing GPU cycles or duration, enable manual timing.
void ConfigureTime(benchmark::internal::Benchmark* benchmark) {
  string timing = FLAGS_timing;
  if (timing == "kernel-cycles" || timing == "kernel-duration") {
    benchmark->UseManualTime();
  }
}

// Returns the KernelTimer requested by the --timing flag.
std::unique_ptr<KernelTimer> GetTimer() {
  string timing = FLAGS_timing;
  if (timing == "kernel-cycles") {
    return KernelTimer::CreateCyclesTimer();
  }
  if (timing == "kernel-duration") {
    return KernelTimer::CreateDurationTimer();
  }
  if (timing == "host-duration") {
    return KernelTimer::CreateNopTimer();
  }
  LOG(FATAL) << "Unrecognized 'timing' flag: '" << timing << "', should be one "
             << "of 'kernel-cycles', 'kernel-duration', or 'host-duration'";
  return nullptr;
}

Status ConvolutionBenchmark(benchmark::State& state,
                            proto::ConvolutionConfig proto) {
  if (CUDNN_MAJOR < 7 && proto.convolution().group_count() > 1) {
    return ErrorStatus("Skipped: Grouped convolution requires cuDNN 7");
  }

  CudnnHandle handle = CreateCudnnHandle();
  RandomGenerator rand_gen(/*seed=*/0);

  ASSIGN_OR_RETURN_STATUS(Convolution benchmark,
                          CreateConvolution(proto, rand_gen));

  ASSIGN_OR_RETURN_STATUS(size_t workspace_limit, GetWorkspaceLimit(proto));

  ConvolutionAlgo algo;
  switch (proto.algo_oneof_case()) {
    case proto::ConvolutionConfig::kFindAlgo: {
      ASSIGN_OR_RETURN_STATUS(
          algo,
          FindConvolutionAlgo(handle, proto.find_algo(), benchmark.input_desc,
                              benchmark.input_data, benchmark.filter_desc,
                              benchmark.filter_data, benchmark.conv_desc,
                              benchmark.output_desc, benchmark.output_data,
                              workspace_limit));
    } break;
    case proto::ConvolutionConfig::kFwdAlgo:
      algo = static_cast<cudnnConvolutionFwdAlgo_t>(proto.fwd_algo());
      break;
    case proto::ConvolutionConfig::kBwdDataAlgo:
      algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(proto.bwd_data_algo());
      break;
    case proto::ConvolutionConfig::kBwdFilterAlgo:
      algo =
          static_cast<cudnnConvolutionBwdFilterAlgo_t>(proto.bwd_filter_algo());
      break;
    default:
      LOG(FATAL) << "Invalid algo_oneof_case.";
  }

  ASSIGN_OR_RETURN_STATUS(
      size_t workspace_size,
      GetWorkspaceSize(handle, benchmark.input_desc, benchmark.filter_desc,
                       benchmark.conv_desc, benchmark.output_desc, algo));

  ASSIGN_OR_RETURN_STATUS(auto workspace, AllocateDeviceMemory(workspace_size));

  double alpha = 1.0 - proto.one_minus_alpha();
  double beta = proto.beta();

  auto timer = GetTimer();
  timer->StartTiming();
  while (state.KeepRunning()) {
    RETURN_IF_ERROR_STATUS(RunConvolution(
        handle, algo, alpha, beta, benchmark.input_desc, benchmark.input_data,
        benchmark.filter_desc, benchmark.filter_data, benchmark.conv_desc,
        benchmark.output_desc, benchmark.output_data, workspace,
        workspace_size));
    RETURN_IF_ERROR_STATUS(GetStatus(cudaDeviceSynchronize()));
    state.SetIterationTime(timer->GetTime());
    timer->ResetTime();
  }
  timer->StopTiming();

  std::ostringstream oss;
  // proto.clear_algo_oneof() has private accessibility, set oneof member and
  // clear it again instead.
  proto.set_find_algo(proto::CONVOLUTION_DIRECTION_UNSPECIFIED);
  proto.clear_find_algo();
  proto.clear_label();
  oss << "algo: " << algo << " " << proto.ShortDebugString()
      << " workspace_size: " << workspace_size;
  state.SetLabel(oss.str());

  return OkStatus();
}

Status TransformationBenchmark(benchmark::State& state,
                               const proto::TensorDescriptor& first,
                               const proto::TensorDescriptor& second) {
  CudnnHandle handle = CreateCudnnHandle();
  RandomGenerator rand_gen(/*seed=*/0);

  TensorDescriptor first_desc = CreateTensorDescriptor(first);
  TensorDescriptor second_desc = CreateTensorDescriptor(second);

  ASSIGN_OR_RETURN_STATUS(auto first_data,
                          CreateTensorData(first_desc, rand_gen));

  ASSIGN_OR_RETURN_STATUS(auto second_data,
                          CreateTensorData(second_desc, rand_gen));

  auto kernel_timer = GetTimer();
  kernel_timer->StartTiming();
  while (state.KeepRunning()) {
    RETURN_IF_ERROR_STATUS(TransformTensor(handle, first_desc, first_data,
                                           second_desc, second_data));
    RETURN_IF_ERROR_STATUS(GetStatus(cudaDeviceSynchronize()));
    state.SetIterationTime(kernel_timer->GetTime());
    kernel_timer->ResetTime();
  }
  kernel_timer->StopTiming();

  std::ostringstream oss;
  for (int dim : first.dimension()) {
    oss << "x" << dim;
  }
  state.SetLabel(oss.str().substr(1));

  return OkStatus();
}

proto::Benchmarks GetCudnnBenchmarksFromFile() {
  proto::Benchmarks benchmarks;
  LoadTextProto(FLAGS_proto_path, &benchmarks);
  return benchmarks;
}

google::protobuf::RepeatedPtrField<proto::ConvolutionConfig> GetTensorFlowBenchmarks() {
  // SAME cooresponds to the amount padding so that input and output image
  // have the same width and height. VALID corresponds to no padding.
  enum Padding { SAME, VALID };

  google::protobuf::RepeatedPtrField<proto::ConvolutionConfig> benchmarks;

  auto add_benchmark = [&](int batch, int in_height, int in_width,
                           int in_channels, int out_channels, int filter_height,
                           int filter_width, int unused_stride, Padding padding,
                           const string& label_suffix) {
    proto::TensorDescriptor input;
    input.add_dimension(batch);
    input.add_dimension(in_channels);
    input.add_dimension(in_height);
    input.add_dimension(in_width);
    input.set_data_type(proto::DATA_FLOAT);

    proto::FilterDescriptor filter;
    filter.add_dimension(out_channels);
    filter.add_dimension(in_channels);
    filter.add_dimension(filter_height);
    filter.add_dimension(filter_width);
    filter.set_data_type(proto::DATA_FLOAT);

    proto::ConvolutionDescriptor convolution;
    convolution.add_pad(padding == SAME ? filter_height / 2 : 0);
    convolution.add_pad(padding == SAME ? filter_width / 2 : 0);
    convolution.set_compute_mode(proto::DATA_FLOAT);
    if (CUDNN_MAJOR >= 7) {
      convolution.set_math_type(proto::TENSOR_OP_MATH);
    }

    proto::ConvolutionConfig benchmark;
    *benchmark.mutable_input() = std::move(input);
    *benchmark.mutable_filter() = std::move(filter);
    *benchmark.mutable_convolution() = std::move(convolution);

    for (auto direction : {proto::CONVOLUTION_FWD, proto::CONVOLUTION_BWD_DATA,
                           proto::CONVOLUTION_BWD_FILTER}) {
      for (auto format : {proto::TENSOR_NCHW, proto::TENSOR_NHWC}) {
        benchmark.set_find_algo(direction);
        benchmark.mutable_input()->set_format(format);
        // Use the same format for the filter as well.
        //
        // For NHWC forward convolution, this is usually faster than with a
        // NCHW filter but has certain restrictions (see cuDNN documentation for
        // details).
        //
        // For NHWC backward data and filter convolution, this is required.
        benchmark.mutable_filter()->set_format(format);
        benchmark.set_label(proto::ConvolutionDirection_Name(direction) + '/' +
                            proto::TensorFormat_Name(format) + '/' +
                            label_suffix);
        *benchmarks.Add() = benchmark;
      }
    }
  };
  add_benchmark(32, 5, 5, 1248, 128, 1, 1, 1, SAME, "conv0");
  add_benchmark(32, 8, 8, 384, 384, 1, 3, 1, SAME, "conv1");
  add_benchmark(32, 8, 8, 384, 384, 3, 1, 1, SAME, "conv2");
  add_benchmark(32, 8, 8, 2048, 192, 1, 1, 1, SAME, "conv3");
  add_benchmark(32, 8, 8, 448, 384, 3, 3, 1, SAME, "conv4");
  add_benchmark(32, 8, 8, 2048, 320, 1, 1, 1, SAME, "conv5");
  add_benchmark(32, 8, 8, 2048, 448, 1, 1, 1, SAME, "conv6");
  add_benchmark(32, 8, 8, 2048, 384, 1, 1, 1, SAME, "conv7");
  add_benchmark(32, 8, 8, 1760, 384, 1, 1, 1, SAME, "conv8");
  add_benchmark(32, 8, 8, 1760, 192, 1, 1, 1, SAME, "conv9");
  add_benchmark(32, 8, 8, 1760, 448, 1, 1, 1, SAME, "conv10");
  add_benchmark(32, 8, 8, 1760, 320, 1, 1, 1, SAME, "conv11");
  add_benchmark(32, 17, 17, 192, 192, 3, 3, 2, VALID, "conv12");
  add_benchmark(32, 17, 17, 192, 192, 3, 3, 1, SAME, "conv13");
  add_benchmark(32, 17, 17, 1248, 192, 1, 1, 1, SAME, "conv14");
  add_benchmark(32, 17, 17, 128, 320, 3, 3, 2, VALID, "conv15");
  add_benchmark(32, 17, 17, 1248, 128, 1, 1, 1, SAME, "conv16");
  add_benchmark(32, 17, 17, 224, 224, 1, 3, 1, SAME, "conv17");
  add_benchmark(32, 17, 17, 192, 256, 3, 1, 1, SAME, "conv18");
  add_benchmark(32, 17, 17, 192, 256, 1, 3, 1, SAME, "conv19");
  add_benchmark(32, 17, 17, 1216, 192, 1, 1, 1, SAME, "conv20");
  add_benchmark(32, 17, 17, 1216, 96, 1, 1, 1, SAME, "conv21");
  add_benchmark(32, 17, 17, 224, 224, 3, 1, 1, SAME, "conv22");
  add_benchmark(32, 17, 17, 192, 224, 3, 3, 1, SAME, "conv23");
  add_benchmark(32, 17, 17, 192, 192, 1, 3, 1, SAME, "conv24");
  add_benchmark(32, 17, 17, 1152, 192, 1, 1, 1, SAME, "conv25");
  add_benchmark(32, 17, 17, 1152, 128, 1, 1, 1, SAME, "conv26");
  add_benchmark(32, 17, 17, 192, 192, 3, 1, 1, SAME, "conv27");
  add_benchmark(32, 17, 17, 160, 192, 3, 3, 1, SAME, "conv28");
  add_benchmark(32, 17, 17, 1152, 160, 1, 1, 1, SAME, "conv29");
  add_benchmark(32, 17, 17, 1024, 128, 1, 1, 1, SAME, "conv30");
  add_benchmark(32, 17, 17, 128, 192, 1, 3, 1, SAME, "conv31");
  add_benchmark(32, 17, 17, 1024, 160, 1, 1, 1, SAME, "conv32");
  add_benchmark(32, 17, 17, 128, 192, 3, 1, 1, SAME, "conv33");
  add_benchmark(32, 17, 17, 1024, 256, 1, 1, 1, SAME, "conv34");
  add_benchmark(32, 17, 17, 128, 128, 3, 1, 1, SAME, "conv35");
  add_benchmark(32, 17, 17, 768, 192, 1, 1, 1, SAME, "conv36");
  add_benchmark(32, 17, 17, 128, 128, 1, 3, 1, SAME, "conv37");
  add_benchmark(32, 17, 17, 128, 128, 3, 3, 1, SAME, "conv38");
  add_benchmark(32, 17, 17, 768, 128, 1, 1, 1, SAME, "conv39");
  add_benchmark(32, 17, 17, 768, 320, 1, 1, 1, SAME, "conv40");
  add_benchmark(32, 35, 35, 96, 96, 3, 3, 2, VALID, "conv41");
  add_benchmark(32, 35, 35, 288, 384, 3, 3, 2, VALID, "conv42");
  add_benchmark(32, 35, 35, 64, 96, 3, 3, 1, SAME, "conv43");
  add_benchmark(32, 35, 35, 288, 64, 1, 1, 1, SAME, "conv44");
  add_benchmark(32, 35, 35, 256, 64, 1, 1, 1, SAME, "conv45");
  add_benchmark(32, 35, 35, 48, 64, 5, 5, 1, SAME, "conv46");
  add_benchmark(32, 35, 35, 256, 48, 1, 1, 1, SAME, "conv47");
  add_benchmark(32, 35, 35, 96, 96, 3, 3, 1, SAME, "conv48");
  add_benchmark(32, 35, 35, 192, 32, 1, 1, 1, SAME, "conv49");
  add_benchmark(32, 35, 35, 192, 64, 1, 1, 1, SAME, "conv50");
  add_benchmark(32, 35, 35, 192, 48, 1, 1, 1, SAME, "conv51");
  add_benchmark(32, 73, 73, 64, 192, 3, 3, 1, VALID, "conv52");
  add_benchmark(32, 73, 73, 64, 64, 1, 1, 1, VALID, "conv53");
  add_benchmark(32, 147, 147, 24, 64, 1, 1, 1, VALID, "conv54");

  std::stable_sort(benchmarks.begin(), benchmarks.end(),
                   [](const proto::ConvolutionConfig& left,
                      const proto::ConvolutionConfig& right) {
                     return left.find_algo() < right.find_algo();
                   });

  return benchmarks;
}

google::protobuf::RepeatedPtrField<proto::ConvolutionConfig> GetDepthwiseBenchmarks() {
  google::protobuf::RepeatedPtrField<proto::ConvolutionConfig> benchmarks;

  if (CUDNN_MAJOR < 7) {
    return benchmarks;  // No grouped convolution before cuDNN 7.
  }

  auto add_benchmark = [&](int batch, int in_height, int in_width,
                           int in_channels, int out_channels, int filter_height,
                           int filter_width, const string& label_suffix) {
    proto::TensorDescriptor input;
    input.add_dimension(batch);
    input.add_dimension(in_channels);
    input.add_dimension(in_height);
    input.add_dimension(in_width);

    proto::FilterDescriptor filter;
    filter.add_dimension(out_channels);
    filter.add_dimension(1);
    filter.add_dimension(filter_height);
    filter.add_dimension(filter_width);

    proto::ConvolutionDescriptor convolution;
    convolution.add_pad(filter_height / 2);
    convolution.add_pad(filter_width / 2);
    convolution.set_group_count(in_channels);
    // Note: tensor cores are not currently (cuDNN 7.1) supported for grouped
    // convolutions.
    convolution.set_math_type(proto::TENSOR_OP_MATH);

    proto::TensorDescriptor output;
    output.add_dimension(batch);
    output.add_dimension(out_channels);
    output.add_dimension(in_height);
    output.add_dimension(in_width);

    proto::ConvolutionConfig benchmark;
    *benchmark.mutable_input() = std::move(input);
    *benchmark.mutable_filter() = std::move(filter);
    *benchmark.mutable_convolution() = std::move(convolution);
    *benchmark.mutable_output() = std::move(output);

    for (auto direction : {proto::CONVOLUTION_FWD, proto::CONVOLUTION_BWD_DATA,
                           proto::CONVOLUTION_BWD_FILTER}) {
      for (auto data_type : {proto::DATA_FLOAT, proto::DATA_HALF}) {
        benchmark.mutable_input()->set_data_type(data_type);
        benchmark.mutable_filter()->set_data_type(data_type);
        benchmark.mutable_output()->set_data_type(data_type);
        std::vector<proto::DataType> compute_modes = {proto::DATA_FLOAT};
        if (data_type == proto::DATA_HALF) {
          compute_modes.push_back(proto::DATA_HALF);
        }
        for (auto compute_mode : compute_modes) {
          // Note: true half configuration is terribly slow.
          benchmark.mutable_convolution()->set_compute_mode(compute_mode);
          std::vector<proto::TensorFormat> formats = {proto::TENSOR_NCHW};
          if (direction == proto::CONVOLUTION_FWD) {
            formats.push_back(proto::TENSOR_NHWC);
          }
          for (auto format : formats) {
            benchmark.set_find_algo(direction);
            benchmark.mutable_input()->set_format(format);
            // Note: OHWI filter format is not supported for grouped
            // convolutions.
            benchmark.mutable_filter()->set_format(proto::TENSOR_NCHW);
            benchmark.mutable_output()->set_format(format);
            benchmark.set_label(proto::ConvolutionDirection_Name(direction) +
                                '/' + proto::TensorFormat_Name(format) + '/' +
                                proto::DataType_Name(data_type) + '/' +
                                proto::DataType_Name(compute_mode) + '/' +
                                label_suffix);
            *benchmarks.Add() = benchmark;
          }
        }
      }
    }
  };

  add_benchmark(128, 64, 64, 32, 32, 3, 3, "depthwise0");

  std::stable_sort(benchmarks.begin(), benchmarks.end(),
                   [](const proto::ConvolutionConfig& left,
                      const proto::ConvolutionConfig& right) {
                     return left.label() < right.label();
                   });

  return benchmarks;
}

void RegisterConvolutionBenchmarks(
    const string& prefix,
    const google::protobuf::RepeatedPtrField<proto::ConvolutionConfig>& benchmarks) {
  for (const auto& benchmark : benchmarks) {
    ConfigureTime(benchmark::RegisterBenchmark(
        ("BM_CudnnConvolution/" + prefix + "/" + benchmark.label()).c_str(),
        [benchmark](benchmark::State& state) {
          auto status = ConvolutionBenchmark(state, benchmark);
          if (!status.ok()) {
            state.SkipWithError(status.message().c_str());
            ResetDevice();
          }
        }));
  }
}

void RegisterTransformationBenchmarks() {
  auto add_benchmark = [](int batch, int height, int width, int channels,
                          const string& label_suffix) {
    auto runner = [](benchmark::State& state,
                     const proto::TensorDescriptor& first,
                     const proto::TensorDescriptor& second) {
      auto status = TransformationBenchmark(state, first, second);
      if (!status.ok()) {
        state.SkipWithError(status.message().c_str());
        ResetDevice();
      }
    };

    proto::TensorDescriptor nchw;
    nchw.add_dimension(batch);
    nchw.add_dimension(channels);
    nchw.add_dimension(height);
    nchw.add_dimension(width);
    nchw.set_format(proto::TENSOR_NCHW);
    nchw.set_data_type(proto::DATA_FLOAT);

    proto::TensorDescriptor nhwc = nchw;
    nchw.set_format(proto::TENSOR_NHWC);

    ConfigureTime(benchmark::RegisterBenchmark(
        ("BM_CudnnTransformation/NchwToNhwc/" + label_suffix).c_str(),
        std::bind(runner, std::placeholders::_1, nchw, nhwc)));

    ConfigureTime(benchmark::RegisterBenchmark(
        ("BM_CudnnTransformation/NhwcToNchw/" + label_suffix).c_str(),
        std::bind(runner, std::placeholders::_1, nhwc, nchw)));
  };

  // Uses the same tensor sizes and indices as GetTensorFlowBenchmarks() above.
  add_benchmark(32, 5, 5, 1248, "trans0");
  add_benchmark(32, 8, 8, 384, "trans1");
  add_benchmark(32, 8, 8, 2048, "trans3");
  add_benchmark(32, 8, 8, 448, "trans4");
  add_benchmark(32, 8, 8, 1760, "trans8");
  add_benchmark(32, 17, 17, 192, "trans12");
  add_benchmark(32, 17, 17, 1248, "trans14");
  add_benchmark(32, 17, 17, 128, "trans15");
  add_benchmark(32, 17, 17, 224, "trans17");
  add_benchmark(32, 17, 17, 1216, "trans20");
  add_benchmark(32, 17, 17, 1152, "trans25");
  add_benchmark(32, 17, 17, 160, "trans28");
  add_benchmark(32, 17, 17, 1024, "trans30");
  add_benchmark(32, 17, 17, 768, "trans36");
  add_benchmark(32, 35, 35, 96, "trans41");
  add_benchmark(32, 35, 35, 288, "trans42");
  add_benchmark(32, 35, 35, 64, "trans43");
  add_benchmark(32, 35, 35, 256, "trans45");
  add_benchmark(32, 35, 35, 48, "trans46");
  add_benchmark(32, 35, 35, 192, "trans49");
  add_benchmark(32, 73, 73, 64, "trans52");
  add_benchmark(32, 147, 147, 24, "trans54");
}

void RegisterBenchmarks() {
  RegisterConvolutionBenchmarks(
      "FromFile", GetCudnnBenchmarksFromFile().convolution_benchmark());
  RegisterConvolutionBenchmarks("TensorFlow", GetTensorFlowBenchmarks());
  RegisterConvolutionBenchmarks("Depthwise", GetDepthwiseBenchmarks());
  RegisterTransformationBenchmarks();
}

}  // namespace
}  // namespace nvidia_libs_test

int main(int argc, char** argv) {
  // Initialize benchmarks before parsing flags. Both consume recognized flags,
  // but only the latter reports an error if a flag is not recognized.
  benchmark::Initialize(&argc, argv);
  // Parse flags before initializing logging, otherwise logs are not printed.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  nvidia_libs_test::RegisterBenchmarks();
  benchmark::RunSpecifiedBenchmarks();
  return nvidia_libs_test::has_errors;
}

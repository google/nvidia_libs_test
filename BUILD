licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

exports_files(["LICENSE"])

cc_library(
    name = "all_pairs",
    hdrs = ["all_pairs.h"],
    deps = [
        ":glog",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/utility",
    ],
)

cc_test(
    name = "all_pairs_test",
    size = "small",
    srcs = [
        "all_pairs_test.cc",
    ],
    deps = [
        ":all_pairs",
        ":glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cuda_util",
    srcs = [
        "cuda_util.cc",
        "cuda_util_cu",
        "status.h",
    ],
    hdrs = [
        "cuda_util.h",
    ],
    deps = [
        ":glog",
        "@com_google_absl//absl/types:optional",
        "@local_config_cuda//:cuda_headers",
        "@local_config_cuda//:cuda_runtime",
        "@local_config_cuda//:cupti",
        "@local_config_cuda//:cupti_headers",
        "@local_config_cuda//:curand_static",
    ],
)

cc_library(
    name = "kernel_timer",
    srcs = ["kernel_timer.cc"],
    hdrs = ["kernel_timer.h"],
    deps = [
        ":cuda_util",
        ":glog",
        "@local_config_cuda//:cuda_headers",
        "@local_config_cuda//:cupti_headers",
    ],
)

proto_library(
    name = "cudnn_proto",
    srcs = ["cudnn.proto"],
)

cc_proto_library(
    name = "cudnn_cc_proto",
    deps = [":cudnn_proto"],
)

cc_library(
    name = "cudnn_util",
    srcs = [
        "cudnn_util.cc",
    ],
    hdrs = ["cudnn_util.h"],
    deps = [
        ":cuda_util",
        ":cudnn_cc_proto",
        ":glog",
        "@com_google_absl//absl/types:variant",
        "@local_config_cuda//:cuda_headers",
        "@local_config_cuda//:cuda_runtime",
        "@local_config_cuda//:cudnn",
    ],
)

# Wraps glog and provides a work-around header.
cc_library(
    name = "glog",
    hdrs = ["ostream_nullptr.h"],
    deps = ["@com_google_glog//:glog"],
)

# See README.md for bazel command lines to build and run benchmarks.
cc_binary(
    name = "cudnn_benchmark",
    srcs = [
        "cudnn_benchmark.cc",
        "load_textproto.h",
    ],
    data = glob(["*.textproto"]),
    deps = [
        ":cuda_util",
        ":cudnn_cc_proto",
        ":cudnn_util",
        ":glog",
        ":kernel_timer",
        "@com_google_benchmark//:benchmark",
        "@com_google_protobuf//:protobuf_lite",
    ],
)

# See README.md for bazel command lines to build and run tests.
cc_test(
    name = "cudnn_test",
    size = "enormous",
    srcs = [
        "cudnn_conv_test.cc",
        "cudnn_test.cc",
        "cudnn_test.h",
        "load_textproto.h",
        "test_util.cc",
        "test_util.h",
    ],
    data = glob(["*.textproto"]),
    linkstatic = 1,
    deps = [
        ":all_pairs",
        ":cuda_util",
        ":cudnn_cc_proto",
        ":cudnn_util",
        ":glog",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf_lite",
        "@local_config_cuda//:cuda_headers",
    ],
)

genrule(
    name = "cuda_util_cu",
    srcs = ["cuda_util.cu"],
    outs = ["cuda_util.o"],
    cmd = "$(location @local_config_cuda//:nvcc) --compile $(SRCS) --output-file $(OUTS)",
    tools = ["@local_config_cuda//:nvcc"],
)

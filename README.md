# Tests and Benchmarks for cuDNN

The repository contains a set of convolution tests and benchmarks for NVIDIA's
[cuDNN](https://developer.nvidia.com/cudnn) library.

This is not an officially supported Google product.

## Prerequisites

Install bazel
([instructions](https://docs.bazel.build/versions/master/install.html)).

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (CUDA 8 is
the minimal supported version)

Install the [cuDNN SDK](https://developer.nvidia.com/cudnn) (cuDNN 6 is the
minimal supported version).

## Common parameters

Bazel parameters:

*   `-c opt` Build with optimizations. Recommended for benchmarks.

*   `--action_env=CUDA_PATH=<path>`: Path to the CUDA SDK directory. Default is
    `/usr/loca/cuda`.

*   `--action_env=CUDNN_PATH=<path>`: Path to the CUDNN SDK directory. Default
    is `CUDA_PATH`.

Executable parameters:

*   `--cuda_device=<device_id>`: CUDA device to use. Default is 0.

*   `--device_memory_limit_mb=<size>`: Upper limit of device memory (in
    megabytes) to use for cuDNN workspace after tensors have been allocated.
    Negative values specify an offset from the memory available at startup.
    Default is 4096.

## Test instructions

`bazel run [bazel parameters] //:cudnn_test -- [test parameters]`

Test parameters:

*   `--gtest_filter=<pattern>`: Only run tests that match the given pattern.
    Example pattern: `'*FWD*'`.

*   `--proto_path=<path>`: Path to textproto file that contains additional
    convolutions to run. Default is `'cudnn_tests.textproto'`.

*   `--gtest_random_seed=<value>`: Seed for random generator. Changing the value
    produces tests with a different mix of tensor shapes, filter sizes, etc.
    Default is 0.

*   `--help` for more options.

## Benchmark instructions

`bazel run [bazel parameters] //:cudnn_benchmark -- [benchmark parameters]`

Benchmark parameters:

*   `--benchmark_filter=<regex>`: Only run tests that match the given regex
    pattern. Example: `'BWD_DATA'`.

*   `--proto_path=<path>`: Path to textproto file that contains additional
    convolutions to run. Default is `'cudnn_benchmarks.textproto'`.

*   `--timing=<method>`: How to measure execution time. One of
    `'kernel-duration'` (default), `'kernel-cycles'`, or `'host-duration'`.

*   `--help` and `--helpfull` for more options.

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

#ifndef NVIDIA_LIBS_TEST_CUDNN_TEST_H_
#define NVIDIA_LIBS_TEST_CUDNN_TEST_H_

#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "cuda_util.h"
#include "cudnn_util.h"

// Helper functions that are common to all cuDNN convolution tests.

namespace nvidia_libs_test {
// Returns error status if not all tensor elements are within tolerance. See
// DeviceDataEqual() for the definition of the distance.
Status TensorDataEqual(const DeviceMemory& first, const DeviceMemory& second,
                       const TensorDescriptor& descriptor, double tolerance);

// Returns the tests from the textproto specified by FLAGS_proto_path.
proto::Tests GetCudnnTestsFromFile();

// SAME cooresponds to the amount padding so that input and output image
// have the same width and height. VALID corresponds to no padding.
enum class Padding { SAME, VALID };

std::ostream& operator<<(std::ostream& ostr, Padding padding);

namespace proto {
// Make gtest print protos as text (instead of raw data).
template <typename T>
typename std::enable_if<
    std::is_convertible<T*, ::google::protobuf::Message*>::value,
    ::std::ostream&>::type
operator<<(::std::ostream& ostr, const T& proto) {
  return ostr << "\n" << proto.DebugString();
}
}  // namespace proto
}  // namespace nvidia_libs_test
#endif  // NVIDIA_LIBS_TEST_CUDNN_TEST_H_

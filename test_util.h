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

#ifndef NVIDIA_LIBS_TEST_TEST_UTIL_H_
#define NVIDIA_LIBS_TEST_TEST_UTIL_H_

#include <cstddef>
#include <string>

#include "gtest/gtest.h"
#include "cuda_util.h"

// Utility function to compare device data.

namespace nvidia_libs_test {
// Returns the --gtest_random_seed flag value.
uint_fast32_t GetRandomSeed();

// Compares consecutive elements in first and second point to device memory, and
// returns the number of elements that are further than tolerance apart.
//
// The distance between two elements is computed as function that evaluates to
// the absolute distance for tiny values and relative distance for huge values:
// distance(a, b) = |a-b| / (max(|a|, |b|) + 1).
Status DeviceDataEqual(const float* first, const float* second,
                       size_t num_elements, double tolerance);
Status DeviceDataEqual(const double* first, const double* second,
                       size_t num_elements, double tolerance);
Status DeviceDataEqual(const __half* first, const __half* second,
                       size_t num_elements, double tolerance);

// Converts a Status into an AssertionResult.
::testing::AssertionResult IsOk(const Status& status);

// Converts a StatusOr<T> into an AssertionResult.
template <typename T>
::testing::AssertionResult IsOk(const StatusOr<T>& status_or) {
  return IsOk(status_or.status());
}

#define ASSERT_OK_AND_ASSIGN(lhs, expr) \
  ASSERT_OK_AND_ASSIGN_IMPL(CONCAT(_status_or_, __COUNTER__), lhs, expr)

#define ASSERT_OK_AND_ASSIGN_IMPL(status_or, lhs, expr) \
  auto status_or = expr;                                \
  ASSERT_TRUE(IsOk(status_or));                         \
  lhs = std::move(status_or.ValueOrDie())

}  // namespace nvidia_libs_test

#endif  // NVIDIA_LIBS_TEST_TEST_UTIL_H_

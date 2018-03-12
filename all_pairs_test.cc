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

#include "all_pairs.h"
#include "gtest/gtest.h"

namespace nvidia_libs_test {
namespace {

// Simple end-to-end test checking whether all pairs are generated and whether
// the validator is respected.
TEST(AllPairsTest, GeneratesAllValidPairs) {
  const size_t n0 = 4;
  const size_t n1 = 6;
  const size_t n2 = 3;

  auto validator = MakeCallWithTuple([](absl::optional<unsigned> op0,
                                        absl::optional<unsigned> op1,
                                        absl::optional<unsigned> op2) {
    unsigned p0 = op0.value_or(~0u);
    unsigned p1 = op1.value_or(~0u);
    unsigned p2 = op2.value_or(~0u);
    if (!p0) return false;                   // One invalid 0-parameter.
    if (!p1 && !p2) return false;            // One invalid 1/2-parameter pair.
    if (p0 == p1 && p1 == p2) return false;  // Should not affect count.
    return true;
  });

  auto make_vector = [](size_t n) {
    std::vector<unsigned> result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
  };

  auto tuples = MakeAllPairs(std::mt19937{}, validator, make_vector(n0),
                             make_vector(n1), make_vector(n2));

  // {first value, second value, key of which parameters they refer to}.
  using PairKey = std::array<unsigned, 3>;

  std::set<PairKey> pairs;
  for (const auto& tuple : tuples) {
    EXPECT_TRUE(validator(tuple));
    unsigned p0 = std::get<0>(tuple);
    unsigned p1 = std::get<1>(tuple);
    unsigned p2 = std::get<2>(tuple);
    EXPECT_LT(p0, n0);
    EXPECT_LT(p1, n1);
    EXPECT_LT(p2, n2);
    pairs.insert({{p0, p1, 0}});
    pairs.insert({{p0, p2, 1}});
    pairs.insert({{p1, p2, 2}});
  }

  EXPECT_EQ(pairs.size(), (n0 - 1) * (n1 + n2) + n1 * n2 - 1);
}

}  // namespace
}  // namespace nvidia_libs_test

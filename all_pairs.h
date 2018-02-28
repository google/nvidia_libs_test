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

#ifndef NVIDIA_LIBS_TEST_ALL_PAIRS_H_
#define NVIDIA_LIBS_TEST_ALL_PAIRS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <forward_list>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "absl/types/optional.h"
#include "absl/utility/utility.h"

// All-pairs test generator. From a multi-dimensional parameter space, the suite
// of pairwise-generated tests covers all combinations of two parameters.
// See http://pairwise.org for more information.

namespace nvidia_libs_test {
namespace detail {

// Simplistic pairwise test generator. Produces roughly double the number
// of tests than state of the art and is not fast, but does the job.
//
// Returns a vector of valid parameter tuples so that for each distinct pair of
// rows in the 2D array parameter_values, all valid combinations of element
// pairs are present at least once. 'Valid' parameters in this context means
// that passing them to the validator returns true.
template <typename RandGen, typename V, typename... ParamTypes, size_t... Is>
std::vector<std::tuple<ParamTypes...>> MakeAllPairs(
    RandGen&& rand_gen, const V& validator,
    const std::tuple<std::vector<ParamTypes>...>& param_values,
    absl::index_sequence<Is...>) {
  constexpr size_t N = sizeof...(ParamTypes);
  // Contains the indices into the rows of param_values. An element value of
  // 'kUnset' corresponds to an unspecified parameter.
  using ParamIndices = std::array<size_t, N>;

  // The length of each row in param_values.
  const ParamIndices counts = {{std::get<Is>(param_values).size()...}};
  CHECK_GT(*std::min_element(counts.begin(), counts.end()), 0);
  // Constant for a parameter that has not been fixed yet.
  const size_t kUnset = std::numeric_limits<size_t>::max();
  // If there is only one choice for a parameter, fix it already.
  const ParamIndices unset_indices = {{(counts[Is] > 1 ? kUnset : 0)...}};

  // Specifies two parameters as param_values[param][idx].
  struct IndexPair {
    size_t param1;
    size_t idx1;
    size_t param2;
    size_t idx2;
  };

  // Generate all combinations of index pairs.
  std::vector<IndexPair> index_pairs;
  for (size_t param1 = 0; param1 < N - 1; ++param1) {
    for (size_t param2 = param1 + 1; param2 < N; ++param2) {
      for (size_t idx1 = 0; idx1 < counts[param1]; ++idx1) {
        for (size_t idx2 = 0; idx2 < counts[param2]; ++idx2) {
          index_pairs.push_back({param1, idx1, param2, idx2});
        }
      }
    }
  }

  // Maps indices to optional parameters and calls the validator.
  auto validate = [&](const ParamIndices& indices) {
    return validator(std::make_tuple(
        (indices[Is] != kUnset
             ? absl::make_optional(std::get<Is>(param_values)[indices[Is]])
             : absl::nullopt)...));
  };

  // Tries to merge indices with an index pair, returns true if result is valid.
  auto try_add = [&](ParamIndices& indices, const IndexPair& pair) {
    if ((indices[pair.param1] != kUnset && indices[pair.param1] != pair.idx1) ||
        (indices[pair.param2] != kUnset && indices[pair.param2] != pair.idx2)) {
      return false;
    }
    ParamIndices tester = indices;
    tester[pair.param1] = pair.idx1;
    tester[pair.param2] = pair.idx2;
    if (!validate(tester)) {
      return false;
    }
    indices = tester;
    return true;
  };

  // Filters index pairs that aren't valid just by themselves.
  auto filter = [&](const IndexPair& pair) {
    ParamIndices tester = unset_indices;
    return !try_add(tester, pair);
  };
  index_pairs.erase(
      std::remove_if(index_pairs.begin(), index_pairs.end(), filter),
      index_pairs.end());
  // Randomly shuffle remaining index pairs.
  std::shuffle(index_pairs.begin(), index_pairs.end(), rand_gen);

  // Add pairs to list of indices, resolving invalid combinations with linear
  // probing, inserting a new element if all current combinations are invalid.
  std::forward_list<ParamIndices> indices_list(1, unset_indices);
  auto out_it = indices_list.begin();
  for (auto pair_it = index_pairs.begin(); pair_it != index_pairs.end();
       ++pair_it) {
    auto start_it = out_it;

    while (!try_add(*out_it, *pair_it)) {
      if (++out_it == indices_list.end()) {
        out_it = indices_list.begin();
      }
      if (out_it == start_it) {
        // Add a new clean tuple, because adding the current pair renders all
        // existing parameter tuples invalid. The pair will be assigned to it in
        // try_add() above.
        out_it = indices_list.insert_after(out_it, unset_indices);
      }
    }
  }

  // At this point, all index_pairs have been consumed, but some indices are
  // not fully specified yet, i.e. have kUnset elements. We replace them with
  // a random index while keeping the tuple valid.
  size_t num_elements_erased = 0;
  for (auto prev = indices_list.before_begin();
       std::next(prev) != indices_list.end();) {
    auto it = std::next(prev);
    for (size_t param = 0; param < N; ++param) {
      if ((*it)[param] != kUnset) {
        continue;
      }
      // Generate an iota range, shuffle it, and assign first valid element.
      std::vector<size_t> shuffled_range(counts[param]);
      std::iota(shuffled_range.begin(), shuffled_range.end(), 0);
      std::shuffle(shuffled_range.begin(), shuffled_range.end(), rand_gen);
      auto valid_it = std::find_if(shuffled_range.begin(), shuffled_range.end(),
                                   [&](size_t index) {
                                     (*it)[param] = index;
                                     return validate(*it);
                                   });
      if (valid_it != shuffled_range.end()) {
        continue;
      }
      // All parameter values render the current element invalid. We have to
      // remove it, which means some index pairs will no longer be covered.
      indices_list.erase_after(prev);
      ++num_elements_erased;  // Reported as error below.
      it = prev;              // Element removed, don't increment prev below.
      break;
    }
    prev = it;
  }

  // All indices are fully specified now. Transform them to parameter tuples.
  auto make_params = [&](const ParamIndices& indices) {
    return std::make_tuple(std::get<Is>(param_values)[indices[Is]]...);
  };
  std::vector<std::tuple<ParamTypes...>> result;
  std::transform(
      indices_list.begin(), indices_list.end(), std::back_inserter(result),
      [&](const ParamIndices& indices) { return make_params(indices); });

  // Report error if some indices had to be removed.
  if (num_elements_erased) {
    LOG(ERROR) << num_elements_erased << " elements of "
               << num_elements_erased + result.size()
               << " erased, coverage is incomplete.";
  }

  return result;
}

template <typename F, typename... BoundArgs>
class CallWithTuple {
  template <typename... Args>
  using Result = typename std::result_of<F(Args...)>::type;

 public:
  CallWithTuple(const F& functor, std::tuple<BoundArgs...> bound)
      : functor_(functor), bound_(bound) {}

  template <typename... Args>
  Result<BoundArgs..., Args...> operator()(std::tuple<Args...> args) const {
    return Apply(std::tuple_cat(bound_, args),
                 absl::index_sequence_for<BoundArgs..., Args...>{});
  }

 private:
  template <typename... Args, size_t... Is>
  Result<Args...> Apply(std::tuple<Args...> args,
                        absl::index_sequence<Is...>) const {
    return functor_(std::get<Is>(args)...);
  }

  F functor_;
  std::tuple<BoundArgs...> bound_;
};
}  // namespace detail

// Given an orthogonal array of parameters and a parameter validator, creates a
// vector of valid parameters to perform all-pairs testing. The validator takes
// a std::tuple<std::optional<ParamTypes>...> and should return whether the
// combination of specified parameters is valid.
template <typename RandGen, typename V, typename... ParamTypes>
std::vector<std::tuple<ParamTypes...>> MakeAllPairs(
    RandGen&& rand_gen, const V& validator,
    const std::vector<ParamTypes>&... param_values) {
  return detail::MakeAllPairs(rand_gen, validator,
                              std::make_tuple(param_values...),
                              absl::index_sequence_for<ParamTypes...>{});
}

// Creates a callable taking a tuple parameter, applying f to the bound
// parameters and the tuple elements.
// Think 'std::bind(std::invoke, f, bound, _...)'.
template <class F, class... BoundArgs>
detail::CallWithTuple<F, BoundArgs...> MakeCallWithTuple(F f,
                                                         BoundArgs... bound) {
  return {f, std::make_tuple(bound...)};
}
}  // namespace nvidia_libs_test

#endif  // NVIDIA_LIBS_TEST_ALL_PAIRS_H_

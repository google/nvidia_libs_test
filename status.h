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

#ifndef NVIDIA_LIBS_TEST_ERRORS_H_
#define NVIDIA_LIBS_TEST_ERRORS_H_

#include <sstream>
#include <string>

#include "ostream_nullptr.h"
#include "glog/logging.h"
#include "absl/types/optional.h"

// Provides a Status object intended as function return type to denote if an
// operation was successful. StatusOr<T> can be used to return an object or
// failure status. The status can be checked with RETURN_IF_ERROR_STATUS or
// CHECK_OK_STATUS. If called with an error status, those macros will return
// that status from the current function or exit the program respectively.

namespace nvidia_libs_test {
using string = std::string;

// Status object representing that an operation was successful, or otherwise
// holding an error string.
class Status final {
 public:
  Status() : ok_(true), message_("ok") {}
  explicit Status(const string& message) : ok_(false), message_(message) {}

  bool ok() const { return ok_; }
  const string& message() const { return message_; }

  // Streams value to message.
  template <typename T>
  Status& operator<<(const T& value) {
    std::ostringstream oss;
    oss << value;
    message_.append(oss.str());
    return *this;
  }

 private:
  bool ok_;
  string message_;
};

// Compares two status objects for equality.
inline bool operator==(const Status& left, const Status& right) {
  return left.ok() == right.ok() && left.message() == right.message();
}

// Compares two status objects for inequality.
inline bool operator!=(const Status& left, const Status& right) {
  return !(left == right);
}

// Prints status message to stream.
inline std::ostream& operator<<(std::ostream& os, const Status& status) {
  return os << status.message();
}

// Creates a success status object.
inline Status OkStatus() { return Status(); }
// Creates an error status object.
inline Status ErrorStatus(const string& message) { return Status(message); }

// Contains an object of type T or an error status.
template <typename T>
class StatusOr {
 public:
  StatusOr(const Status& status) : status_(status) {
    CHECK_NE(OkStatus(), status_);
  }
  StatusOr(const T& value) : optional_(value) {}
  StatusOr(T&& value) noexcept : optional_(std::move(value)) {}

  const Status& status() const { return status_; }

  const T& ValueOrDie() const {
    CHECK_EQ(OkStatus(), status_);
    return optional_.value();
  }
  T& ValueOrDie() {
    CHECK_EQ(OkStatus(), status_);
    return optional_.value();
  }

 private:
  Status status_;
  absl::optional<T> optional_;
};

// If 'expr' returns an error status, returns from the current function with
// that status.
#define RETURN_IF_ERROR_STATUS(expr)                                     \
  do {                                                                   \
    auto _status = expr;                                                 \
    if (!_status.ok()) {                                                 \
      return _status << "\nin " << __FILE__ << "(" << __LINE__ << "): '" \
                     << #expr << "'";                                    \
    }                                                                    \
  } while (false)

#define CONCAT(left, right) CONCAT_IMPL(left, right)
#define CONCAT_IMPL(left, right) left##right

#define ASSIGN_OR_RETURN_STATUS(lhs, expr) \
  ASSIGN_OR_RETURN_STATUS_IMPL(CONCAT(_status_or_, __COUNTER__), lhs, expr)

#define ASSIGN_OR_RETURN_STATUS_IMPL(status_or, lhs, expr) \
  auto status_or = expr;                                   \
  RETURN_IF_ERROR_STATUS(status_or.status());              \
  lhs = std::move(status_or.ValueOrDie())

// Exits the program if 'expr' returns an error status.
#define CHECK_OK_STATUS(expr) CHECK_EQ(OkStatus(), expr)
}  // namespace nvidia_libs_test

#endif  // NVIDIA_LIBS_TEST_ERRORS_H_

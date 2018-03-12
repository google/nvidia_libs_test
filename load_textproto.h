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

#ifndef NVIDIA_LIBS_TEST_LOAD_TEXTPROTO_H_
#define NVIDIA_LIBS_TEST_LOAD_TEXTPROTO_H_

#include <string>
#include <fstream>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

namespace nvidia_libs_test {

template <typename T>
void LoadTextProto(string proto_path, T* proto) {
  std::ifstream ifs(proto_path);
  google::protobuf::io::IstreamInputStream iis(&ifs);
  CHECK_EQ(google::protobuf::TextFormat::Parse(&iis, proto), true);
}

}  // namespace nvidia_libs_test

#endif  // NVIDIA_LIBS_TEST_LOAD_TEXTPROTO_H_

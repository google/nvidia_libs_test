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

#ifndef NVIDIA_LIBS_TEST_OSTREAM_NULLPTR_H_
#define NVIDIA_LIBS_TEST_OSTREAM_NULLPTR_H_

namespace std {
// Workaround for http://cplusplus.github.io/LWG/lwg-active.html#2221
template <class CharT, class Traits>
basic_ostream<CharT, Traits>& operator<<(basic_ostream<CharT, Traits>& os,
                                         nullptr_t) {
  return os << (void*)nullptr;
}
}  // namespace std

#endif  // NVIDIA_LIBS_TEST_OSTREAM_NULLPTR_H_

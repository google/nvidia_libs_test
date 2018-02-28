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

#ifndef NVIDIA_LIBS_TEST_KERNEL_TIMER_H_
#define NVIDIA_LIBS_TEST_KERNEL_TIMER_H_

#include <memory>

// Timers measuring GPU kernel runtimes using CUPTI.

namespace nvidia_libs_test {

class KernelTimer {
 protected:
  KernelTimer() = default;

 public:
  // Creates a timer which always reports zero elapsed time.
  static std::unique_ptr<KernelTimer> CreateNopTimer();
  // Creates a timer which measures GPU kernel duration for the current context.
  // Only one instance can exist at a time.
  static std::unique_ptr<KernelTimer> CreateDurationTimer();
  // Creates a timer which measures GPU kernel cycles for the current context.
  static std::unique_ptr<KernelTimer> CreateCyclesTimer();

  virtual ~KernelTimer() = default;
  virtual void StartTiming() = 0;
  virtual void StopTiming() = 0;
  // Returns elapsed time in seconds.
  virtual double GetTime() = 0;
  // Resets the timer.
  virtual void ResetTime() = 0;
};
}  // namespace nvidia_libs_test
#endif  // NVIDIA_LIBS_TEST_KERNEL_TIMER_H_

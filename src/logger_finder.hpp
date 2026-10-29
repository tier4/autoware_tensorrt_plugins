// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LOGGER_FINDER_HPP_
#define LOGGER_FINDER_HPP_

#include <NvInferRuntime.h>

#include <mutex>

namespace autoware::tensorrt_plugins
{

// Holds the ILoggerFinder TensorRT hands to a plugin library via
// setLoggerFinder(). Each plugin shared library owns one instance of this.
class ThreadSafeLoggerFinder
{
public:
  ThreadSafeLoggerFinder() = default;

  // Set the logger finder.
  void setLoggerFinder(nvinfer1::ILoggerFinder * finder)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (logger_finder_ == nullptr && finder != nullptr) {
      logger_finder_ = finder;
    }
  }

  // Get the logger.
  nvinfer1::ILogger * getLogger() noexcept
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (logger_finder_ != nullptr) {
      return logger_finder_->findLogger();
    }
    return nullptr;
  }

private:
  nvinfer1::ILoggerFinder * logger_finder_{nullptr};
  std::mutex mutex_;
};

}  // namespace autoware::tensorrt_plugins

#endif  // LOGGER_FINDER_HPP_

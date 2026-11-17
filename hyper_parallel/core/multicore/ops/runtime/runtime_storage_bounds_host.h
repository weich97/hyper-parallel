/*
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_RUNTIME_STORAGE_BOUNDS_HOST_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_RUNTIME_STORAGE_BOUNDS_HOST_H_

#include <cstring>
#include "runtime_storage_bounds.h"

namespace MulticoreRuntime {

// Byte lengths come from UINT8 input storage shapes, never from the runtime header.
inline size_t AppendRuntimeStorageBounds(void *data, size_t capacity, size_t native_bytes,
                                        int64_t runtime_bytes, int64_t event_bytes) {
  constexpr int64_t min_runtime_bytes = 64;
  constexpr int64_t max_runtime_bytes = int64_t{1} << 32;
  constexpr int64_t min_event_bytes = 1024 * sizeof(int32_t);
  if (data == nullptr || native_bytes == 0 || native_bytes > capacity ||
      runtime_bytes < min_runtime_bytes || runtime_bytes >= max_runtime_bytes || event_bytes < min_event_bytes) {
    return 0;
  }
  size_t padding = (alignof(RuntimeStorageBounds) - native_bytes % alignof(RuntimeStorageBounds)) %
                   alignof(RuntimeStorageBounds);
  if (capacity - native_bytes < padding + sizeof(RuntimeStorageBounds)) {
    return 0;
  }
  size_t offset = RuntimeStorageBoundsOffset(native_bytes);
  RuntimeStorageBounds bounds{static_cast<uint64_t>(runtime_bytes), static_cast<uint64_t>(event_bytes)};
  std::memcpy(static_cast<uint8_t *>(data) + offset, &bounds, sizeof(bounds));
  return offset + sizeof(bounds);
}

}  // namespace MulticoreRuntime

#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_RUNTIME_STORAGE_BOUNDS_HOST_H_

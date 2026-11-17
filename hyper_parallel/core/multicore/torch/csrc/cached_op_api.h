/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_CACHED_OP_API_H_
#define HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_CACHED_OP_API_H_

#include <mutex>

#include "op_plugin/include/npu_cpp_extension.h"

namespace hyper_parallel::multicore {

/**
 * Resolve one ACLNN operator's execution and workspace symbols exactly once.
 *
 * Each instance belongs to one HYPER_EXEC_NPU_CMD_EXT call site and one task
 * queue branch. The caller-owned symbol slots and the resolved symbols must
 * remain valid for the process lifetime. Concurrent first use is serialized;
 * if GetApiFunc throws, a later call retries the resolution.
 */
class CachedOpApiResolver final {
 public:
  /**
   * Populate one call site's symbol slots on its first invocation.
   *
   * @param api_name Null-terminated ACLNN execution symbol name.
   * @param workspace_api_name Null-terminated workspace-query symbol name.
   * @param execute Process-lifetime slot receiving the execution symbol.
   * @param workspace Process-lifetime slot receiving the workspace symbol.
   */
  void operator()(const char *api_name, const char *workspace_api_name, void *&execute, void *&workspace) {
    std::call_once(resolve_once_, [&]() { ::GetApiFunc(api_name, workspace_api_name, execute, workspace); });
  }

 private:
  std::once_flag resolve_once_;
};

}  // namespace hyper_parallel::multicore

/**
 * Execute an ACLNN operator through the native torch_npu extension bridge.
 *
 * The execution and workspace symbols are cached once per macro call site and
 * independently for the V1 and V2 task queue paths. Argument conversion,
 * workspace ownership, stream submission, cache lookup, and completion
 * semantics remain those of EXEC_NPU_CMD_V1_EXT and EXEC_NPU_CMD_V2_EXT.
 *
 * @param aclnn_api Compile-time ACLNN execution symbol token.
 * @param ... Arguments forwarded unchanged to the native extension bridge.
 */
#define HYPER_EXEC_NPU_CMD_EXT(aclnn_api, ...)                                       \
  do {                                                                               \
    const auto hyper_task_queue_enable = OpApiGetTaskQueueEnable();                  \
    if (hyper_task_queue_enable == 2) {                                              \
      static ::hyper_parallel::multicore::CachedOpApiResolver hyper_v2_api_resolver; \
      auto &GetApiFunc = hyper_v2_api_resolver;                                      \
      EXEC_NPU_CMD_V2_EXT(aclnn_api, __VA_ARGS__);                                   \
    } else {                                                                         \
      static ::hyper_parallel::multicore::CachedOpApiResolver hyper_v1_api_resolver; \
      auto &GetApiFunc = hyper_v1_api_resolver;                                      \
      EXEC_NPU_CMD_V1_EXT(aclnn_api, __VA_ARGS__);                                   \
    }                                                                                \
  } while (false)

#endif  // HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_CACHED_OP_API_H_

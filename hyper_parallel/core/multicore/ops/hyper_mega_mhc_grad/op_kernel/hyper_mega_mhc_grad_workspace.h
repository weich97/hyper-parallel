/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef HYPER_MEGA_MHC_GRAD_WORKSPACE_H_
#define HYPER_MEGA_MHC_GRAD_WORKSPACE_H_

#include <cstdint>

namespace HyperMegaMhcGradWorkspace {
constexpr uint64_t FUSED_STREAMS = 4;
constexpr uint64_t FUSED_HIDDEN_SIZE = 5120;
}  // namespace HyperMegaMhcGradWorkspace

#endif  // HYPER_MEGA_MHC_GRAD_WORKSPACE_H_

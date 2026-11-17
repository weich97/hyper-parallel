/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*! \file hyper_mega_mhc_tiling.cpp */
#include "mhc_pre_sinkhorn_tiling.h"
#include "../op_kernel/hyper_mega_mhc_tiling_key.h"
#include "register/op_def_registry.h"
#include "../op_kernel/runtime/runtime_storage_bounds_host.h"

namespace optiling {
ge::graphStatus TilingForMhcPreSinkhorn(gert::TilingContext *context);
ge::graphStatus TilingPrepareForMhcPreSinkhorn(gert::TilingParseContext *context);

// Use the legacy declaration so CANN also generates the nested upstream device type.
REGISTER_TILING_DATA_CLASS(MhcPreSinkhornTilingDataOp, MhcPreSinkhornTilingData)
BEGIN_TILING_DATA_DEF(HyperMegaMhcLaunchTilingData)
TILING_DATA_FIELD_DEF_STRUCT(MhcPreSinkhornTilingData, native);
TILING_DATA_FIELD_DEF(uint64_t, runtimeConfigBytes);
TILING_DATA_FIELD_DEF(uint64_t, eventCounterBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HyperMegaMhc, HyperMegaMhcLaunchTilingData)

namespace {
constexpr uint64_t ALIGN_BYTES = 512;
constexpr uint64_t MHC_PRE_USER_WORKSPACE_BYTES = 128 * 1024 * 1024;
constexpr uint64_t MAX_RMS_HIDDEN_SIZE = 12288;

uint64_t AlignUp(uint64_t value) { return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES; }

ge::graphStatus TilingFunc(gert::TilingContext *context) {
  ge::graphStatus status = TilingForMhcPreSinkhorn(context);
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }
  const auto *residualShape = context->GetInputShape(0);
  if (residualShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &shape = residualShape->GetStorageShape();
  const size_t dimNum = shape.GetDimNum();
  if (dimNum != 3 && dimNum != 4) {
    return ge::GRAPH_FAILED;
  }
  const uint64_t tokenCount = dimNum == 3 ? shape.GetDim(0) : shape.GetDim(0) * shape.GetDim(1);
  const uint64_t hiddenSize = shape.GetDim(dimNum - 1);
  if (hiddenSize > MAX_RMS_HIDDEN_SIZE) {
    return ge::GRAPH_FAILED;
  }
  const uint64_t hcMult = shape.GetDim(dimNum - 2);
  const uint64_t hcMix = hcMult * hcMult + 2 * hcMult;
  const uint64_t sinkhornSteps = 40;
  uint64_t additional = 0;
  additional += AlignUp(tokenCount * hcMix * sizeof(float));
  additional += AlignUp(tokenCount * sizeof(float));
  additional += AlignUp(sinkhornSteps * tokenCount * hcMult * sizeof(float));
  additional += AlignUp(sinkhornSteps * tokenCount * hcMult * hcMult * sizeof(float));
  additional += AlignUp(tokenCount * hiddenSize * sizeof(uint16_t));
  additional += AlignUp(tokenCount * sizeof(float));
  // Native tiling reserves 16 MiB of system workspace plus 128 MiB of user
  // workspace. GetUserWorkspace() skips the system prefix, so only the latter
  // is available after the X-cast workspace for these intermediates.
  if (additional > MHC_PRE_USER_WORKSPACE_BYTES) {
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] += additional - MHC_PRE_USER_WORKSPACE_BYTES;
  }
  context->SetTilingKey(GET_TPL_TILING_KEY(HYPER_MEGA_MHC_M_SPLIT));
  const auto *runtimeShape = context->GetInputShape(9);
  const auto *eventShape = context->GetInputShape(10);
  const auto *runtimeDesc = context->GetInputDesc(9);
  const auto *eventDesc = context->GetInputDesc(10);
  auto raw = context->GetRawTilingData();
  if (runtimeShape == nullptr || eventShape == nullptr || runtimeDesc == nullptr || eventDesc == nullptr ||
      raw == nullptr || runtimeDesc->GetDataType() != ge::DT_UINT8 || eventDesc->GetDataType() != ge::DT_UINT8) {
    return ge::GRAPH_FAILED;
  }
  size_t bytes = MulticoreRuntime::AppendRuntimeStorageBounds(
    raw->GetData(), raw->GetCapacity(), raw->GetDataSize(),
    runtimeShape->GetStorageShape().GetShapeSize(), eventShape->GetStorageShape().GetShapeSize());
  if (bytes == 0) {
    return ge::GRAPH_FAILED;
  }
  raw->SetDataSize(bytes);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(HyperMegaMhc)
  .Tiling(TilingFunc)
  .TilingParse<MhcPreSinkhornCompileInfo>(TilingPrepareForMhcPreSinkhorn);
}  // namespace optiling

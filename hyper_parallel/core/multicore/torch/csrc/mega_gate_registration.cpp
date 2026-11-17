/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def(
    "mega_gate_route_out(Tensor logits, Tensor text_bias, Tensor vision_bias, Tensor image_mask, "
    "Tensor runtime_config, Tensor diagnostic_buffer, Tensor(a!) routing_weights, Tensor(b!) expert_indices, "
    "Tensor(c!) route_scores, Tensor(d!) selected_scores, Tensor(e!) normalization_denominator, "
    "int top_k, float routed_scaling_factor, bool use_vision_bias) "
    "-> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))");
  m.def(
    "mega_gate_route("
    "Tensor logits, "
    "Tensor text_bias, "
    "Tensor vision_bias, "
    "Tensor image_mask, "
    "Tensor runtime_config, "
    "Tensor diagnostic_buffer, "
    "int top_k, "
    "float routed_scaling_factor, "
    "bool use_vision_bias"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def(
    "mega_gate_route_grad("
    "Tensor logits, "
    "Tensor route_scores, "
    "Tensor selected_scores, "
    "Tensor normalization_denominator, "
    "Tensor expert_indices, "
    "Tensor grad_routing_weights, "
    "Tensor runtime_config, "
    "Tensor diagnostic_buffer, "
    "int top_k, "
    "float routed_scaling_factor"
    ") -> Tensor");
}

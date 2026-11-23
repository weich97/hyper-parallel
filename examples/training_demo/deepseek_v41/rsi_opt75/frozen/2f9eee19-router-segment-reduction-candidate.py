# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Replace contended small-segment index accumulation with dense FP32 reductions."""

from functools import wraps
import inspect

import torch

from hyper_parallel.models.deepseek_v41.adapter import aux_loss


def dense_segment_sum(values, sequence_ids, num_sequences):
    """Preserve arbitrary IDs and masks without a host read or a low-precision matmul."""
    if num_sequences > 8:
        return values.new_zeros(num_sequences, values.shape[-1]).index_add(0, sequence_ids, values)
    return torch.stack([
        (values * (sequence_ids == index).unsqueeze(-1)).sum(dim=0)
        for index in range(num_sequences)
    ])


def make_candidate():
    """Keep normalization, expert counts and distributed loss equations unchanged."""
    original = aux_loss.sequence_load_balancing_loss
    source = inspect.getsource(original)
    before = ('        probability_sum = scores.new_zeros(num_sequences, num_experts).index_add(\n'
              '            0, sequence_ids, weighted_probabilities,\n'
              '        )')
    after = '        probability_sum = dense_segment_sum(weighted_probabilities, sequence_ids, num_sequences)'
    if source.count(before) != 1:
        raise ValueError('Unexpected sequence auxiliary-loss implementation')
    namespace = dict(original.__globals__, dense_segment_sum=dense_segment_sum)
    exec(compile(source.replace(before, after), __file__, 'exec'), namespace)
    return namespace['sequence_load_balancing_loss']


def install():
    """Replace both the adapter entry and its model-owned imported binding."""
    # The model is imported only when this experimental replacement is requested.
    from hyper_parallel.models.deepseek_v41 import modeling_deepseek_v41

    candidate = make_candidate()
    counts = dict(dense_calls=0, fallback_calls=0)

    @wraps(candidate)
    def observed(scores, selected_experts, sequence_ids, num_sequences, *args, **kwargs):
        counts['dense_calls' if 1 < num_sequences <= 8 else 'fallback_calls'] += 1
        return candidate(scores, selected_experts, sequence_ids, num_sequences, *args, **kwargs)

    aux_loss.sequence_load_balancing_loss = observed
    modeling_deepseek_v41.sequence_load_balancing_loss = observed
    return counts

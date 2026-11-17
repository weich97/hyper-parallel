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
"""Unit tests for hyper_parallel.core.dtensor.init_weights."""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.dtensor import init_weights as init_weights_module

_INIT_ON_DEVICE_TARGET = "hyper_parallel.core.dtensor.init_weights._init_on_device"


@contextmanager
def _fake_init_on_device(device, include_buffers=False):
    """Stand-in for the Torch ``init_on_device`` context manager."""
    del device, include_buffers
    yield


class TestInitEmptyWeights(unittest.TestCase):
    """Tests for InitEmptyWeights."""

    def test_deterministic_meta_construction_does_not_fill_discarded_storage(self):
        """Feature: Empty parameter initialization under deterministic algorithms.

        Description: Inspect real factory calls while constructing a meta Linear.
        Expectation: Skip filling discarded CPU storage and preserve deterministic algorithms.
        """
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        original_empty = torch.empty
        observed = []

        def _observed_empty(*args, **kwargs):
            observed.append(torch.utils.deterministic.fill_uninitialized_memory)
            return original_empty(*args, **kwargs)

        try:
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            with patch.object(torch, "empty", side_effect=_observed_empty):
                with init_weights_module.init_empty_weights():
                    module = torch.nn.Linear(32, 32)
            self.assertTrue(module.weight.is_meta, "The parameter must retain no physical storage")
            self.assertTrue(observed, "Real tensor factories must execute during construction")
            self.assertFalse(any(observed), "Discarded CPU allocations must not be filled")
            self.assertTrue(torch.are_deterministic_algorithms_enabled(), "Keep deterministic algorithms enabled")
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory, "Restore the caller's fill policy")
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def test_nested_empty_context_restores_fill_policy_after_exception(self):
        """Feature: Scoped deterministic initialization state.

        Description: Raise inside nested empty initialization for both caller fill policies.
        Expectation: Restore the original fill policy even on exceptional exit.
        """
        original_fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            for fill in (True, False):
                with self.subTest(fill=fill):
                    torch.utils.deterministic.fill_uninitialized_memory = fill
                    with self.assertRaisesRegex(RuntimeError, "construction failed"):
                        with init_weights_module.init_empty_weights():
                            with init_weights_module.init_empty_weights():
                                raise RuntimeError("construction failed")
                    self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, fill,
                                     "Nested or exceptional construction must preserve caller state")
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = original_fill

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default_include_buffers_false(self):
        """Test default include buffers false."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights():
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights(include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=True
        )


class TestInitOnDevice(unittest.TestCase):
    """Tests for InitOnDevice."""

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default(self):
        """Test default."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "npu:0"
        with init_on_device(my_device):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "cpu"
        with init_on_device(my_device, include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=True
        )


if __name__ == "__main__":
    unittest.main()

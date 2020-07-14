# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from parameterized import parameterized
from scipy.ndimage import zoom as zoom_scipy
from tests.utils import NumpyImageTestCase2D

from monai.transforms import Zoom

VALID_CASES = [(1.5, "nearest"), (1.5, "nearest"), (0.8, "bilinear"), (0.8, "area")]

INVALID_CASES = [((None, None), "bilinear", TypeError), ((0.9, 0.9), "s", NotImplementedError)]


class TestZoom(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, zoom, mode):
        zoom_fn = Zoom(zoom=zoom, mode=mode, keep_size=False)
        zoomed = zoom_fn(self.imt[0])
        _order = 0
        if mode.endswith("linear"):
            _order = 1
        expected = list()
        for channel in self.imt[0]:
            expected.append(zoom_scipy(channel, zoom=zoom, mode="nearest", order=_order, prefilter=False))
        expected = np.stack(expected).astype(np.float32)
        np.testing.assert_allclose(zoomed, expected, atol=1.0)

    def test_keep_size(self):
        zoom_fn = Zoom(zoom=[0.6, 0.6], keep_size=True, align_corners=True)
        zoomed = zoom_fn(self.imt[0], mode="bilinear")
        np.testing.assert_allclose(zoomed.shape, self.imt.shape[1:])

        zoom_fn = Zoom(zoom=[1.3, 1.3], keep_size=True)
        zoomed = zoom_fn(self.imt[0])
        np.testing.assert_allclose(zoomed.shape, self.imt.shape[1:])

    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, zoom, mode, raises):
        with self.assertRaises(raises):
            zoom_fn = Zoom(zoom=zoom, mode=mode)
            zoom_fn(self.imt[0])


if __name__ == "__main__":
    unittest.main()

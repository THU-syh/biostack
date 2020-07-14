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

from monai.transforms import RandZoomd

VALID_CASES = [(0.8, 1.2, "nearest", None, False)]


class TestRandZoomd(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, min_zoom, max_zoom, mode, align_corners, keep_size):
        key = "img"
        random_zoom = RandZoomd(
            key,
            prob=1.0,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            mode=mode,
            align_corners=align_corners,
            keep_size=keep_size,
        )
        random_zoom.set_random_state(1234)

        zoomed = random_zoom({key: self.imt[0]})
        expected = list()
        for channel in self.imt[0]:
            expected.append(zoom_scipy(channel, zoom=random_zoom._zoom, mode="nearest", order=0, prefilter=False))
        expected = np.stack(expected).astype(np.float32)
        np.testing.assert_allclose(expected, zoomed[key], atol=1.0)

    def test_keep_size(self):
        key = "img"
        random_zoom = RandZoomd(key, prob=1.0, min_zoom=0.6, max_zoom=0.7, keep_size=True)
        zoomed = random_zoom({key: self.imt[0]})
        self.assertTrue(np.array_equal(zoomed[key].shape, self.imt.shape[1:]))

    @parameterized.expand(
        [("no_min_zoom", None, 1.1, "bilinear", TypeError), ("invalid_order", 0.9, 1.1, "s", NotImplementedError)]
    )
    def test_invalid_inputs(self, _, min_zoom, max_zoom, mode, raises):
        key = "img"
        with self.assertRaises(raises):
            random_zoom = RandZoomd(key, prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, mode=mode)
            random_zoom({key: self.imt[0]})


if __name__ == "__main__":
    unittest.main()

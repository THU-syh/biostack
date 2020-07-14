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
import scipy.ndimage
from parameterized import parameterized
from tests.utils import NumpyImageTestCase2D, NumpyImageTestCase3D

from monai.transforms import Rotated

TEST_CASES_2D = [
    (-30, False, "bilinear", "border", False),
    (-45, True, "bilinear", "border", False),
    (40, True, "nearest", "reflection", False),
    (-180, False, "nearest", "zeros", False),
    (90, False, "bilinear", "zeros", True),
]

TEST_CASES_3D = [
    (-30, False, "bilinear", "border", False),
    (-45, True, "bilinear", "border", False),
    (40, True, "nearest", "reflection", False),
    (-180, False, "nearest", "zeros", False),
    (90, False, "bilinear", "zeros", True),
]


class TestRotated2D(NumpyImageTestCase2D):
    @parameterized.expand(TEST_CASES_2D)
    def test_correct_results(self, angle, keep_size, mode, padding_mode, align_corners):
        rotate_fn = Rotated(("img", "seg"), angle, keep_size, (mode, "nearest"), padding_mode, align_corners)
        rotated = rotate_fn({"img": self.imt[0], "seg": self.segn[0]})
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated["img"].shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -angle, (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        np.testing.assert_allclose(expected, rotated["img"][0], atol=1e-3)

        expected = scipy.ndimage.rotate(
            self.segn[0, 0], -angle, (0, 1), not keep_size, order=0, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(int)
        self.assertLessEqual(np.count_nonzero(expected != rotated["seg"][0]), 20)


class TestRotated3D(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, angle, keep_size, mode, padding_mode, align_corners):
        rotate_fn = Rotated(("img", "seg"), [0, angle, 0], keep_size, (mode, "nearest"), padding_mode, align_corners)
        rotated = rotate_fn({"img": self.imt[0], "seg": self.segn[0]})
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated["img"].shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], angle, (0, 2), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        np.testing.assert_allclose(expected, rotated["img"][0], atol=1e-3)

        expected = scipy.ndimage.rotate(
            self.segn[0, 0], angle, (0, 2), not keep_size, order=0, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(int)
        self.assertLessEqual(np.count_nonzero(expected != rotated["seg"][0]), 100)


class TestRotated3DXY(NumpyImageTestCase3D):
    @parameterized.expand(TEST_CASES_3D)
    def test_correct_results(self, angle, keep_size, mode, padding_mode, align_corners):
        rotate_fn = Rotated(("img", "seg"), [0, 0, angle], keep_size, (mode, "nearest"), padding_mode, align_corners)
        rotated = rotate_fn({"img": self.imt[0], "seg": self.segn[0]})
        if keep_size:
            np.testing.assert_allclose(self.imt[0].shape, rotated["img"].shape)
        _order = 0 if mode == "nearest" else 1
        if padding_mode == "border":
            _mode = "nearest"
        elif padding_mode == "reflection":
            _mode = "reflect"
        else:
            _mode = "constant"
        expected = scipy.ndimage.rotate(
            self.imt[0, 0], -angle, (0, 1), not keep_size, order=_order, mode=_mode, prefilter=False
        )
        np.testing.assert_allclose(expected, rotated["img"][0], atol=1e-3)

        expected = scipy.ndimage.rotate(
            self.segn[0, 0], -angle, (0, 1), not keep_size, order=0, mode=_mode, prefilter=False
        )
        expected = np.stack(expected).astype(int)
        self.assertLessEqual(np.count_nonzero(expected != rotated["seg"][0]), 100)


if __name__ == "__main__":
    unittest.main()

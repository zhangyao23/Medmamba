import json
import os
import sys
import tempfile
import types
import unittest
import importlib.util

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_module(name: str, relative_path: str):
    module_path = os.path.join(REPO_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if "nibabel" not in sys.modules:
    sys.modules["nibabel"] = types.ModuleType("nibabel")


volumetric_dataset = _load_module("volumetric_dataset_preprocessing_test", "src/data/volumetric_dataset.py")
transforms_module = _load_module("transforms_preprocessing_test", "src/data/transforms.py")

VolumetricMILDataset = volumetric_dataset.VolumetricMILDataset
_normalize_volume = volumetric_dataset._normalize_volume
get_train_transforms = transforms_module.get_train_transforms
get_val_transforms = transforms_module.get_val_transforms


class _DummyImage:
    def __init__(self, array: np.ndarray):
        self.array = array

    def get_fdata(self):
        return self.array


class PreprocessingTests(unittest.TestCase):
    def test_fixed_volume_normalization_clips_to_unit_interval(self):
        volume = np.array([-2000.0, -1024.0, 0.0, 3071.0, 5000.0], dtype=np.float32)
        normalized = _normalize_volume(
            volume,
            volume_path="dummy_ct.nii.gz",
            min_hu=-1024.0,
            max_hu=3071.0,
            adaptive_norm=False,
        )

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized.min()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.max()), 1.0, places=6)

    def test_dataset_applies_transform_after_patch_tensor_creation(self):
        fake_volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        dummy_nib = volumetric_dataset.nib
        original_load = getattr(dummy_nib, "load", None)
        original_canonical = getattr(dummy_nib, "as_closest_canonical", None)

        dummy_nib.load = lambda _: _DummyImage(fake_volume)
        dummy_nib.as_closest_canonical = lambda img: img

        transform = lambda x: x + 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "entries.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([{"image": "dummy_case.nii.gz", "label": 1}], f)

            dataset = VolumetricMILDataset(
                json_path=json_path,
                patch_size=(2, 2, 2),
                stride=(2, 2, 2),
                transform=transform,
                min_hu=0.0,
                max_hu=7.0,
                adaptive_norm=False,
            )
            item = dataset[0]

        if original_load is not None:
            dummy_nib.load = original_load
        if original_canonical is not None:
            dummy_nib.as_closest_canonical = original_canonical

        patches = item["patches"]
        self.assertEqual(tuple(patches.shape), (1, 1, 2, 2, 2))
        self.assertAlmostEqual(float(patches.min().item()), 0.5, places=6)
        self.assertAlmostEqual(float(patches.max().item()), 1.5, places=6)

    def test_train_val_transforms_preserve_tensor_shape(self):
        tensor = torch.linspace(0.0, 1.0, steps=2 * 1 * 4 * 4 * 4).view(2, 1, 4, 4, 4)

        train_transform = get_train_transforms(
            standardize=True,
            intensity_jitter=True,
            gaussian_noise=True,
        )
        val_transform = get_val_transforms(standardize=True)

        train_out = train_transform(tensor.clone())
        val_out = val_transform(tensor.clone())

        self.assertEqual(train_out.shape, tensor.shape)
        self.assertEqual(val_out.shape, tensor.shape)
        self.assertTrue(torch.isfinite(train_out).all())
        self.assertTrue(torch.isfinite(val_out).all())


if __name__ == "__main__":
    unittest.main()

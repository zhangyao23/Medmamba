import os
import sys
import types
import unittest
import importlib.util

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

if "monai" not in sys.modules:
    sys.modules["monai"] = types.ModuleType("monai")

if "monai.transforms" not in sys.modules:
    monai_transforms = types.ModuleType("monai.transforms")

    class _Compose:
        def __init__(self, *args, **kwargs):
            self.args = args

        def __call__(self, value):
            return value

    monai_transforms.Compose = _Compose
    sys.modules["monai.transforms"] = monai_transforms


volumetric_dataset = _load_module("volumetric_dataset_test", "src/data/volumetric_dataset.py")
self_correction_module = _load_module("self_correction_test", "src/models/self_correction.py")
spatial_scanner_module = _load_module("spatial_scanner_test", "src/models/spatial_scanner_3d.py")
vector_quantizer_module = _load_module("vector_quantizer_test", "src/models/vector_quantizer_3d.py")
losses_module = _load_module("losses_test", "src/training/losses.py")
metrics_module = _load_module("metrics_test", "src/utils/metrics.py")

compute_axis_starts = volumetric_dataset.compute_axis_starts
SelfCorrectionModule = self_correction_module.SelfCorrectionModule
reorder_sequence = spatial_scanner_module.reorder_sequence
restore_sequence_order = spatial_scanner_module.restore_sequence_order
PartitionedVectorQuantizer = vector_quantizer_module.PartitionedVectorQuantizer
CombinedLoss = losses_module.CombinedLoss
evaluate_bag_level = metrics_module.evaluate_bag_level


class RegressionTests(unittest.TestCase):
    def test_reorder_sequence_roundtrip(self):
        seq = torch.tensor(
            [[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]
        )
        perm = torch.tensor([[2, 0, 1]])

        sorted_seq = reorder_sequence(seq, perm)
        restored = restore_sequence_order(sorted_seq, perm)

        self.assertTrue(torch.equal(sorted_seq[0, 0], seq[0, 2]))
        self.assertTrue(torch.equal(restored, seq))

    def test_compute_axis_starts_covers_tail(self):
        self.assertEqual(compute_axis_starts(3, 4, 2), [0])
        self.assertEqual(compute_axis_starts(10, 4, 4), [0, 4, 6])
        starts = compute_axis_starts(17, 6, 5)
        self.assertEqual(starts[-1], 11)
        self.assertGreaterEqual(starts[-1] + 6, 17)

    def test_vector_quantizer_ignores_padded_tokens(self):
        model = PartitionedVectorQuantizer(
            num_embeddings=4,
            embedding_dim=2,
            healthy_ratio=0.5,
            commitment_cost=0.25,
            use_ema=False,
        )
        model.train()
        with torch.no_grad():
            model.embedding.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
                )
            )

        z = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
        labels = torch.tensor([0])
        mask = torch.tensor([[1, 0, 0]], dtype=torch.long)

        quantized, codes, vq_loss = model(z, labels=labels, mask=mask)

        self.assertEqual(float(model.normal_patch_count.sum().item()), 1.0)
        self.assertEqual(float(model.cancer_patch_count.sum().item()), 0.0)
        self.assertTrue(torch.equal(codes[0, 1:], torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.equal(quantized[0, 1:], torch.zeros(2, 2)))
        self.assertGreaterEqual(vq_loss.item(), 0.0)

    def test_self_correction_returns_normalized_attention(self):
        module = SelfCorrectionModule(
            entropy_threshold=0.1,
            neighbor_radius=2,
            spatial_weight=1.0,
        )
        codes = torch.tensor([[1, 2, 2, 3]])
        coords = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 2], [5, 5, 5]]])
        attention_logits = torch.tensor([[0.0, 5.0, 5.0, -5.0]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

        corrected_codes, corrected_attention, correction_rate = module(
            codes=codes,
            coords=coords,
            attention_logits=attention_logits,
            mask=mask,
        )

        self.assertEqual(int(corrected_codes[0, 0].item()), 2)
        self.assertAlmostEqual(float(corrected_attention[0, :3].sum().item()), 1.0, places=6)
        self.assertEqual(float(corrected_attention[0, 3].item()), 0.0)
        self.assertGreater(float(correction_rate.item()), 0.0)

    def test_single_class_bag_metrics_do_not_crash(self):
        predictions = torch.tensor([[4.0, -2.0], [3.0, -1.0]])
        labels = torch.tensor([0, 0])

        metrics = evaluate_bag_level(predictions, labels)

        self.assertTrue(metrics["single_class_eval"])
        self.assertEqual(metrics["auc"], 0.0)
        self.assertEqual(metrics["pr_auc"], 0.0)

    def test_combined_loss_uses_distill_feat(self):
        loss_fn = CombinedLoss(
            lambda_cls=0.0,
            lambda_nu=0.0,
            lambda_vq=0.0,
            lambda_temporal=0.0,
            lambda_entropy=0.0,
            lambda_feat_distill=1.0,
        )
        logits = torch.zeros((1, 2))
        attention = torch.tensor([[1.0, 0.0]])
        labels = torch.tensor([0])
        vq_loss = torch.tensor(0.0)
        z_e = torch.zeros((1, 2, 2))
        distill_feat = torch.ones((1, 2, 2))
        z_e_frozen = torch.zeros((1, 2, 2))
        mask = torch.tensor([[1, 0]], dtype=torch.long)

        loss, loss_dict = loss_fn(
            logits=logits,
            attention=attention,
            labels=labels,
            vq_loss=vq_loss,
            z_e=z_e,
            distill_feat=distill_feat,
            z_e_frozen=z_e_frozen,
            mask=mask,
            mode="mil_train_phase2",
        )

        self.assertAlmostEqual(float(loss.item()), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["feat_distill"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

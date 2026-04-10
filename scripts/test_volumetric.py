import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging
from src.data.volumetric_loader import create_volumetric_dataloader
from src.models.feature_extractor_3d import VolumetricFeatureExtractor
from src.models.vector_quantizer_3d import PartitionedVectorQuantizer
from src.models.spatial_scanner_3d import ZOrderSpatialScanner
from src.models.video_mamba import VideoMamba3D
from src.models.mil_head import AttentionMILHead
from src.training import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    logger.info("="*60)
    logger.info("Test 1: Data Loading")
    logger.info("="*60)
    
    config = Config.from_yaml('configs/volumetric_config.yaml')
    
    try:
        dataloader = create_volumetric_dataloader(
            json_path=config.data['train_json'],
            batch_size=1,
            patch_size=tuple(config.data['patch_size']),
            stride=tuple(config.data['stride']),
            num_workers=0,
            shuffle=False,
            balanced_sampling=False
        )
        
        logger.info(f"Dataloader created successfully")
        logger.info(f"Number of samples: {len(dataloader.dataset)}")
        
        batch = next(iter(dataloader))
        logger.info(f"\nBatch contents:")
        logger.info(f"  patches shape: {batch['patches'].shape}")
        logger.info(f"  coords shape: {batch['coords'].shape}")
        logger.info(f"  labels shape: {batch['labels'].shape}")
        logger.info(f"  masks shape: {batch['masks'].shape}")
        logger.info(f"  num_patches: {batch['num_patches']}")
        
        logger.info(f"\nFirst sample stats:")
        logger.info(f"  Number of patches: {batch['num_patches'][0].item()}")
        logger.info(f"  Volume shape: {batch['volume_shapes'][0].tolist()}")
        logger.info(f"  Label: {batch['labels'][0].item()}")
        
        logger.info("\n✓ Data loading test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    logger.info("\n" + "="*60)
    logger.info("Test 2: Model Forward Pass")
    logger.info("="*60)
    
    config = Config.from_yaml('configs/volumetric_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        logger.info("\nCreating model components...")
        
        feature_extractor = VolumetricFeatureExtractor(
            arch='resnet18',
            spatial_dims=3,
            n_input_channels=1,
            pretrained=False,
            frozen=False
        ).to(device)
        
        codebook = PartitionedVectorQuantizer(
            num_embeddings=100,
            embedding_dim=512,
            healthy_ratio=0.8,
            commitment_cost=0.25,
            use_ema=True
        ).to(device)
        
        scanner = ZOrderSpatialScanner().to(device)
        
        mamba = VideoMamba3D(
            d_model=512,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=4,
            bidirectional=True
        ).to(device)
        
        mil_head = AttentionMILHead(
            input_dim=512,
            hidden_dim=256,
            num_classes=2
        ).to(device)
        
        logger.info("✓ All model components created")
        
        logger.info("\nLoading a batch...")
        dataloader = create_volumetric_dataloader(
            json_path=config.data['train_json'],
            batch_size=1,
            patch_size=tuple(config.data['patch_size']),
            stride=(32, 32, 32),
            num_workers=0,
            shuffle=False,
            balanced_sampling=False
        )
        
        batch = next(iter(dataloader))
        
        n_patches = batch['num_patches'][0].item()
        max_patches = min(n_patches, 64)
        
        patches = batch['patches'][:, :max_patches].to(device)
        coords = batch['coords'][:, :max_patches].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'][:, :max_patches].to(device)
        
        logger.info(f"Batch loaded: {patches.shape} (limited to {max_patches}/{n_patches} patches)")
        
        logger.info("\nForward pass through components...")
        
        logger.info("  1. Feature extraction (mini-batch=32)...")
        features = feature_extractor(patches, mini_batch_size=32)
        logger.info(f"     Features: {features.shape}")
        
        logger.info("  2. Vector quantization...")
        quantized, codes, vq_loss = codebook(features, labels)
        logger.info(f"     Quantized: {quantized.shape}")
        logger.info(f"     Codes: {codes.shape}")
        logger.info(f"     VQ Loss: {vq_loss.item():.6f}")
        
        logger.info("  3. Spatial scanning...")
        sorted_features, sorted_coords, _ = scanner(quantized, coords)
        logger.info(f"     Sorted features: {sorted_features.shape}")
        
        logger.info("  4. Mamba...")
        context = mamba(sorted_features, mask=masks)
        logger.info(f"     Context: {context.shape}")
        
        logger.info("  5. MIL head...")
        logits, attention = mil_head(context, mask=masks)
        logger.info(f"     Logits: {logits.shape}")
        logger.info(f"     Attention: {attention.shape}")
        
        logger.info(f"\nPredictions:")
        probs = torch.softmax(logits, dim=1)
        for i in range(len(labels)):
            pred_class = probs[i].argmax().item()
            pred_prob = probs[i, pred_class].item()
            true_class = labels[i].item()
            logger.info(f"  Sample {i}: pred={pred_class} (prob={pred_prob:.4f}), true={true_class}")
        
        logger.info(f"\nCodebook statistics:")
        stats = codebook.get_code_statistics()
        logger.info(f"  Healthy codes used: {stats['healthy_codes_used']}/80")
        logger.info(f"  Cancer codes used: {stats['cancer_codes_used']}/20")
        
        logger.info(f"\nSegmentation mask preview:")
        seg_mask = codebook.codes_to_segmentation_mask(codes)
        for i in range(len(labels)):
            cancer_patches = seg_mask[i].sum().item()
            total_patches = masks[i].sum().item()
            logger.info(f"  Sample {i}: {cancer_patches}/{total_patches} patches marked as cancer")
        
        logger.info("\n✓ Model forward pass test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Model forward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "="*60)
    logger.info("3D Volumetric Model Test Suite")
    logger.info("="*60)
    
    results = []
    
    results.append(("Data Loading", test_data_loading()))
    
    results.append(("Model Forward Pass", test_model_forward()))
    
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.error("\n❌ Some tests failed")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

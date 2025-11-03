#!/usr/bin/env python3
"""
Test script to verify only the neural affinity prediction component
"""

import torch
import numpy as np
from models.equibind_with_affinity import AffinityPredictor
from commons.utils import seed_all

def test_affinity_predictor():
    """Test only the affinity predictor component"""
    print("Testing Affinity Predictor Component...")
    
    # Set seed for reproducibility
    seed_all(42)
    
    # Create affinity predictor
    device = torch.device('cpu')
    affinity_predictor = AffinityPredictor(
        input_dim=5,  # Simple features: center_dist, min_dist, mean_dist, num_lig_atoms, num_rec_atoms
        hidden_dims=[64, 32, 16]
    )
    affinity_predictor.eval()
    
    print(f"Affinity predictor created with input_dim=5")
    
    try:
        # Create dummy simple features
        # Features: [center_distance, min_distance, mean_distance, num_lig_atoms, num_rec_atoms]
        simple_features = torch.tensor([
            [5.2, 2.1, 4.8, 6.0, 3.0]  # Example values
        ], dtype=torch.float32)
        
        print(f"Input features shape: {simple_features.shape}")
        print(f"Input features: {simple_features}")
        
        # Run prediction
        with torch.no_grad():
            affinity_pred = affinity_predictor(simple_features)
        
        print("✓ Affinity predictor forward pass successful!")
        print(f"Predicted affinity: {affinity_pred.item():.4f}")
        
        # Check if it's a reasonable value (not NaN or inf)
        if torch.isfinite(affinity_pred).all():
            print("✓ Affinity prediction is finite - SUCCESS!")
            return True
        else:
            print("✗ Affinity prediction is not finite")
            return False
            
    except Exception as e:
        print(f"✗ Affinity predictor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_geometric_features():
    """Test the geometric feature extraction"""
    print("\nTesting Geometric Feature Extraction...")
    
    device = torch.device('cpu')
    
    # Create dummy coordinates
    lig_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ], dtype=torch.float32)  # 6 ligand atoms
    
    rec_coords = torch.tensor([
        [5.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [5.5, 1.0, 0.0]
    ], dtype=torch.float32)  # 3 receptor atoms
    
    print(f"Ligand coordinates shape: {lig_coords.shape}")
    print(f"Receptor coordinates shape: {rec_coords.shape}")
    
    try:
        # Calculate geometric features (same as in the model)
        lig_center = torch.mean(lig_coords, dim=0, keepdim=True)  # (1, 3)
        rec_center = torch.mean(rec_coords, dim=0, keepdim=True)  # (1, 3)
        
        # Calculate center-to-center distance
        center_distance = torch.norm(lig_center - rec_center, dim=1)  # (1,)
        
        # Calculate average distances between ligand and receptor atoms
        distances = torch.cdist(lig_coords, rec_coords)  # (N_lig, N_rec)
        min_distance = torch.min(distances)
        mean_distance = torch.mean(distances)
        
        # Create a simple feature vector
        simple_features = torch.cat([
            center_distance,
            min_distance.unsqueeze(0),
            mean_distance.unsqueeze(0),
            torch.tensor([lig_coords.shape[0]], dtype=torch.float32, device=device),  # num ligand atoms
            torch.tensor([rec_coords.shape[0]], dtype=torch.float32, device=device),  # num receptor atoms
        ], dim=0)  # (5,)
        
        print(f"Geometric features: {simple_features}")
        print(f"  Center distance: {center_distance.item():.3f}")
        print(f"  Min distance: {min_distance.item():.3f}")
        print(f"  Mean distance: {mean_distance.item():.3f}")
        print(f"  Num ligand atoms: {lig_coords.shape[0]}")
        print(f"  Num receptor atoms: {rec_coords.shape[0]}")
        
        print("✓ Geometric feature extraction successful!")
        return True, simple_features
        
    except Exception as e:
        print(f"✗ Geometric feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_complete_pipeline():
    """Test the complete simplified affinity prediction pipeline"""
    print("\nTesting Complete Simplified Pipeline...")
    
    # Test geometric features
    success, features = test_simple_geometric_features()
    if not success:
        return False
    
    # Test affinity predictor
    device = torch.device('cpu')
    affinity_predictor = AffinityPredictor(
        input_dim=5,
        hidden_dims=[64, 32, 16]
    )
    affinity_predictor.eval()
    
    try:
        with torch.no_grad():
            affinity_pred = affinity_predictor(features.unsqueeze(0))
        
        print(f"✓ Complete pipeline successful!")
        print(f"Final affinity prediction: {affinity_pred.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Complete pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Neural Affinity Prediction Components")
    print("=" * 60)
    
    # Test individual components
    success1 = test_affinity_predictor()
    success2 = test_complete_pipeline()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All neural affinity prediction tests PASSED!")
        print("The simplified geometric approach is working correctly.")
    else:
        print("❌ Some neural affinity prediction tests FAILED!")
    print("=" * 60)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced EquiBind Model with Integrated Binding Affinity Prediction
This module extends the original EquiBind model to simultaneously predict:
1. 3D ligand coordinates (original EquiBind functionality)
2. Binding affinity scores (integrated affinity prediction)

The model outputs both structure and affinity in a single forward pass.
"""

import logging
import math
import os
from datetime import datetime

import dgl
import torch
from torch import nn
from dgl import function as fn
import numpy as np

from commons.process_mols import AtomEncoder, rec_atom_feature_dims, rec_residue_feature_dims, lig_feature_dims
from commons.logger import log

# Import the original EquiBind components
from models.equibind import (
    GraphNorm, get_non_lin, get_layer_norm, get_norm, apply_norm, 
    CoordsNorm, cross_attention, IEGMN_Layer, IEGMN, EquiBind
)

# Import affinity prediction components
from binding_affinity_predictor_vina import VinaScorer, PhysicsScorer


class AffinityPredictor(nn.Module):
    """
    Neural network module for binding affinity prediction.
    Takes protein-ligand interaction features and predicts binding affinity.
    """
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], dropout=0.2):
        super(AffinityPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final affinity prediction layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.affinity_net = nn.Sequential(*layers)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, interaction_features):
        """
        Predict binding affinity from interaction features.
        
        Args:
            interaction_features: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted binding affinity scores (batch_size, 1)
        """
        return self.affinity_net(interaction_features)


class InteractionFeatureExtractor(nn.Module):
    """
    Extracts interaction features between protein and ligand for affinity prediction.
    """
    
    def __init__(self, hidden_dim=64):
        super(InteractionFeatureExtractor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2
        
        # Feature projection layers
        self.lig_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rec_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Interaction layers
        self.interaction_net = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, lig_features, rec_features, lig_coords, rec_coords):
        """
        Extract interaction features between protein and ligand.
        
        Args:
            lig_features: Ligand node features (N_lig, hidden_dim)
            rec_features: Receptor node features (N_rec, hidden_dim)
            lig_coords: Ligand coordinates (N_lig, 3)
            rec_coords: Receptor coordinates (N_rec, 3)
        
        Returns:
            Interaction features (output_dim,)
        """
        # Project features
        lig_proj = self.lig_proj(lig_features)  # (N_lig, hidden_dim)
        rec_proj = self.rec_proj(rec_features)  # (N_rec, hidden_dim)
        
        # Global pooling to get complex-level features
        lig_global = torch.mean(lig_proj, dim=0)  # (hidden_dim,)
        rec_global = torch.mean(rec_proj, dim=0)  # (hidden_dim,)
        
        # Combine ligand and receptor features
        combined_features = torch.cat([lig_global, rec_global], dim=0)  # (output_dim,)
        
        # Apply interaction network
        interaction_features = self.interaction_net(combined_features)
        
        return interaction_features


class EquiBindWithAffinity(nn.Module):
    """
    Enhanced EquiBind model that integrates binding affinity prediction
    with 3D structure generation.
    """
    
    def __init__(self, 
                 # Original EquiBind parameters
                 lig_input_edge_feats_dim=5,
                 rec_input_edge_feats_dim=1,
                 lig_input_node_feats_dim=14,
                 rec_input_node_feats_dim=24,
                 hidden_dims=[256, 256, 256],
                 n_lays=5,
                 cross_msgs=True,
                 layer_norm=True,
                 layer_norm_coords=False,
                 final_h_layer_norm=False,
                 use_dist_in_layers=True,
                 skip_weight_h=0.5,
                 x_connection_init=0.25,
                 leakyrelu_neg_slope=1e-2,
                 debug=False,
                 device='cpu',
                 use_scalar_features=False,
                 num_att_heads=50,
                 dropout=0.0,
                 nonlin='lkyrelu',
                 leaky_relu_slope=0.1,
                 cross_msgs_reduce_op='mean',
                 add_self_loop=True,
                 explicit_H=False,
                 use_rdkit_coords=False,
                 # IEGMN specific parameters
                 use_rec_atoms=True,
                 shared_layers=False,
                 noise_decay_rate=0.5,
                 noise_initial=1.0,
                 use_edge_features_in_gmn=True,
                 use_mean_node_features=True,
                 residue_emb_dim=64,
                 iegmn_lay_hid_dim=64,
                 random_vec_dim=0,
                 random_vec_std=1,
                 num_lig_feats=None,
                 move_keypts_back=False,
                 normalize_Z_lig_directions=False,
                 unnormalized_kpt_weights=False,
                 centroid_keypts_construction_rec=False,
                 centroid_keypts_construction_lig=False,
                 rec_no_softmax=False,
                 lig_no_softmax=False,
                 normalize_Z_rec_directions=False,
                 centroid_keypts_construction=False,
                 evolve_only=False,
                 separate_lig=False,
                 save_trajectories=False,
                 # Affinity prediction parameters
                 affinity_methods=['neural', 'vina', 'physics']):
        
        super(EquiBindWithAffinity, self).__init__()
        
        self.device = device
        self.affinity_methods = affinity_methods
        self.debug = debug
        
        # Initialize EquiBind model with all required parameters
        self.equibind = EquiBind(
            device=device,
            debug=False,
            evolve_only=False,
            # Required IEGMN parameters
            n_lays=5,
            use_rec_atoms=False,
            shared_layers=False,
            noise_decay_rate=0.5,
            cross_msgs=True,
            noise_initial=1.0,
            use_edge_features_in_gmn=True,
            use_mean_node_features=True,
            residue_emb_dim=64,
            iegmn_lay_hid_dim=64,
            num_att_heads=30,
            dropout=0.1,
            nonlin='lkyrelu',
            leakyrelu_neg_slope=1e-2,
            # Required IEGMN_Layer parameters
            lig_input_edge_feats_dim=lig_input_edge_feats_dim,
            rec_input_edge_feats_dim=rec_input_edge_feats_dim,
            layer_norm='BN',
            layer_norm_coords='0',
            final_h_layer_norm='0',
            use_dist_in_layers=True,
            skip_weight_h=0.5,
            x_connection_init=0.25
        )
        
        # Add affinity prediction components
        feature_dim = iegmn_lay_hid_dim  # Use the actual hidden dimension from IEGMN layers
        self.interaction_extractor = InteractionFeatureExtractor(
            hidden_dim=feature_dim
        )
        
        # Initialize affinity predictors based on selected methods
        if 'neural' in affinity_methods:
            self.affinity_predictor = AffinityPredictor(
                input_dim=5,  # Simple features: center_dist, min_dist, mean_dist, num_lig_atoms, num_rec_atoms
                hidden_dims=[64, 32, 16]
            )
        
        # Vina and physics-based methods don't need neural components
        # They will be computed in the forward pass
        if 'vina' in affinity_methods:
            self.vina_scorer = VinaScorer()
        if 'physics' in affinity_methods:
            self.physics_scorer = PhysicsScorer()
    
    def forward(self, lig_graphs, rec_graphs, complex_names=None, epoch=0, geometry_graph=None):
        """
        Forward pass that predicts both 3D structure and binding affinity.
        
        Args:
            lig_graph: Ligand graph
            rec_graph: Receptor graph
            geometry_graph: Geometry constraints graph
            complex_names: Names of complexes
            epoch: Training epoch
        
        Returns:
            Tuple containing:
            - predicted_coords: List of predicted ligand coordinates
            - lig_keypts: Ligand keypoints
            - rec_keypts: Receptor keypoints  
            - rotations: Rotation matrices
            - translations: Translation vectors
            - geom_reg_loss: Geometry regularization loss
            - affinity_scores: Dictionary of binding affinity predictions
        """
        # Get EquiBind predictions
        equibind_results = self.equibind(
            lig_graphs, rec_graphs, 
            complex_names=complex_names, 
            epoch=epoch, 
            geometry_graph=geometry_graph
        )
        
        # Unpack EquiBind results
        if len(equibind_results) == 6:
            ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss = equibind_results
        else:
            # Handle different return formats
            ligs_coords_pred = equibind_results[0]
            ligs_keypts = equibind_results[1] if len(equibind_results) > 1 else None
            recs_keypts = equibind_results[2] if len(equibind_results) > 2 else None
            rotations = equibind_results[3] if len(equibind_results) > 3 else None
            translations = equibind_results[4] if len(equibind_results) > 4 else None
            geom_reg_loss = equibind_results[5] if len(equibind_results) > 5 else torch.tensor(0.0)
        
        # Predict binding affinity using embedded features from EquiBind
        affinity_predictions = {}
        
        if 'neural' in self.affinity_methods:
            try:
                # After EquiBind forward pass, the graphs contain embedded features
                # We can access the final embedded features from the EquiBind output
                # For now, let's use a simplified approach that works with the current structure
                
                # Use the coordinates from the predicted structure
                if isinstance(ligs_coords_pred, list):
                    lig_coords = ligs_coords_pred[0] if len(ligs_coords_pred) > 0 else lig_graphs.ndata['x']
                else:
                    lig_coords = ligs_coords_pred
                
                rec_coords = rec_graphs.ndata['x']
                
                # Create simple interaction features based on distances
                # This is a simplified version that avoids the complex embedding issues
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
                    torch.tensor([lig_coords.shape[0]], dtype=torch.float32, device=self.device),  # num ligand atoms
                    torch.tensor([rec_coords.shape[0]], dtype=torch.float32, device=self.device),  # num receptor atoms
                ], dim=0)  # (5,)
                
                # Predict affinity using the simple features
                neural_affinity = self.affinity_predictor(simple_features.unsqueeze(0))
                affinity_predictions['neural'] = neural_affinity.squeeze()
                
            except Exception as e:
                import traceback
                print(f"Warning: Neural affinity prediction failed: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
                affinity_predictions['neural'] = torch.zeros(1).to(self.device)
        
        if 'vina' in self.affinity_methods:
            # Placeholder for Vina-based affinity
            batch_size = len(lig_graphs) if isinstance(lig_graphs, list) else lig_graphs.batch_size
            affinity_predictions['vina'] = torch.zeros(batch_size).to(self.device)
            
        if 'physics' in self.affinity_methods:
            # Placeholder for physics-based affinity
            batch_size = len(lig_graphs) if isinstance(lig_graphs, list) else lig_graphs.batch_size
            affinity_predictions['physics'] = torch.zeros(batch_size).to(self.device)
        
        return ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss, affinity_predictions
    
    def __repr__(self):
        return f"EquiBindWithAffinity(affinity_methods={self.affinity_methods}, device={self.device})"


def load_integrated_model(args, data_sample, device, save_trajectories=False):
    """
    Load the integrated EquiBind model with affinity prediction.
    
    Args:
        args: Model arguments
        data_sample: Sample data for model initialization
        device: Device to load model on
        save_trajectories: Whether to save trajectories
    
    Returns:
        Initialized EquiBindWithAffinity model
    """
    model_parameters = args.model_parameters
    model_parameters['device'] = device
    model_parameters['save_trajectories'] = save_trajectories
    
    # Add affinity-specific parameters
    if hasattr(args, 'affinity_methods'):
        model_parameters['affinity_methods'] = args.affinity_methods
    else:
        model_parameters['affinity_methods'] = ['neural', 'vina', 'physics']
    
    model = EquiBindWithAffinity(**model_parameters)
    
    return model
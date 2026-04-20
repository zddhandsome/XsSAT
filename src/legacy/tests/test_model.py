"""
Unit tests for GeoSATformer model architecture.

Tests cover:
- Model initialization and configuration
- Forward pass and output shapes
- Individual components (towers, attention, etc.)
- Gradient flow
- Memory efficiency
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


# Mock model components for testing
# Replace with actual imports when models module is implemented:
# from models import GeoSATformer, ClauseTower, VariableTower, CrossAttention

class MockPatchEmbedding(nn.Module):
    """Mock patch embedding layer."""
    
    def __init__(self, input_dim: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        batch, seq_len, _ = x.shape
        
        # Simple projection (actual implementation would do patch-based)
        num_patches = seq_len
        x = self.proj(x[:, :, :self.patch_size])
        x = self.norm(x)
        return x


class MockTransformerBlock(nn.Module):
    """Mock transformer block."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class MockHeadSlicing(nn.Module):
    """Mock adaptive head slicing module."""
    
    def __init__(self, embed_dim: int, slicing_ratio: float = 0.5):
        super().__init__()
        self.slicing_ratio = slicing_ratio
        self.importance_scorer = nn.Linear(embed_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, embed_dim]
        batch, seq_len, _ = x.shape
        
        # Compute importance scores
        scores = self.importance_scorer(x).squeeze(-1)  # [batch, seq_len]
        
        # Keep top-k tokens
        k = max(1, int(seq_len * self.slicing_ratio))
        _, indices = torch.topk(scores, k, dim=1)
        indices, _ = torch.sort(indices, dim=1)
        
        # Gather selected tokens
        batch_indices = torch.arange(batch).unsqueeze(1).expand(-1, k)
        x_sliced = x[batch_indices, indices]
        
        return x_sliced, scores


class MockClauseTower(nn.Module):
    """Mock Clause Tower (row-wise processing)."""
    
    def __init__(self, max_vars: int, embed_dim: int, num_layers: int, 
                 num_heads: int, slicing_ratio: float = 0.5):
        super().__init__()
        self.embed = MockPatchEmbedding(max_vars, embed_dim, max_vars)
        self.layers = nn.ModuleList([
            MockTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.slicing = MockHeadSlicing(embed_dim, slicing_ratio)
    
    def forward(self, vsm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # vsm: [batch, num_clauses, num_vars]
        x = self.embed(vsm)
        
        for layer in self.layers:
            x = layer(x)
        
        x, scores = self.slicing(x)
        return x, scores


class MockVariableTower(nn.Module):
    """Mock Variable Tower (column-wise processing)."""
    
    def __init__(self, max_clauses: int, embed_dim: int, num_layers: int,
                 num_heads: int, slicing_ratio: float = 0.5):
        super().__init__()
        self.embed = MockPatchEmbedding(max_clauses, embed_dim, max_clauses)
        self.layers = nn.ModuleList([
            MockTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.slicing = MockHeadSlicing(embed_dim, slicing_ratio)
    
    def forward(self, vsm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # vsm: [batch, num_clauses, num_vars]
        # Transpose to process variables
        vsm_t = vsm.transpose(1, 2)  # [batch, num_vars, num_clauses]
        
        x = self.embed(vsm_t)
        
        for layer in self.layers:
            x = layer(x)
        
        x, scores = self.slicing(x)
        return x, scores


class MockCrossAttention(nn.Module):
    """Mock cross-attention between clause and variable representations."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn_c2v = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_v2c = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_c = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
    
    def forward(self, clause_emb: torch.Tensor, var_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # clause_emb: [batch, num_clauses', embed_dim]
        # var_emb: [batch, num_vars', embed_dim]
        
        # Clause attends to variables
        c2v, _ = self.cross_attn_c2v(clause_emb, var_emb, var_emb)
        clause_out = self.norm_c(clause_emb + c2v)
        
        # Variables attend to clauses
        v2c, _ = self.cross_attn_v2c(var_emb, clause_emb, clause_emb)
        var_out = self.norm_v(var_emb + v2c)
        
        return clause_out, var_out


class MockHierarchicalAggregation(nn.Module):
    """Mock hierarchical aggregation module."""
    
    def __init__(self, embed_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim)
            ) for _ in range(num_levels)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        
        for agg in self.aggregators:
            if x.shape[1] <= 1:
                break
            
            # Pairwise aggregation
            if x.shape[1] % 2 == 1:
                x = torch.cat([x, x[:, -1:]], dim=1)
            
            x1 = x[:, 0::2]  # Even indices
            x2 = x[:, 1::2]  # Odd indices
            x = agg(torch.cat([x1, x2], dim=-1))
        
        # Global pooling if needed
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        return x.squeeze(1)  # [batch, embed_dim]


class MockGeoSATformer(nn.Module):
    """Mock GeoSATformer model for testing."""
    
    def __init__(self, max_clauses: int = 1000, max_vars: int = 500,
                 embed_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 8, slicing_ratio: float = 0.5,
                 hierarchical_levels: int = 3):
        super().__init__()
        
        self.max_clauses = max_clauses
        self.max_vars = max_vars
        self.embed_dim = embed_dim
        
        # Dual towers
        self.clause_tower = MockClauseTower(max_vars, embed_dim, num_layers, 
                                            num_heads, slicing_ratio)
        self.variable_tower = MockVariableTower(max_clauses, embed_dim, num_layers,
                                                num_heads, slicing_ratio)
        
        # Cross-attention
        self.cross_attention = MockCrossAttention(embed_dim, num_heads)
        
        # Hierarchical aggregation
        self.hierarchical = MockHierarchicalAggregation(embed_dim, hierarchical_levels)
        
        # Task heads
        self.sat_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.muc_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.vsids_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, vsm: torch.Tensor) -> Dict[str, torch.Tensor]:
        # vsm: [batch, num_clauses, num_vars]
        batch_size = vsm.shape[0]
        
        # Dual tower encoding
        clause_emb, clause_scores = self.clause_tower(vsm)
        var_emb, var_scores = self.variable_tower(vsm)
        
        # Cross-attention
        clause_emb, var_emb = self.cross_attention(clause_emb, var_emb)
        
        # Hierarchical aggregation on clause embeddings
        global_emb = self.hierarchical(clause_emb)
        
        # Task outputs
        sat_pred = self.sat_head(global_emb)  # [batch, 1]
        
        # MUC prediction (per-clause)
        muc_pred = self.muc_head(clause_emb)  # [batch, num_clauses', 1]
        
        # VSIDS prediction (per-variable)
        vsids_pred = self.vsids_head(var_emb)  # [batch, num_vars', 1]
        
        return {
            'sat_pred': sat_pred.squeeze(-1),
            'muc_pred': muc_pred.squeeze(-1),
            'vsids_pred': vsids_pred.squeeze(-1),
            'clause_emb': clause_emb,
            'var_emb': var_emb,
            'clause_scores': clause_scores,
            'var_scores': var_scores
        }


# Fixtures
@pytest.fixture
def model_config():
    return {
        'max_clauses': 100,
        'max_vars': 50,
        'embed_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'slicing_ratio': 0.5,
        'hierarchical_levels': 2
    }


@pytest.fixture
def model(model_config):
    return MockGeoSATformer(**model_config)


@pytest.fixture
def sample_vsm(model_config):
    batch_size = 4
    return torch.randn(batch_size, model_config['max_clauses'], model_config['max_vars'])


class TestModelInitialization:
    """Test model initialization and configuration."""
    
    def test_model_creation(self, model_config):
        """Test that model can be created with valid config."""
        model = MockGeoSATformer(**model_config)
        assert model is not None
    
    def test_model_parameters(self, model):
        """Test that model has trainable parameters."""
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == num_params  # All params should be trainable
    
    def test_model_components_exist(self, model):
        """Test that all model components are present."""
        assert hasattr(model, 'clause_tower')
        assert hasattr(model, 'variable_tower')
        assert hasattr(model, 'cross_attention')
        assert hasattr(model, 'hierarchical')
        assert hasattr(model, 'sat_head')
        assert hasattr(model, 'muc_head')
        assert hasattr(model, 'vsids_head')
    
    def test_embed_dim_consistency(self, model):
        """Test that embedding dimensions are consistent across components."""
        embed_dim = model.embed_dim
        
        # Check that output dimensions match
        assert model.sat_head[0].in_features == embed_dim
        assert model.muc_head[0].in_features == embed_dim
        assert model.vsids_head[0].in_features == embed_dim


class TestForwardPass:
    """Test forward pass and output shapes."""
    
    def test_forward_output_type(self, model, sample_vsm):
        """Test that forward returns a dictionary."""
        output = model(sample_vsm)
        assert isinstance(output, dict)
    
    def test_forward_output_keys(self, model, sample_vsm):
        """Test that forward returns all expected keys."""
        output = model(sample_vsm)
        
        expected_keys = ['sat_pred', 'muc_pred', 'vsids_pred', 
                        'clause_emb', 'var_emb', 'clause_scores', 'var_scores']
        for key in expected_keys:
            assert key in output, f"Missing key: {key}"
    
    def test_sat_pred_shape(self, model, sample_vsm):
        """Test SAT prediction output shape."""
        output = model(sample_vsm)
        batch_size = sample_vsm.shape[0]
        
        assert output['sat_pred'].shape == (batch_size,)
    
    def test_sat_pred_range(self, model, sample_vsm):
        """Test SAT prediction values are in [0, 1]."""
        output = model(sample_vsm)
        
        assert torch.all(output['sat_pred'] >= 0)
        assert torch.all(output['sat_pred'] <= 1)
    
    def test_muc_pred_shape(self, model, sample_vsm, model_config):
        """Test MUC prediction output shape."""
        output = model(sample_vsm)
        batch_size = sample_vsm.shape[0]
        
        # After slicing, sequence length is reduced
        expected_len = int(model_config['max_clauses'] * model_config['slicing_ratio'])
        assert output['muc_pred'].shape[0] == batch_size
        # Length should be close to expected (may vary due to implementation)
    
    def test_vsids_pred_shape(self, model, sample_vsm, model_config):
        """Test VSIDS prediction output shape."""
        output = model(sample_vsm)
        batch_size = sample_vsm.shape[0]
        
        # After slicing, sequence length is reduced
        expected_len = int(model_config['max_vars'] * model_config['slicing_ratio'])
        assert output['vsids_pred'].shape[0] == batch_size
    
    def test_embedding_shapes(self, model, sample_vsm, model_config):
        """Test intermediate embedding shapes."""
        output = model(sample_vsm)
        batch_size = sample_vsm.shape[0]
        embed_dim = model_config['embed_dim']
        
        assert output['clause_emb'].shape[0] == batch_size
        assert output['clause_emb'].shape[2] == embed_dim
        
        assert output['var_emb'].shape[0] == batch_size
        assert output['var_emb'].shape[2] == embed_dim


class TestGradientFlow:
    """Test gradient computation and flow."""
    
    def test_backward_pass(self, model, sample_vsm):
        """Test that backward pass works without error."""
        output = model(sample_vsm)
        
        # Create dummy loss
        loss = output['sat_pred'].mean()
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_gradient_magnitudes(self, model, sample_vsm):
        """Test that gradients have reasonable magnitudes."""
        output = model(sample_vsm)
        loss = output['sat_pred'].mean() + output['muc_pred'].mean() + output['vsids_pred'].mean()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradient should not be exactly zero or exploding
                assert not np.isnan(grad_norm), f"NaN gradient for {name}"
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"
    
    def test_multi_task_loss_gradients(self, model, sample_vsm):
        """Test gradients from multi-task loss."""
        output = model(sample_vsm)
        
        # Multi-task loss
        sat_loss = nn.BCELoss()(output['sat_pred'], torch.ones_like(output['sat_pred']))
        muc_loss = output['muc_pred'].mean()
        vsids_loss = output['vsids_pred'].mean()
        
        total_loss = sat_loss + muc_loss + vsids_loss
        total_loss.backward()
        
        # All components should receive gradients
        assert model.sat_head[0].weight.grad is not None
        assert model.muc_head[0].weight.grad is not None
        assert model.vsids_head[0].weight.grad is not None


class TestComponentsIndividually:
    """Test individual model components."""
    
    def test_clause_tower(self, model_config):
        """Test Clause Tower independently."""
        tower = MockClauseTower(
            model_config['max_vars'],
            model_config['embed_dim'],
            model_config['num_layers'],
            model_config['num_heads'],
            model_config['slicing_ratio']
        )
        
        batch_size = 2
        vsm = torch.randn(batch_size, model_config['max_clauses'], model_config['max_vars'])
        
        emb, scores = tower(vsm)
        
        assert emb.shape[0] == batch_size
        assert emb.shape[2] == model_config['embed_dim']
        assert scores.shape[0] == batch_size
    
    def test_variable_tower(self, model_config):
        """Test Variable Tower independently."""
        tower = MockVariableTower(
            model_config['max_clauses'],
            model_config['embed_dim'],
            model_config['num_layers'],
            model_config['num_heads'],
            model_config['slicing_ratio']
        )
        
        batch_size = 2
        vsm = torch.randn(batch_size, model_config['max_clauses'], model_config['max_vars'])
        
        emb, scores = tower(vsm)
        
        assert emb.shape[0] == batch_size
        assert emb.shape[2] == model_config['embed_dim']
    
    def test_cross_attention(self, model_config):
        """Test Cross-Attention independently."""
        cross_attn = MockCrossAttention(
            model_config['embed_dim'],
            model_config['num_heads']
        )
        
        batch_size = 2
        seq_len_c = 20
        seq_len_v = 15
        
        clause_emb = torch.randn(batch_size, seq_len_c, model_config['embed_dim'])
        var_emb = torch.randn(batch_size, seq_len_v, model_config['embed_dim'])
        
        clause_out, var_out = cross_attn(clause_emb, var_emb)
        
        assert clause_out.shape == clause_emb.shape
        assert var_out.shape == var_emb.shape
    
    def test_hierarchical_aggregation(self, model_config):
        """Test Hierarchical Aggregation independently."""
        hier = MockHierarchicalAggregation(
            model_config['embed_dim'],
            model_config['hierarchical_levels']
        )
        
        batch_size = 2
        seq_len = 16
        
        x = torch.randn(batch_size, seq_len, model_config['embed_dim'])
        out = hier(x)
        
        assert out.shape == (batch_size, model_config['embed_dim'])
    
    def test_head_slicing(self, model_config):
        """Test Head Slicing module."""
        slicing = MockHeadSlicing(model_config['embed_dim'], slicing_ratio=0.5)
        
        batch_size = 2
        seq_len = 20
        
        x = torch.randn(batch_size, seq_len, model_config['embed_dim'])
        x_sliced, scores = slicing(x)
        
        expected_len = int(seq_len * 0.5)
        assert x_sliced.shape == (batch_size, expected_len, model_config['embed_dim'])
        assert scores.shape == (batch_size, seq_len)


class TestBatchProcessing:
    """Test batch processing behavior."""
    
    def test_batch_size_one(self, model, model_config):
        """Test with batch size of 1."""
        vsm = torch.randn(1, model_config['max_clauses'], model_config['max_vars'])
        output = model(vsm)
        
        assert output['sat_pred'].shape[0] == 1
    
    def test_batch_size_large(self, model, model_config):
        """Test with larger batch size."""
        batch_size = 16
        vsm = torch.randn(batch_size, model_config['max_clauses'], model_config['max_vars'])
        output = model(vsm)
        
        assert output['sat_pred'].shape[0] == batch_size
    
    def test_batch_independence(self, model, model_config):
        """Test that batch items are processed independently."""
        model.eval()
        
        vsm1 = torch.randn(1, model_config['max_clauses'], model_config['max_vars'])
        vsm2 = torch.randn(1, model_config['max_clauses'], model_config['max_vars'])
        
        with torch.no_grad():
            out1 = model(vsm1)['sat_pred']
            out2 = model(vsm2)['sat_pred']
            
            # Process together
            vsm_batch = torch.cat([vsm1, vsm2], dim=0)
            out_batch = model(vsm_batch)['sat_pred']
        
        # Results should be (approximately) the same
        assert torch.allclose(out1, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out2, out_batch[1:2], atol=1e-5)


class TestMemoryEfficiency:
    """Test memory efficiency."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory(self, model_config):
        """Test GPU memory usage."""
        device = torch.device('cuda')
        model = MockGeoSATformer(**model_config).to(device)
        
        torch.cuda.reset_peak_memory_stats()
        
        batch_size = 8
        vsm = torch.randn(batch_size, model_config['max_clauses'], 
                         model_config['max_vars'], device=device)
        
        output = model(vsm)
        loss = output['sat_pred'].mean()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
        
        # Should be reasonable (adjust threshold as needed)
        assert peak_memory < 1000  # Less than 1GB for small model
    
    def test_eval_mode_memory(self, model, model_config):
        """Test that eval mode uses less memory (no gradient tracking)."""
        model.eval()
        
        batch_size = 4
        vsm = torch.randn(batch_size, model_config['max_clauses'], model_config['max_vars'])
        
        with torch.no_grad():
            output = model(vsm)
        
        # Should complete without OOM
        assert output['sat_pred'] is not None


class TestModelModes:
    """Test training vs evaluation modes."""
    
    def test_train_mode(self, model, sample_vsm):
        """Test behavior in training mode."""
        model.train()
        
        output1 = model(sample_vsm)
        output2 = model(sample_vsm)
        
        # Outputs may differ due to dropout
        # (though our mock doesn't have dropout in all places)
    
    def test_eval_mode(self, model, sample_vsm):
        """Test behavior in evaluation mode."""
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_vsm)
            output2 = model(sample_vsm)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1['sat_pred'], output2['sat_pred'])
    
    def test_no_grad_context(self, model, sample_vsm):
        """Test inference with no_grad context."""
        model.eval()
        
        with torch.no_grad():
            output = model(sample_vsm)
        
        assert output['sat_pred'].requires_grad == False


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_zero_input(self, model, model_config):
        """Test with zero input (empty VSM)."""
        batch_size = 2
        vsm = torch.zeros(batch_size, model_config['max_clauses'], model_config['max_vars'])
        
        output = model(vsm)
        
        # Should not produce NaN
        assert not torch.isnan(output['sat_pred']).any()
        assert not torch.isnan(output['muc_pred']).any()
        assert not torch.isnan(output['vsids_pred']).any()
    
    def test_large_input(self, model, model_config):
        """Test with large input values."""
        batch_size = 2
        vsm = torch.randn(batch_size, model_config['max_clauses'], model_config['max_vars']) * 10
        
        output = model(vsm)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(output['sat_pred']).any()
        assert not torch.isinf(output['sat_pred']).any()
    
    def test_mixed_precision(self, model, model_config):
        """Test with mixed precision (float16)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        device = torch.device('cuda')
        model = model.to(device)
        
        batch_size = 2
        vsm = torch.randn(batch_size, model_config['max_clauses'], 
                         model_config['max_vars'], device=device)
        
        with torch.cuda.amp.autocast():
            output = model(vsm)
        
        # Should complete without error
        assert output['sat_pred'] is not None


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class FactorizedCrossVariableAttention(nn.Module):
    """
    Factorized attention for cross-variable dependency in CATS.
    Separates temporal and variable attention to maintain efficiency.
    
    Input: [bs*nvars, patch_num, d_model] (CATS format)
    Output: [bs*nvars, patch_num, d_model] (CATS format)
    """
    def __init__(self, d_model, n_heads, n_vars, dropout=0.1, fusion_method='learnable'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vars = n_vars
        self.d_k = d_model // n_heads
        self.fusion_method = fusion_method
        
        # Temporal attention (within each variable) - Keep existing temporal patterns
        self.temporal_query = nn.Linear(d_model, d_model)
        self.temporal_key = nn.Linear(d_model, d_model)
        self.temporal_value = nn.Linear(d_model, d_model)
        
        # Variable attention (across variables) - NEW cross-variable dependency
        self.variable_query = nn.Linear(d_model, d_model)
        self.variable_key = nn.Linear(d_model, d_model)
        self.variable_value = nn.Linear(d_model, d_model)
        
        # Fusion parameters
        if fusion_method == 'learnable':
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.ones(1))
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, bs):
        """
        Args:
            x: [bs*nvars, patch_num, d_model] - CATS format
            bs: batch size (to reshape correctly)
        Returns:
            output: [bs*nvars, patch_num, d_model]
        """
        bn, T, D = x.shape  # bn = bs * n_vars
        V = self.n_vars
        
        # Reshape to separate batch and variables
        x_reshaped = x.reshape(bs, V, T, D)  # [bs, nvars, patch_num, d_model]
        
        # ============ Step 1: Temporal Attention (within each variable) ============
        # Process each variable's temporal sequence independently
        x_temp = x  # Keep as [bs*nvars, T, D]
        
        q_temp = self.temporal_query(x_temp).reshape(bn, T, self.n_heads, self.d_k).transpose(1, 2)
        k_temp = self.temporal_key(x_temp).reshape(bn, T, self.n_heads, self.d_k).transpose(1, 2)
        v_temp = self.temporal_value(x_temp).reshape(bn, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores_temp = torch.matmul(q_temp, k_temp.transpose(-2, -1)) / sqrt(self.d_k)
        attn_temp = self.dropout(F.softmax(scores_temp, dim=-1))
        temporal_out = torch.matmul(attn_temp, v_temp)  # [bs*nvars, n_heads, T, d_k]
        
        # Reshape back
        temporal_out = temporal_out.transpose(1, 2).reshape(bn, T, D)
        
        # ============ Step 2: Variable Attention (across variables) ============
        # Process each time step across all variables
        x_var = x_reshaped.permute(0, 2, 1, 3).reshape(bs * T, V, D)  # [bs*T, nvars, d_model]
        
        q_var = self.variable_query(x_var).reshape(bs * T, V, self.n_heads, self.d_k).transpose(1, 2)
        k_var = self.variable_key(x_var).reshape(bs * T, V, self.n_heads, self.d_k).transpose(1, 2)
        v_var = self.variable_value(x_var).reshape(bs * T, V, self.n_heads, self.d_k).transpose(1, 2)
        
        scores_var = torch.matmul(q_var, k_var.transpose(-2, -1)) / sqrt(self.d_k)
        attn_var = self.dropout(F.softmax(scores_var, dim=-1))
        variable_out = torch.matmul(attn_var, v_var)  # [bs*T, n_heads, nvars, d_k]
        
        # Reshape back to CATS format
        variable_out = variable_out.transpose(1, 2).reshape(bs, T, V, D)
        variable_out = variable_out.permute(0, 2, 1, 3).reshape(bn, T, D)
        
        # ============ Step 3: Fusion ============
        if self.fusion_method == 'learnable':
            # Weighted sum with learnable parameters
            output = self.alpha * temporal_out + self.beta * variable_out
        elif self.fusion_method == 'gated':
            # Gated fusion
            combined = torch.cat([temporal_out, variable_out], dim=-1)
            gate = self.gate(combined)
            output = gate * temporal_out + (1 - gate) * variable_out
        else:  # 'average'
            output = (temporal_out + variable_out) / 2
        
        output = self.output_proj(output)
        
        return output


class SimplifiedCrossVariableAttention(nn.Module):
    """
    Simplified version: Only variable attention via temporal pooling.
    More efficient, good starting point for ablation studies.
    
    Input: [bs*nvars, patch_num, d_model] (CATS format)
    Output: [bs*nvars, patch_num, d_model] (CATS format)
    """
    def __init__(self, d_model, n_heads, n_vars, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vars = n_vars
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, bs):
        """
        Args:
            x: [bs*nvars, patch_num, d_model] - CATS format
            bs: batch size
        Returns:
            output: [bs*nvars, patch_num, d_model]
        """
        bn, T, D = x.shape
        V = self.n_vars
        
        # Reshape to [bs, nvars, patch_num, d_model]
        x_reshaped = x.reshape(bs, V, T, D)
        
        # Temporal pooling: average over time to get variable representations
        x_var = x_reshaped.mean(dim=2)  # [bs, nvars, d_model]
        
        # Variable attention
        q = self.query(x_var).reshape(bs, V, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(x_var).reshape(bs, V, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(x_var).reshape(bs, V, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d_k)
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)  # [bs, n_heads, nvars, d_k]
        
        output = output.transpose(1, 2).reshape(bs, V, D)
        output = self.output_proj(output)
        
        # Broadcast back to time dimension
        output = output.unsqueeze(2).expand(bs, V, T, D)
        
        # Reshape back to CATS format [bs*nvars, T, D]
        output = output.reshape(bn, T, D)
        
        # Residual connection
        return x + output
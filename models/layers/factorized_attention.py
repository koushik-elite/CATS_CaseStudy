class FactorizedCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
        attention_dropout: float = 0.1,
        output_attention: bool = True,
    ):
        super().__init__()
        
        # Temporal cross-attention (per variable)
        self.temporal_attention = CrossAttention(
            d_model, n_heads, d_keys, d_values, 
            attention_dropout, output_attention
        )
        
        # Variable cross-attention (across variables)
        self.variable_attention = CrossAttention(
            d_model, n_heads, d_keys, d_values,
            attention_dropout, output_attention
        )
        
        # Layer norms
        self.temporal_norm = nn.LayerNorm(d_model)
        self.variable_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x_input,           # [B, M, L, D] - input sequences
        learnable_queries, # [H, D] - horizon queries
        attn_mask=None,
    ):
        B, M, L, D = x_input.shape
        H = learnable_queries.shape[0]
        
        # Step 1: Temporal attention for each variable
        temporal_outputs = []
        for m in range(M):
            x_var = x_input[:, m, :, :]  # [B, L, D]
            
            # Expand queries for batch processing
            queries = learnable_queries.unsqueeze(0).expand(B, -1, -1)
            
            # Cross-attention: future as query, past as key/value
            temp_out, temp_attn = self.temporal_attention(
                queries, x_var, x_var, attn_mask
            )
            temporal_outputs.append(temp_out)  # [B, H, D]
        
        temporal_outputs = torch.stack(temporal_outputs, dim=1)  # [B, M, H, D]
        temporal_outputs = self.temporal_norm(temporal_outputs)
        
        # Step 2: Variable attention for each horizon
        final_outputs = []
        for h in range(H):
            horizon_features = temporal_outputs[:, :, h, :]  # [B, M, D]
            
            # Cross-attention across variables
            var_out, var_attn = self.variable_attention(
                horizon_features, horizon_features, horizon_features
            )
            final_outputs.append(var_out)  # [B, M, D]
        
        final_outputs = torch.stack(final_outputs, dim=2)  # [B, M, H, D]
        final_outputs = self.variable_norm(final_outputs)
        
        return final_outputs
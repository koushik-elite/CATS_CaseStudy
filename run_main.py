from models.CATS_factorized import SparseVariableAttention, AdaptiveVariableRouter, VariableGraphAttention
import torch
import torch.nn as nn
from models.CATS_factorized import create_factorized_cats_model

if __name__ == "__main__":
    # Example usage and testing
    
    # Create dummy configuration
    class Config:
        seq_len = 96
        pred_len = 24
        enc_in = 7  # Number of variables
        d_model = 128
        n_heads = 8
        e_layers = 3
        d_ff = 512
        dropout = 0.1
        patch_len = 16
        stride = 8
        padding = 0
        output_attention = True
        variable_attention_type = 'full'
        masking_prob = 0.1
    
    configs = Config()
    
    # Create model
    model = create_factorized_cats_model(configs)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    
    output = model(x)
    if isinstance(output, tuple):
        output, attention = output
        print(f"Output shape: {output.shape}")
        print(f"Attention keys: {attention[0].keys() if attention else 'None'}")
    else:
        print(f"Output shape: {output.shape}")
    
    # Test different components
    print("\nTesting individual components:")
    
    # Test sparse attention
    sparse_attn = SparseVariableAttention(configs.d_model, configs.n_heads, top_k=3)
    x_var = torch.randn(batch_size, configs.enc_in, configs.d_model)
    out, attn = sparse_attn(x_var)
    print(f"Sparse attention output: {out.shape}")
    
    # Test adaptive router
    router = AdaptiveVariableRouter(configs.d_model, configs.enc_in, n_experts=4)
    var_ids = torch.arange(configs.enc_in)
    out = router(x_var, var_ids)
    print(f"Adaptive router output: {out.shape}")
    
    # Test graph attention
    graph_attn = VariableGraphAttention(configs.d_model, configs.n_heads)
    out, adj = graph_attn(x_var)
    print(f"Graph attention output: {out.shape}, Adjacency: {adj.shape if adj is not None else 'None'}")
    
    print("\nAll tests passed!")
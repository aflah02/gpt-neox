# Layer-Specific Positional Embedding Examples

This directory contains example configurations demonstrating the new `pos_emb_config` functionality.

## Files

### `alternating_pos_emb.json`
A 12-layer model with alternating rotary and alibi positional embeddings:
- Layers 0, 2, 4, 6, 8, 10: rotary embeddings
- Layers 1, 3, 5, 7, 9, 11: alibi embeddings

Configuration:
```json
"pos_emb_config": [["rotary", "alibi"], 6]
```

### `hybrid_pos_emb.json`
An 8-layer model with three different positional embedding types:
- Layers 0-1: learned embeddings
- Layers 2-5: rotary embeddings  
- Layers 6-7: alibi embeddings

Configuration:
```json
"pos_emb_config": [
  [["learned"], 2],
  [["rotary"], 4], 
  [["alibi"], 2]
]
```

## Usage

These configurations can be used as starting points for:
- Research into hybrid positional embedding architectures
- Ablation studies comparing different positional embedding strategies
- Models designed for specific sequence length requirements

To use these examples:
```bash
python train.py configs/examples/alternating_pos_emb.json
```

## Customization

Modify the `pos_emb_config` parameter to experiment with different combinations:
- All supported types: `learned`, `rotary`, `sinusoidal`, `rpe`, `alibi`, `none`
- Use `"all"` keyword to repeat patterns across all layers
- Mix different segment sizes for complex architectures

Remember to adjust other parameters like `num_layers`, `seq_length`, and training hyperparameters based on your specific requirements.
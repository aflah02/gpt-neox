# Layer-Specific Positional Embedding Configuration

This document describes the new `pos_emb_config` feature that allows you to specify different positional embeddings for different layers in GPT-NeoX, similar to how `attention_config` works for attention types.

## Overview

Previously, GPT-NeoX only supported a single global positional embedding type (via the `pos_emb` parameter) that was applied to all layers. Now you can specify different positional embedding types for different layers using the `pos_emb_config` parameter.

## Supported Positional Embedding Types

The same types supported by the global `pos_emb` parameter:
- `learned`: Learned positional embeddings
- `rotary`: Rotary Position Embedding (RoPE)
- `sinusoidal`: Sinusoidal positional embeddings  
- `rpe`: Relative Position Embedding (T5-style)
- `alibi`: Attention with Linear Biases (ALiBi)
- `none`: No positional embeddings

## Configuration Format

The `pos_emb_config` parameter follows the same format as `attention_config`:

```yaml
pos_emb_config: [
  [["pos_emb_type_1", "pos_emb_type_2", ...], repeat_count]
]
```

### Examples

#### Example 1: Single type for all layers
```yaml
num_layers: 12
pos_emb_config: [["rotary"], 12]
```
This applies rotary embeddings to all 12 layers.

#### Example 2: Alternating pattern
```yaml
num_layers: 12
pos_emb_config: [["rotary", "alibi"], 6]
```
This creates the pattern: `rotary, alibi, rotary, alibi, rotary, alibi, rotary, alibi, rotary, alibi, rotary, alibi`

#### Example 3: Complex mixed pattern
```yaml
num_layers: 8
pos_emb_config: [
  [["rotary"], 4],
  [["alibi"], 2], 
  [["none"], 2]
]
```
This creates: `rotary, rotary, rotary, rotary, alibi, alibi, none, none`

#### Example 4: Using "all" keyword
```yaml
num_layers: 12
pos_emb_config: [["rotary", "alibi"], "all"]
```
This repeats the pattern across all layers: `rotary, alibi, rotary, alibi, ...` (6 times total)

## Backward Compatibility

If `pos_emb_config` is not specified, the system automatically uses the global `pos_emb` parameter for all layers:

```yaml
num_layers: 12
pos_emb: "rotary"
# pos_emb_config is automatically set to [["rotary"], 12]
```

This ensures existing configurations continue to work without modification.

## Complete Configuration Example

```yaml
# Model architecture
num_layers: 8
hidden_size: 768
num_attention_heads: 12
seq_length: 2048
max_position_embeddings: 2048

# Global positional embedding (used as fallback)
pos_emb: "rotary"

# Layer-specific positional embeddings
pos_emb_config: [
  [["rotary"], 4],      # First 4 layers use rotary
  [["alibi"], 2],       # Next 2 layers use alibi  
  [["none"], 2]         # Last 2 layers use no positional embeddings
]

# Rest of configuration...
```

## Use Cases

### Hybrid Architectures
Combine different positional embedding approaches in a single model:
- Early layers with learned embeddings for local patterns
- Middle layers with rotary embeddings for medium-range dependencies
- Later layers with alibi for long-range attention

### Ablation Studies
Compare different positional embedding strategies by layer:
```yaml
pos_emb_config: [["rotary", "alibi", "learned", "none"], 3]
```

### Progressive Complexity
Start simple and add complexity in deeper layers:
```yaml
pos_emb_config: [
  [["none"], 2],        # No positional info in early layers
  [["learned"], 4],     # Add learned positions
  [["rotary"], 6]       # Full rotary in deeper layers  
]
```

## Implementation Notes

- Each layer's attention mechanism receives its specific positional embedding type
- The `rpe` (relative position embedding) parameter is automatically set based on layer configuration
- The `rotary` flag is automatically set for layers using rotary embeddings
- All validation ensures the expanded config matches the number of layers
- Layer numbering starts from 0 (first layer is layer 0)

## Error Handling

The system will validate your configuration and provide helpful error messages:

1. **Invalid embedding type**: `Positional embedding type 'invalid_type' not recognized`
2. **Length mismatch**: `Length of pos_emb_config list must equal num_layers`
3. **Pattern divisibility**: `Number of layers (12) is not divisible by the length of pattern: ['rotary', 'alibi', 'learned']`

## Migration Guide

To migrate existing configurations:

1. **No changes needed**: Existing configurations using only `pos_emb` continue to work
2. **Add layer-specific config**: Add `pos_emb_config` parameter alongside existing `pos_emb`
3. **Remove global fallback**: Once `pos_emb_config` is set, you can remove `pos_emb` if desired

The new feature provides the flexibility to explore novel positional embedding strategies while maintaining full backward compatibility with existing configurations.
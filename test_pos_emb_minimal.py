#!/usr/bin/env python3

"""
Minimal test script to validate the expand_pos_emb_types function logic.
"""

def expand_pos_emb_types(pos_emb_config, num_layers):
    """
    Expands a `pos_emb_config` list in the following format:

        [
        [['pos_emb_type_1', ..., `pos_emb_type_n`], 12]
        ]

    to a flattened list of length `num_layers`.

    :param pos_emb_config: positional embedding configuration list
    :param num_layers: number of layers
    :return: expanded list of positional embedding types
    """
    # if only strings are found in the config, we assume it's already expanded
    if all([isinstance(i, str) for i in pos_emb_config]):
        return pos_emb_config
    newlist = []
    for item in pos_emb_config:
        # instead of specifying a number - we can specify 'all' to extend this pattern across all layers
        if item[1] == "all":
            assert num_layers % len(item[0]) == 0, (
                f"Number of layers ({num_layers}) is not divisible by the length "
                f"of pattern: {item[0]}"
            )
            return item[0] * (num_layers // len(item[0]))
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist


def test_expand_pos_emb_types():
    """Test the expand_pos_emb_types function"""
    
    print("Testing expand_pos_emb_types function...")
    
    # Test case 1: Simple expansion
    config = [["rotary"], 4]
    result = expand_pos_emb_types([config], 4)
    expected = ["rotary", "rotary", "rotary", "rotary"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test 1 passed: Simple expansion")
    
    # Test case 2: Alternating pattern
    config = [["rotary", "alibi"], 2]
    result = expand_pos_emb_types([config], 4)
    expected = ["rotary", "alibi", "rotary", "alibi"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test 2 passed: Alternating pattern")
    
    # Test case 3: Already expanded list
    config = ["rotary", "alibi", "rotary", "alibi"]
    result = expand_pos_emb_types(config, 4)
    expected = ["rotary", "alibi", "rotary", "alibi"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test 3 passed: Already expanded list")
    
    # Test case 4: Using 'all' keyword
    config = [["rotary", "alibi"], "all"]
    result = expand_pos_emb_types([config], 6)
    expected = ["rotary", "alibi", "rotary", "alibi", "rotary", "alibi"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test 4 passed: Using 'all' keyword")
    
    # Test case 5: Mixed positional embedding types
    config = [["learned", "rotary", "alibi"], 1]
    result = expand_pos_emb_types([config], 3)
    expected = ["learned", "rotary", "alibi"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test 5 passed: Mixed positional embedding types")
    
    # Test case 6: Multiple segments
    config = [["rotary"], 2]
    config2 = [["alibi"], 2]
    result = expand_pos_emb_types([config, config2], 4)
    expected = ["rotary", "rotary", "alibi", "alibi"]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Test 6 passed: Multiple segments")
    
    print("All expand_pos_emb_types tests passed!")


if __name__ == "__main__":
    try:
        test_expand_pos_emb_types()
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
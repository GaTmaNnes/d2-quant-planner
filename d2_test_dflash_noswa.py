#!/usr/bin/env python3
"""Create a patched DFlash GGUF that removes sliding_window metadata to test if SWA causes the crash."""
import struct, os

def patch_gguf_remove_sw(input_path, output_path):
    """Read GGUF, find and remove sliding_window and sliding_window_pattern keys."""
    # Simply re-convert without SWA by modifying config
    import json
    with open('hf_weights_dflash/config.json', 'r') as f:
        config = json.load(f)
    
    # Disable sliding window
    config['use_sliding_window'] = False
    if 'sliding_window' in config:
        del config['sliding_window']
    if 'layer_types' in config:
        del config['layer_types']
    
    # Save patched config
    with open('hf_weights_dflash/config_noswa.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Patched config saved to hf_weights_dflash/config_noswa.json")
    print(f"use_sliding_window: {config.get('use_sliding_window', False)}")

if __name__ == '__main__':
    patch_gguf_remove_sw('hf_weights_dflash/config.json', 'hf_weights_dflash/config_noswa.json')

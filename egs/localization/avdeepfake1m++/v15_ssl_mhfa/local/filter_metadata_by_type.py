#!/usr/bin/env python3
"""
Filter metadata.json by modify_type and generate separate .scp files
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path

def filter_metadata_by_type(metadata_json, output_dir):
    """
    Filter metadata.json by modify_type and generate separate .scp files
    
    Args:
        metadata_json: Path to the metadata.json file
        output_dir: Directory to save the filtered .scp files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store data by modify_type
    data_by_type = defaultdict(list)
    
    # Read and filter metadata
    print(f"Reading {metadata_json}...")
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    # Group by modify_type
    for item in metadata:
        modify_type = item.get('modify_type', 'unknown')
        data_by_type[modify_type].append(item)
    
    # Generate .scp files for each modify_type
    print(f"\nFound modify_types: {list(data_by_type.keys())}\n")
    
    for modify_type, items in data_by_type.items():
        output_file = output_dir / f"metadata_{modify_type}.scp"
        
        with open(output_file, 'w') as f:
            for item in items:
                # Format: utt_id metadata_json_path
                # Using 'file' as utt_id and storing the json path
                utt_id = item['file']
                json_path = f"/export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/{item['split']}/{item['file'].replace('.mp4', '.json')}"  # Assuming corresponding json file
                f.write(f"{utt_id} {json_path}\n")
        
        print(f"✓ Generated {output_file} with {len(items)} items")
    
    # Generate summary
    print("\n" + "="*60)
    print("Summary:")
    for modify_type, items in sorted(data_by_type.items()):
        print(f"  {modify_type:20s}: {len(items):6d} items")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Filter metadata.json by modify_type and generate .scp files"
    )
    parser.add_argument(
        '--metadata_json',
        type=str,
        required=True,
        help='Path to metadata.json file (e.g., /export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/train_metadata.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save filtered .scp files'
    )
    
    args = parser.parse_args()
    
    filter_metadata_by_type(args.metadata_json, args.output_dir)

if __name__ == '__main__':
    main()

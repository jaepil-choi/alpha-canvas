"""
Experiment 01: YAML Config Loading

Date: 2025-01-20
Status: In Progress

Objective:
- Validate that we can load and parse config/data.yaml structure

Hypothesis:
- PyYAML can load the YAML file and provide dict-like access to nested fields

Success Criteria:
- [ ] Load config/data.yaml successfully
- [ ] Access field definitions: config['adj_close']['table'] → 'PRICEVOLUME'
- [ ] Access SQL query: config['adj_close']['query'] → full query string
- [ ] List all available field names
"""

import yaml
from pathlib import Path


def main():
    print("=" * 60)
    print("EXPERIMENT 01: YAML Config Loading")
    print("=" * 60)
    
    # Step 1: Load config/data.yaml
    print("\n[Step 1] Loading config/data.yaml...")
    config_path = Path('config') / 'data.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  [OK] File loaded successfully")
    except FileNotFoundError:
        print(f"  [FAIL] File not found: {config_path}")
        return
    except Exception as e:
        print(f"  [FAIL] Error loading file: {e}")
        return
    
    # Step 2: Parse field definitions
    print("\n[Step 2] Parsing field definitions...")
    field_names = list(config.keys())
    print(f"  [OK] Found {len(field_names)} field definitions: {', '.join(field_names)}")
    
    # Step 3: Access nested values
    print("\n[Step 3] Accessing nested values...")
    try:
        adj_close = config['adj_close']
        print(f"  [OK] adj_close.table = '{adj_close['table']}'")
        print(f"  [OK] adj_close.index_col = '{adj_close['index_col']}'")
        query_preview = adj_close['query'].strip()[:50] + "..."
        print(f"  [OK] adj_close.query = '{query_preview}'")
    except KeyError as e:
        print(f"  [FAIL] Missing key: {e}")
        return
    
    # Step 4: Validate structure
    print("\n[Step 4] Validating structure...")
    required_keys = ['table', 'index_col', 'security_col', 'value_col', 'query']
    all_valid = True
    
    for field_name, field_def in config.items():
        missing_keys = [key for key in required_keys if key not in field_def]
        if missing_keys:
            print(f"  [FAIL] Field '{field_name}' missing keys: {missing_keys}")
            all_valid = False
    
    if all_valid:
        print(f"  [OK] All fields have required keys: {', '.join(required_keys)}")
    
    # Verdict
    print("\n" + "=" * 60)
    if all_valid:
        print("VERDICT: [SUCCESS] - YAML config structure is valid and accessible")
    else:
        print("VERDICT: [FAILURE] - Some fields have invalid structure")
    print("=" * 60)


if __name__ == '__main__':
    main()


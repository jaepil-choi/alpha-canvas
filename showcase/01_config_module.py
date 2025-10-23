"""
Showcase 01: Config Module

This script demonstrates the ConfigLoader functionality, showing how
alpha-canvas loads and manages YAML configuration files.

Run: poetry run python showcase/01_config_module.py
"""

from alpha_canvas.core.config import ConfigLoader


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  ALPHA-CANVAS MVP SHOWCASE")
    print("  Phase 1: Config Module")
    print("=" * 70)
    
    # Section 1: Initialize ConfigLoader
    print_section("1. Initializing ConfigLoader")
    loader = ConfigLoader(config_dir='config')
    print("[OK] ConfigLoader initialized")
    print(f"     Config directory: {loader.config_dir}")
    
    # Section 2: List available fields
    print_section("2. Listing Available Data Fields")
    fields = loader.list_fields()
    print(f"[OK] Found {len(fields)} field definitions:")
    for i, field_name in enumerate(fields, 1):
        print(f"     {i}. {field_name}")
    
    # Section 3: Get specific field definition
    print_section("3. Accessing Field Definitions")
    
    # Example 1: adj_close
    print("\n  Field: 'adj_close'")
    adj_close = loader.get_field('adj_close')
    print(f"  [OK] table:        {adj_close['table']}")
    print(f"  [OK] index_col:    {adj_close['index_col']}")
    print(f"  [OK] security_col: {adj_close['security_col']}")
    print(f"  [OK] value_col:    {adj_close['value_col']}")
    print(f"  [OK] query length: {len(adj_close['query'])} characters")
    
    # Example 2: market_cap
    print("\n  Field: 'market_cap'")
    market_cap = loader.get_field('market_cap')
    print(f"  [OK] table:        {market_cap['table']}")
    print(f"  [OK] value_col:    {market_cap['value_col']}")
    
    # Example 3: subindustry (for grouping)
    print("\n  Field: 'subindustry' (grouping field)")
    subindustry = loader.get_field('subindustry')
    print(f"  [OK] table:        {subindustry['table']}")
    print(f"  [OK] value_col:    {subindustry['value_col']}")
    
    # Section 4: Validate structure
    print_section("4. Validating Configuration Structure")
    required_keys = ['table', 'index_col', 'security_col', 'value_col', 'query']
    all_valid = True
    
    for field_name in fields:
        field_def = loader.get_field(field_name)
        missing = [key for key in required_keys if key not in field_def]
        if missing:
            print(f"  [FAIL] '{field_name}' missing keys: {missing}")
            all_valid = False
    
    if all_valid:
        print(f"[OK] All {len(fields)} fields have required keys")
        print(f"     Required: {', '.join(required_keys)}")
    
    # Section 5: Error handling
    print_section("5. Error Handling")
    print("\n  Testing non-existent field...")
    try:
        loader.get_field('nonexistent_field')
        print("  [FAIL] Should have raised KeyError")
    except KeyError as e:
        print(f"  [OK] Correctly raised KeyError: {e}")
    
    # Final summary
    print_section("SUMMARY")
    print("[SUCCESS] Config Module Demonstration Complete")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ ConfigLoader initialization")
    print("  ✓ YAML file loading")
    print("  ✓ Field definition access")
    print("  ✓ Structure validation")
    print("  ✓ Error handling")
    print()
    print(f"Configuration Status: {len(fields)} fields loaded and validated")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()


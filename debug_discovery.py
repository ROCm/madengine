#!/usr/bin/env python3
"""Debug script to see what models are discovered and how tag matching works.

Usage:
    python3 debug_discovery.py [TAG]

Arguments:
    TAG: Optional tag to test (default: MAD/dummy_multi)

Examples:
    python3 debug_discovery.py
    python3 debug_discovery.py MAD/dummy_multi
    python3 debug_discovery.py MAD-private/some_model
"""

import sys
import os
import argparse

# Try to import from installed package first, then fall back to src/
try:
    from madengine.utils.discover_models import DiscoverModels
except ImportError:
    # Not installed, try from src/ (development mode)
    sys.path.insert(0, 'src')
    try:
        from madengine.utils.discover_models import DiscoverModels
    except ImportError:
        print("ERROR: Cannot import madengine. Either:")
        print("  1. Run 'pip install -e .' from madengine repository, or")
        print("  2. Run this script from madengine repository root")
        sys.exit(1)

# Show current directory context
print("=" * 60)
print("DEBUGGING MODEL DISCOVERY")
print("=" * 60)
print(f"Current directory: {os.getcwd()}")
print(f"Scripts directory exists: {os.path.exists('scripts')}")
print(f"Root models.json exists: {os.path.exists('models.json')}")

if os.path.exists('scripts'):
    print(f"\nContents of scripts/:")
    for item in os.listdir('scripts'):
        item_path = os.path.join('scripts', item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            # Check if it has models.json
            models_json = os.path.join(item_path, 'models.json')
            if os.path.exists(models_json):
                print(f"     ✓ has models.json")
            # Check for nested scripts
            nested_scripts = os.path.join(item_path, 'scripts')
            if os.path.exists(nested_scripts):
                print(f"     ✓ has scripts/ subdirectory")
        else:
            print(f"  📄 {item}")
print()

dm = DiscoverModels(args=argparse.Namespace(tags=None))

try:
    dm.discover_models()
    print(f"\n✅ Discovery successful! Found {len(dm.models)} models\n")

    print("All discovered models:")
    print("-" * 60)
    for i, model in enumerate(dm.models, 1):
        print(f"{i}. Name: {model['name']}")
        print(f"   Tags: {model.get('tags', [])}")
        print(f"   Scripts: {model.get('scripts', 'N/A')}")
        print(f"   Internal fs_path: {model.get('_fs_rel_path', 'N/A')}")
        print()

    # Now try with the specific tag
    test_tag = sys.argv[1] if len(sys.argv) > 1 else 'MAD/dummy_multi'
    print("\n" + "=" * 60)
    print(f"TESTING TAG: {test_tag}")
    print("=" * 60)

    dm2 = DiscoverModels(args=argparse.Namespace(tags=[test_tag]))
    dm2.discover_models()

    # Parse the tag to check expected matches
    if '/' in test_tag:
        scope, tag_filter = test_tag.split('/', 1)
        prefix = scope + "/"
        full_name_match = prefix + tag_filter
        dir_prefix_match = full_name_match + "/"

        print(f"\nModels that start with '{scope}/':")
        scope_models = [m for m in dm2.models if m['name'].startswith(prefix)]
        if scope_models:
            for model in scope_models:
                print(f"  - {model['name']}")
        else:
            print(f"  (none found)")

        print(f"\nModels that start with '{dir_prefix_match}':")
        dir_models = [m for m in dm2.models if m['name'].startswith(dir_prefix_match)]
        if dir_models:
            for model in dir_models:
                print(f"  - {model['name']}")
        else:
            print(f"  (none found)")

    print(f"\nAttempting selection with tag '{test_tag}'...")
    try:
        dm2.select_models()
        print(f"✅ SUCCESS! Selected {len(dm2.selected_models)} models:")
        for model in dm2.selected_models:
            print(f"  - {model['name']}")
    except ValueError as e:
        print(f"❌ FAILED: {e}")
        print("\nDEBUG: Checking why no matches...")

        # Parse the scoped tag if applicable
        if '/' in test_tag:
            scope, tag_filter = test_tag.split('/', 1)
            prefix = scope + "/"
            full_name_match = prefix + tag_filter
            dir_prefix_match = full_name_match + "/"

            print(f"  Scope: {scope}")
            print(f"  Tag filter: {tag_filter}")
            print(f"  Prefix: {prefix}")
            print(f"  Full name match: {full_name_match}")
            print(f"  Dir prefix match: {dir_prefix_match}")

            print(f"\n  Checking each model:")
            for model in dm2.models:
                name = model['name']
                has_tags = model.get('tags', [])

                starts_with_prefix = name.startswith(prefix)
                is_all = tag_filter == 'all'
                has_tag = tag_filter in has_tags if isinstance(has_tags, list) else False
                is_exact = name == full_name_match
                is_dir_match = name.startswith(dir_prefix_match)

                print(f"\n  Model: {name}")
                print(f"    Starts with '{prefix}': {starts_with_prefix}")
                if starts_with_prefix:
                    print(f"    Tag field has '{tag_filter}': {has_tag}")
                    print(f"    Is exact match '{full_name_match}': {is_exact}")
                    print(f"    Starts with dir '{dir_prefix_match}': {is_dir_match}")
                    print(f"    Would match: {is_all or has_tag or is_exact or is_dir_match}")
        else:
            # Unscoped tag
            print(f"  Tag '{test_tag}' is unscoped.")
            print(f"  Models with this tag in their 'tags' field:")
            for model in dm2.models:
                if test_tag in model.get('tags', []):
                    print(f"    - {model['name']}")

except Exception as e:
    print(f"\n❌ Discovery failed: {e}")
    import traceback
    traceback.print_exc()

# Debugging Model Discovery

If you're experiencing issues with model discovery or tag matching (e.g., "No models found corresponding to the given tag"), use the `debug_discovery.py` utility to diagnose the problem.

## Quick Start

From your MAD or MAD-internal repository (where you run madengine):

```bash
# Copy the debug script from madengine repository
cp /path/to/madengine/debug_discovery.py .

# Run it (requires madengine installed: pip install -e /path/to/madengine)
python3 debug_discovery.py [TAG]
```

## Examples

```bash
# Debug the default MAD/dummy_multi tag
python3 debug_discovery.py

# Debug a specific tag
python3 debug_discovery.py MAD/dummy_multi
python3 debug_discovery.py MAD-private/some_model
python3 debug_discovery.py category1
```

## What the Script Shows

1. **Current directory context**: Where you're running from, whether scripts/ exists
2. **Directory structure**: Contents of scripts/ directory
3. **All discovered models**: Names, tags, scripts paths, and internal filesystem paths
4. **Tag matching test**: Attempts to select models with your tag and shows why matches succeed/fail

## Common Issues

### Issue: "models.json file not found"

**Cause**: Running from wrong directory or missing root models.json

**Solution**: 
- Run from the repository root (where models.json should be)
- If using per-directory models.json (submodules), create an empty root models.json:
  ```bash
  echo "[]" > models.json
  ```

### Issue: "No models found corresponding to the given tag"

The debug script will show you exactly why:

1. **No models discovered at all**:
   - Check that scripts/SCOPE/ directories exist
   - Check that models.json files are present in the right locations

2. **Models discovered but don't match**:
   - Check the discovered model names (e.g., `MAD/dummy/dummy_multi/model1`)
   - Check if your tag matches the pattern
   - For scoped tags (SCOPE/filter), models must either:
     - Have `filter` in their tags field, OR
     - Have name exactly matching `SCOPE/filter`, OR
     - Have name starting with `SCOPE/filter/` (direct child directory), OR
     - Have `filter` appearing as any path component after `SCOPE/`

3. **Wrong model names**:
   - Model names are derived from directory structure
   - Interim "scripts" directories are stripped
   - Example: `scripts/MAD/scripts/dummy_multi/` → models named `MAD/dummy_multi/modelX`

## Tag Matching Rules

### Scoped Tags (format: `SCOPE/filter`)

Examples: `MAD/dummy_multi`, `MAD-private/some_model`

A model matches if ALL of these are true:
1. Model name starts with `SCOPE/`
2. At least ONE of:
   - `filter == "all"` (matches everything in that scope)
   - `filter` is in the model's `tags` field
   - Model name exactly equals `SCOPE/filter`
   - Model name starts with `SCOPE/filter/` (direct child directory)
   - `filter` appears as any path component after `SCOPE/` (flexible path matching)

### Unscoped Tags (format: `tag_name`)

Examples: `category1`, `test`

A model matches if:
- `tag_name` is in the model's `tags` field

## Directory Structure Examples

### Flat Structure
```
scripts/
└── MAD/
    └── dummy_multi/
        ├── models.json          # Models: MAD/dummy_multi/model1, MAD/dummy_multi/model2
        └── run.sh
```
Tag `MAD/dummy_multi` matches via direct child directory matching.

### Nested Directory Structure
```
scripts/
└── MAD/
    └── dummy/
        └── dummy_multi/
            ├── models.json      # Models: MAD/dummy/dummy_multi/model1
            └── run.sh
```
Tag `MAD/dummy_multi` matches via path component matching (new in this version).

### Nested Submodule Structure
```
scripts/
└── MAD/                         # MAD submodule
    └── scripts/
        └── dummy_multi/
            ├── models.json      # Models: MAD/dummy_multi/model1, MAD/dummy_multi/model2
            └── run.sh
```
Tag `MAD/dummy_multi` matches via direct child directory matching (interim scripts/ stripped).

### Multiple Nested Submodules
```
scripts/
└── Model-Repo1/                 # First submodule
    └── scripts/
        └── Model-Repo2/         # Second submodule (inside first)
            └── scripts/
                └── dummy/
                    ├── models.json  # Models: Model-Repo2/dummy/modelX
                    └── run.sh
```

Note: The directory immediately before the last "scripts" becomes part of the model name prefix.

### Multiple Locations with Same Component
```
scripts/
└── MAD/
    ├── dummy/
    │   └── dummy_multi/
    │       └── models.json      # Models: MAD/dummy/dummy_multi/model1
    └── other/
        └── dummy_multi/
            └── models.json      # Models: MAD/other/dummy_multi/model2
```
Tag `MAD/dummy_multi` matches BOTH locations via path component matching.

## Getting Help

If the debug script doesn't help you resolve the issue, include its full output when reporting the problem. This helps maintainers understand:
- What directory structure you have
- What models are being discovered
- Why tag matching is failing

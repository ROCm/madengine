# Performance Reporting Layer

**Status**: Active  
**Used by**: Modern `madengine` CLI

---

## 🎯 Purpose

Handles performance metrics collection, processing, and CSV output generation for model execution results.

---

## 📦 Components

### **`update_perf_csv.py`**

Updates performance CSV files with run results from both legacy and new CLI.

**Used by:**
- ✅ `execution/container_runner.py` (modern madengine CLI)

**Key Functions:**
```python
from madengine.reporting.update_perf_csv import update_perf_csv, flatten_tags

# Update CSV with new results
update_perf_csv(
    perf_json_path="results.json",
    output_csv="performance.csv"
)

# Flatten nested tags for CSV export
flattened = flatten_tags(perf_entry)
```

---

## 🗂️ Legacy Reporting Tools

The following legacy-only reporting tools remain in `tools/`:

| File | Purpose | Used By | Status |
|------|---------|---------|--------|
| `tools/csv_to_html.py` | Convert CSV to HTML | `mad.py`, `run_models.py` | Legacy only |
| `tools/csv_to_email.py` | Email CSV reports | `mad.py` | Legacy only |

These tools are **NOT** used by the modern `madengine` CLI.

---

## 📋 Architecture Decision

**Why is `update_perf_csv.py` in `reporting/` instead of `tools/`?**

1. ✅ **Shared across architectures**: Used by both legacy and new CLI
2. ✅ **Active development**: Not deprecated, actively maintained
3. ✅ **Clear responsibility**: Performance data processing
4. ✅ **Semantic clarity**: Reporting is a distinct concern

**Why are other CSV tools still in `tools/`?**

- They are **not used** by the modern `madengine` CLI
- Kept for backward compatibility only
- Will be deprecated when legacy CLI is retired

---

## 🔄 Usage Examples

### **New madengine** (via `container_runner.py`)

```python
from madengine.reporting.update_perf_csv import update_perf_csv

# After model execution completes
results_json = "/path/to/results.json"
output_csv = "/path/to/performance.csv"

update_perf_csv(results_json, output_csv)
```

### **Legacy madengine** (via `run_models.py` or `mad.py`)

```python
from madengine.reporting.update_perf_csv import UpdatePerfCsv

# Class-based interface (legacy)
updater = UpdatePerfCsv(args)
updater.run()
```

---

## 📊 Data Flow

```
Model Execution
    ↓
  Results JSON
    ↓
update_perf_csv()
    ↓
Performance CSV
    ↓
(Optional) CSV → HTML (legacy only)
(Optional) CSV → Email (legacy only)
```

---

## 🧪 Testing

```bash
# Test the reporting module
pytest tests/test_update_perf_csv.py -v

# Test integration with container runner
pytest tests/test_container_runner.py -v -k "perf"
```

---

## 🚀 Future Enhancements

Potential improvements (not currently planned):

- JSON output format (in addition to CSV)
- Parquet output for large datasets
- Real-time metrics streaming
- Integration with `database/` layer for direct ingestion

---

**Last Updated**: November 30, 2025  
**Maintainer**: madengine Team


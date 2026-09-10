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
    perf_csv="performance.csv",
    single_result="results.json",
)

# Flatten nested tags in place (no return value)
flatten_tags(perf_entry)  # mutates perf_entry in place
```

### **`update_perf_super.py`**

Maintains `perf_super.json`, a cumulative superset performance record that also
captures matched config data, and provides conversion to `perf_super.csv` /
`perf_entry_super.json` / `perf_entry_super.csv`.

**Used by:**
- ✅ `execution/container_runner.py` (modern madengine CLI)

### **`csv_to_html.py`**

Converts a single CSV file to an HTML table. Provides the `ConvertCsvToHtml`
handler class (`ConvertCsvToHtml.__init__(self, args: argparse.Namespace)`,
`.run(self) -> bool`) used by the CLI.

**Used by:**
- ✅ `cli/commands/report.py` (backs `madengine report to-html`)

### **`csv_to_email.py`**

Converts all CSV files in a directory into a single consolidated HTML report
suitable for emailing. Provides the `ConvertCsvToEmail` handler class
(`ConvertCsvToEmail.__init__(self, args: argparse.Namespace)`, `.run(self) -> bool`)
used by the CLI.

**Used by:**
- ✅ `cli/commands/report.py` (backs `madengine report to-email`)

---

## 🔄 Usage Examples

### **New madengine** (via `container_runner.py`)

```python
from madengine.reporting.update_perf_csv import update_perf_csv

# After model execution completes
perf_csv = "/path/to/performance.csv"
results_json = "/path/to/results.json"

update_perf_csv(perf_csv, single_result=results_json)
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

**Note:** As a side effect, `update_perf_csv()` (and the `handle_*_result()` helpers it
calls) also always write/append `perf_entry.csv` and `perf_entry.json` with the
latest result, regardless of the output file passed in.

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


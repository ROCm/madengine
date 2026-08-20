# Database Layer

**Status**: Active
**Purpose**: Upload CSV/JSON performance results to MongoDB

---

## 🎯 Responsibility

This module implements MongoDB ingestion for madengine results (e.g. `perf.csv`,
`perf_entry.csv`, or arbitrary JSON documents). It is a single, self-contained
file — `mongodb.py` — with no sub-packages. It handles:

- Auto-detecting file format (CSV vs JSON)
- Loading files with native type preservation (numbers, bools, nested
  JSON-in-CSV-cell strings)
- Transforming/normalizing documents (metadata stamping, numpy/pandas type
  cleanup)
- Auto-detecting unique fields for deduplication when not specified
- Bulk upload to MongoDB with batching, upsert, and automatic indexing

It is wired directly into the CLI via `madengine database` (see
`src/madengine/cli/commands/database.py`).

---

## 📦 Components (`mongodb.py`)

### Configuration

- **`MongoDBConfig`** — dataclass holding `host`, `port`, `username`,
  `password`, `auth_source`, `timeout_ms`. `MongoDBConfig.from_env()` builds
  one from `MONGO_HOST`, `MONGO_PORT`, `MONGO_USER`, `MONGO_PASSWORD`,
  `MONGO_AUTH_SOURCE`, `MONGO_TIMEOUT_MS`. The `.uri` property builds the
  `mongodb://` connection string.
- **`UploadOptions`** — dataclass controlling upload behavior:
  `unique_fields`, `upsert`, `batch_size`, `ordered`, `create_indexes`,
  `index_fields`, `add_metadata`, `metadata_prefix`, `validate_schema`
  (reserved field, currently unused by the implementation), `dry_run`.
- **`UploadResult`** — dataclass returned by every upload: `status`
  (`"success"` / `"partial"` / `"failed"`), `documents_read`,
  `documents_processed`, `documents_inserted`, `documents_updated`,
  `documents_failed`, `errors`, `duration_seconds`. Has a
  `print_summary()` method that renders a formatted Rich summary.

### File loading (Strategy pattern)

- **`DocumentLoader`** (ABC) — defines `load(file_path)` and
  `infer_schema(documents)`.
- **`JSONLoader`** — loads a JSON object or array of objects, preserving
  native types.
- **`CSVLoader`** — loads via `pandas.read_csv`, preserving native types and
  attempting to parse cell values that look like JSON (`{...}` / `[...]`)
  back into dicts/lists.
- **`detect_file_format(file_path)`** — picks `FileFormat.CSV` or
  `FileFormat.JSON` from the extension, falling back to content sniffing.
- **`get_loader(file_format)`** — returns the right loader instance.

### Transformation

- **`DocumentTransformer`** — takes `UploadOptions` and:
  - `transform(documents)` adds metadata (`_meta_uploaded_at`,
    `created_date`) and normalizes types (numpy scalars → Python, pandas
    `Timestamp` → `datetime`, `NaN` → `None`).
  - `infer_unique_fields(documents)` guesses a dedup key by checking
    candidate fields (`model`, `name`, `id`, `timestamp`, `date`,
    `pipeline`) for uniqueness across a sample of documents.

### Upload

- **`MongoDBUploader`** — connection + bulk-write class, usable as a context
  manager (`with MongoDBUploader(config) as uploader:`).
  - `connect()` / `disconnect()`
  - `upload(documents, database_name, collection_name, options)` →
    `UploadResult`. Creates indexes (if `options.create_indexes`) then does
    either a plain `insert_many` (when no `unique_fields`/`upsert`) or a
    batched `bulk_write` of `UpdateOne(..., upsert=True)` operations keyed
    on `unique_fields`.

### Entry points

- **`upload_file_to_mongodb(file_path, database_name, collection_name, config=None, options=None) -> UploadResult`**
  — the main entry point. Detects format, loads, auto-infers unique fields
  if not given, transforms, honors `dry_run` (returns without connecting to
  MongoDB), then uploads.
- **`upload_csv_to_mongodb(csv_file_path, database_name, collection_name, mongo_config=None) -> Dict[str, Any]`**
  — deprecated wrapper around `upload_file_to_mongodb` that returns a legacy
  dict shape instead of `UploadResult`.
- **`MongoDBHandler`** — deprecated class-based wrapper (`MongoDBHandler(args).run() -> bool`)
  kept for backward compatibility with old argparse-style call sites.

---

## 🔗 CLI mapping (`madengine database`)

`src/madengine/cli/commands/database.py` is a thin Typer wrapper around
`upload_file_to_mongodb`:

| CLI flag | Maps to |
|---|---|
| `--file` / `-f` | `file_path` |
| `--database` / `--db` | `database_name` |
| `--collection` / `-c` | `collection_name` |
| `--unique-key` / `-k` (comma-separated) | `UploadOptions.unique_fields` |
| `--batch-size` | `UploadOptions.batch_size` |
| `--no-upsert` | `UploadOptions.upsert = False` |
| `--no-index` | `UploadOptions.create_indexes = False` |
| `--dry-run` | `UploadOptions.dry_run` |

Connection config always comes from `MongoDBConfig.from_env()` (the CLI does
not expose host/port/credential flags — set `MONGO_HOST`, `MONGO_PORT`,
`MONGO_USER`, `MONGO_PASSWORD` instead). See `madengine database --help` or
`docs/cli-reference.md` for the full flag reference.

---

## 🚀 Usage

**Via the CLI:**

```bash
export MONGO_HOST=localhost
export MONGO_USER=admin
export MONGO_PASSWORD=secret

madengine database -f perf.csv --db madengine --collection results -k model,timestamp
madengine database -f perf_entry.json --db madengine --collection results --dry-run
```

**Via the Python API:**

```python
from madengine.database import upload_file_to_mongodb, MongoDBConfig, UploadOptions

result = upload_file_to_mongodb(
    file_path="perf.csv",
    database_name="madengine",
    collection_name="results",
    config=MongoDBConfig.from_env(),
    options=UploadOptions(unique_fields=["model", "timestamp"], batch_size=500),
)

result.print_summary()
print(result.status, result.documents_inserted, result.documents_updated)
```

`config` and `options` are both optional — omit them to use
`MongoDBConfig.from_env()` and default `UploadOptions()`.

---

## 📦 Difference from `db/` Package (Removed)

| Aspect | `db/` (Removed) | `database/` (Current) |
|--------|------------------|---------------------|
| **Purpose** | MySQL operations via SSH | MongoDB support |
| **Target** | Remote MySQL server | Local/distributed MongoDB |
| **Transport** | SSH tunnel | Direct connection |
| **Status** | **REMOVED** | Active |

MySQL support has been fully removed from madengine:

1. ✅ Removed `db/` package (MySQL operations)
2. ✅ Removed `tools/create_table_db.py` and `tools/update_table_db.py`
3. ✅ Removed `utils/ssh_to_db.py` (SSH to MySQL host)
4. ✅ Removed MySQL dependencies (`mysql-connector-python`, `pymysql`)

**Current state**: Only MongoDB support remains, via this `database/` package.

---

## 📚 References

- **Implementation**: `src/madengine/database/mongodb.py`
- **CLI command**: `src/madengine/cli/commands/database.py`, `madengine database --help`

---

**Last Updated**: 2026-08-05
**Maintainer**: madengine Team

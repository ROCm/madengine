# Database Layer (Future MongoDB Ingestion)

**Status**: Planned for future development  
**Purpose**: Modern data ingestion API for local and distributed deployments

---

## 🎯 Objective

This directory is reserved for a future unified database ingestion layer that will support:
- MongoDB data persistence
- Local result storage
- Distributed data collection from build and run phases
- Unified API for performance metrics ingestion

---

## 📋 Current State

⚠️ **Not yet implemented**. This directory is a placeholder for future development.

For current database operations, use the existing `db/` package which handles MySQL operations via SSH.

---

## 🗂️ Legacy Database Tools

The following legacy tools remain in `tools/` for backward compatibility:

| File | Purpose | Status |
|------|---------|--------|
| `tools/create_table_db.py` | MySQL table creation | Legacy (used by `mad.py`) |
| `tools/update_table_db.py` | MySQL table updates | Legacy (used by `mad.py`) |
| `tools/upload_mongodb.py` | MongoDB upload | Legacy (used by `mad.py`) |

---

## 🚀 Future Implementation Plan

When implemented, this layer will provide:

### **1. MongoDB Client** (`mongodb_client.py`)
```python
from madengine.database.mongodb_client import MongoDBClient

# Connect to local or remote MongoDB
client = MongoDBClient(connection_string="mongodb://localhost:27017")

# Ingest build results
client.ingest_build_results(build_manifest)

# Ingest run results
client.ingest_run_results(run_summary)
```

### **2. Local Storage** (`local_storage.py`)
```python
from madengine.database.local_storage import LocalStorage

# Store results locally (JSON, Parquet, etc.)
storage = LocalStorage(base_path="./madengine_results")
storage.save_results(results_dict)
```

### **3. Unified API** (`api.py`)
```python
from madengine.database import ingest_results

# Works with both local and distributed deployments
ingest_results(
    results=run_summary,
    target="mongodb",  # or "local", "mysql"
    config={"connection": "mongodb://..."}
)
```

---

## 📦 Difference from `db/` Package

| Aspect | `db/` (Existing) | `database/` (Future) |
|--------|------------------|---------------------|
| **Purpose** | MySQL operations via SSH | Modern MongoDB + local storage |
| **Target** | Remote MySQL server | Local/distributed MongoDB |
| **Transport** | SSH tunnel | Direct connection / API |
| **Status** | Active (until MySQL deprecated) | Planned |

---

## 🔄 Migration Path

When this layer is implemented, legacy tools will be deprecated:

1. ✅ **Phase 1**: Keep both `db/` and legacy `tools/` (current)
2. 🚧 **Phase 2**: Implement new `database/` layer
3. 📋 **Phase 3**: Migrate users to new API
4. 🗑️ **Phase 4**: Deprecate legacy MySQL tools

---

## 📚 References

- **Existing MySQL package**: `src/madengine/db/`
- **Legacy tools**: `src/madengine/tools/*_db.py`
- **Future tracking**: TBD (create GitHub issue when ready to implement)

---

**Last Updated**: November 30, 2025  
**Maintainer**: madengine Team


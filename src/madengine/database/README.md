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

## 🗂️ Legacy MySQL Tools (Removed)

**MySQL support has been removed from madengine**. The following tools are no longer available:

| File | Purpose | Status |
|------|---------|--------|
| ~~`tools/create_table_db.py`~~ | MySQL table creation | **REMOVED** |
| ~~`tools/update_table_db.py`~~ | MySQL table updates | **REMOVED** |
| ~~`db/` package~~ | MySQL operations via SSH | **REMOVED** |

For database operations, use MongoDB via the `database` command in the new CLI or legacy `mad.py`.

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

## 📦 Difference from `db/` Package (Removed)

| Aspect | `db/` (Removed) | `database/` (Current) |
|--------|------------------|---------------------|
| **Purpose** | MySQL operations via SSH | MongoDB support |
| **Target** | Remote MySQL server | Local/distributed MongoDB |
| **Transport** | SSH tunnel | Direct connection |
| **Status** | **REMOVED** | Active |

---

## 🔄 Migration Status

MySQL support has been fully removed from madengine:

1. ✅ **Phase 1**: Removed `db/` package (MySQL operations)
2. ✅ **Phase 2**: Removed `tools/create_table_db.py` and `tools/update_table_db.py`
3. ✅ **Phase 3**: Removed `utils/ssh_to_db.py` (SSH to MySQL host)
4. ✅ **Phase 4**: Removed MySQL dependencies (`mysql-connector-python`, `pymysql`)

**Current state**: Only MongoDB support remains via the `database/` package.

---

## 📚 References

- **MongoDB package**: `src/madengine/database/mongodb.py`
- **CLI database command**: `madengine database --help`

---

**Last Updated**: November 30, 2025  
**Maintainer**: madengine Team


"""
Schema migrations for the existing database (Step 4C-2f-2).

The project has no migration framework: tables are created with
Base.metadata.create_all(), which creates missing TABLES but never adds
columns to an existing table. ensure_schema() adds the columns introduced
after the first release, in place, with plain ALTER TABLE ... ADD COLUMN
(supported by SQLite and PostgreSQL). Existing rows keep NULL in the new
columns: their provenance is unknown and is not guessed.

Idempotent: a column that already exists is left alone.
"""

from sqlalchemy import inspect, text

# table -> [(column, SQL type)], in the order they were introduced
ADDED_COLUMNS = {
    "transactions": [
        ("model_set", "VARCHAR"),
        ("model_version", "VARCHAR"),
        ("reasons_json", "TEXT"),          # SHAP reasons stored with the score (trusted PDF reports)
    ],
}


def ensure_schema(engine) -> list[str]:
    """Adds any missing ADDED_COLUMNS. Returns the columns added ("table.column")."""
    added = []
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    for table, columns in ADDED_COLUMNS.items():
        if table not in tables:
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        for name, sql_type in columns:
            if name in existing:
                continue
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))
            added.append(f"{table}.{name}")
    return added

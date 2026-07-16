"""
Database setup.

Uses SQLite by default (zero setup, works immediately for development/demo).
To switch to PostgreSQL for production, just change DATABASE_URL, e.g.:
    DATABASE_URL = "postgresql://user:password@localhost:5432/fraud_platform"
and install psycopg2-binary (already in requirements.txt). No other code
needs to change because SQLAlchemy abstracts the dialect.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud_platform.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

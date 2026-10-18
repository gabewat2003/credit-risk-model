"""
src/build_features.py
Executes the SQL feature pipeline scripts against PostgreSQL.
"""

import os
from sqlalchemy import create_engine, text

DB_URI = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/lending"
)


def build_features():
    print("Connecting to database...")
    engine = create_engine(DB_URI)

    with engine.begin() as conn:
        print("Executing 01_build_target.sql...")
        with open("sql/01_build_target.sql", "r") as f:
            conn.execute(text(f.read()))

        print("Executing 02_features.sql...")
        with open("sql/02_features.sql", "r") as f:
            conn.execute(text(f.read()))

    print("Feature pipeline built successfully!")


if __name__ == "__main__":
    build_features()

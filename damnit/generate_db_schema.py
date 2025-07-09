import argparse
from pathlib import Path
import sys


DAMNIT_DB_SUBPATH = Path(__file__).parent / "backend" / "db"
SCHEMA_DIR_NAME = "schema"
MIGRATION_DIR_NAME = "migration"
START_VERSION = 1 # We start from v1 schema, as v0 is incompatible

def generate_schema(target_version: int, output_file: Path | None = None):
    """
    Generates a full SQLite schema for the target_version by applying diffs.

    Args:
        target_version: The desired final schema version (integer, e.g., 3).
        output_file: Optional path to save the generated schema. Prints to stdout if None.
    """
    if target_version < START_VERSION:
        print(f"Error: Target version must be {START_VERSION} or greater. v0 is not supported for this process.", file=sys.stderr)
        sys.exit(1)

    schema_base_dir = DAMNIT_DB_SUBPATH / SCHEMA_DIR_NAME
    migration_base_dir = DAMNIT_DB_SUBPATH / MIGRATION_DIR_NAME

    # 1. Start with the v1 schema
    initial_schema_file = schema_base_dir / f"v{START_VERSION}.sqlite3"
    if not initial_schema_file.is_file():
        print(f"Error: Initial schema file '{initial_schema_file}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Starting with base schema: {initial_schema_file.name}", file=sys.stderr)
    full_schema_content = initial_schema_file.read_text()
    full_schema_content += f"\n\n-- End of {initial_schema_file.name} --\n"

    # 2. Apply migration diffs iteratively
    current_version = START_VERSION
    while current_version < target_version:
        next_version = current_version + 1
        migration_file_name = f"v{current_version}_to_v{next_version}.sqlite3"
        migration_file_path = migration_base_dir / migration_file_name

        if not migration_file_path.is_file():
            print(f"Error: Migration script '{migration_file_path}' not found. "
                  f"Cannot proceed to version {next_version}.", file=sys.stderr)
            sys.exit(1)

        print(f"Applying migration: {migration_file_name}", file=sys.stderr)
        migration_content = migration_file_path.read_text()
        full_schema_content += f"\n\n-- Migration from v{current_version} to v{next_version} ({migration_file_name}) --\n"
        full_schema_content += migration_content
        full_schema_content += f"\n-- End of {migration_file_name} --\n"

        current_version = next_version

    if output_file:
        output_file.write_text(full_schema_content)
        print(f"\nGenerated schema for v{target_version} saved to '{output_file}'", file=sys.stderr)
    else:
        print("\n-- Generated Schema (stdout) --", file=sys.stderr)
        print(full_schema_content)

def main():
    parser = argparse.ArgumentParser(
        description=f"Generate a full SQLite schema for a target DAMNIT version (>= {START_VERSION}) "
                    "by applying incremental migration scripts to the v{START_VERSION} schema."
    )
    parser.add_argument(
        "target_version",
        type=int,
        help=f"The target schema version to generate (e.g., 2, 3). Must be >= {START_VERSION}."
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=None,
        help="Optional file path to save the generated schema. If not provided, prints to stdout."
    )

    args = parser.parse_args()

    generate_schema(args.target_version, args.output_file)

if __name__ == "__main__":
    main()

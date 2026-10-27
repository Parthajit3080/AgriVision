"""Safe operator commands for creating the first AgriVision administrator."""
from __future__ import annotations

import argparse
import getpass
import re
import sys

from . import auth_store as store


def create_initial_admin() -> int:
    store.app_secret()
    store.initialize_database()
    username = input("Admin ID (3–40 letters, digits, underscore, or dash): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", username):
        print("Admin ID must be 3–40 letters, digits, underscores, or dashes.", file=sys.stderr)
        return 2
    email = input("Admin email: ").strip()
    try:
        email = store.validate_email(email)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    display_name = input("Display name: ").strip()
    if not display_name or len(display_name) > 80:
        print("Display name must be between 1 and 80 characters.", file=sys.stderr)
        return 2
    password = getpass.getpass("Admin password: ")
    confirmation = getpass.getpass("Confirm admin password: ")
    if password != confirmation:
        print("Passwords do not match.", file=sys.stderr)
        return 2
    if not password or len(password) > 128 or len(password.encode("utf-8")) > 512:
        print("Admin password must not be empty and must be 128 characters or fewer.", file=sys.stderr)
        return 2

    now = store.timestamp()
    with store.db_session() as connection:
        duplicate = connection.execute(
            "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
            (username, email),
        ).fetchone()
        if duplicate:
            print("That username or email is already registered.", file=sys.stderr)
            return 2
        cursor = connection.execute(
            "INSERT INTO users(email, username, display_name, password_hash, role, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'superadmin', ?, ?)",
            (email, username, display_name, store.hash_password(password), now, now),
        )
        store.add_activity(connection, cursor.lastrowid, "initial_superadmin_created")
    print(f"Initial superadmin '{username}' created. Sign in at /admin/login.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.management")
    parser.add_argument("command", choices=["create-admin"], help="Create the initial superadmin interactively")
    args = parser.parse_args()
    if args.command == "create-admin":
        return create_initial_admin()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

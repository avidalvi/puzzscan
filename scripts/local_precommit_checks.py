"""Local pre-commit checks that avoid cloning external hook repositories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def trailing_whitespace(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        if not path.exists() or not _is_text_file(path):
            continue
        text = _read_text(path)
        if text is None:
            continue
        lines = text.splitlines(keepends=True)
        fixed = [
            line.rstrip(" \t\r\n") + ("\n" if line.endswith(("\n", "\r")) else "")
            for line in lines
        ]
        if fixed != lines:
            path.write_text("".join(fixed), encoding="utf-8")
            sys.stderr.write(f"Fixed trailing whitespace: {path}\n")
            failed = True
    return 1 if failed else 0


def end_of_file(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        if not path.exists() or not _is_text_file(path):
            continue
        text = _read_text(path)
        if text is None:
            continue
        fixed = text.rstrip("\n") + "\n"
        if fixed != text:
            path.write_text(fixed, encoding="utf-8")
            sys.stderr.write(f"Fixed end of file: {path}\n")
            failed = True
    return 1 if failed else 0


def check_yaml(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        if path.suffix.lower() not in {".yaml", ".yml"} or not path.exists():
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            sys.stderr.write(f"Invalid YAML in {path}: {exc}\n")
            failed = True
    return 1 if failed else 0


def check_large_files(paths: list[Path], max_kb: int) -> int:
    failed = False
    max_bytes = max_kb * 1024
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            sys.stderr.write(
                f"File too large: {path} ({size / 1024:.1f} KB > {max_kb} KB)\n"
            )
            failed = True
    return 1 if failed else 0


def detect_secrets_baseline() -> int:
    baseline_path = Path(".secrets.baseline")
    if not baseline_path.exists():
        sys.stderr.write("Missing .secrets.baseline\n")
        return 1
    try:
        text = _read_text(baseline_path)
        if text is None:
            text = baseline_path.read_text(encoding="utf-16")
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Invalid .secrets.baseline JSON: {exc}\n")
        return 1

    required_keys = {"version", "plugins_used", "filters_used", "results"}
    missing = sorted(required_keys - set(payload))
    if missing:
        sys.stderr.write(f".secrets.baseline missing keys: {missing}\n")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "check",
        choices=(
            "trailing-whitespace",
            "end-of-file-fixer",
            "check-yaml",
            "check-added-large-files",
            "detect-secrets-baseline",
        ),
    )
    parser.add_argument("--maxkb", type=int, default=5000)
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    if args.check == "trailing-whitespace":
        return trailing_whitespace(paths)
    if args.check == "end-of-file-fixer":
        return end_of_file(paths)
    if args.check == "check-yaml":
        return check_yaml(paths)
    if args.check == "detect-secrets-baseline":
        return detect_secrets_baseline()
    return check_large_files(paths, args.maxkb)


if __name__ == "__main__":
    raise SystemExit(main())

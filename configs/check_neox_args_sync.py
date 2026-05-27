#!/usr/bin/env python3
"""
Check whether configs/neox_arguments.md is in sync with argument source files.

This intentionally parses files statically instead of importing NeoX modules, so
it can run in lightweight environments without torch/deepspeed installed.
"""

import argparse
import ast
from pathlib import Path
import re
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = REPO_ROOT / "configs" / "neox_arguments.md"
DEFAULT_NEOX_PY = REPO_ROOT / "megatron" / "neox_arguments" / "neox_args.py"
DEFAULT_DEEPSPEED_PY = (
    REPO_ROOT / "megatron" / "neox_arguments" / "deepspeed_args.py"
)
DEFAULT_PY_FILES = (DEFAULT_NEOX_PY, DEFAULT_DEEPSPEED_PY)

SECTION_RE = re.compile(r"^##\s+(\S+)\s*$")
ARG_RE = re.compile(r"^-\s+\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*:")
TRIPLE_QUOTE_RE = re.compile(r'("""|\'\'\')(.*?)(\1)', re.DOTALL)


def normalize_description(description: str) -> str:
    return " ".join(textwrap.dedent(description).strip().split())


def extract_markdown_description(block):
    default_index = None
    for index, line in enumerate(block):
        if line.strip().startswith("Default ="):
            default_index = index
            break

    if default_index is None:
        return ""

    description_lines = block[default_index + 1 :]
    while description_lines and not description_lines[0].strip():
        description_lines.pop(0)
    while description_lines and not description_lines[-1].strip():
        description_lines.pop()

    return normalize_description("\n".join(description_lines))


def extract_markdown_default(block):
    for line in block:
        stripped = line.strip()
        if stripped.startswith("Default ="):
            return stripped.removeprefix("Default =").strip()
    return ""


def parse_markdown_args(path: Path, include_deepspeed: bool = False):
    """Return {arg_name: {line, section, default, description}} from md."""
    args = {}
    section = None
    lines = path.read_text().splitlines()

    for index, line in enumerate(lines):
        line_number = index + 1
        section_match = SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1)
            continue

        arg_match = ARG_RE.match(line)
        if not arg_match or section is None:
            continue

        if not include_deepspeed and section.startswith("NeoXArgsDeepspeed"):
            continue

        end = index + 1
        while end < len(lines):
            if SECTION_RE.match(lines[end]) or ARG_RE.match(lines[end]):
                break
            end += 1

        args[arg_match.group(1)] = {
            "line": line_number,
            "section": section,
            "default": extract_markdown_default(lines[index + 1 : end]),
            "description": extract_markdown_description(lines[index + 1 : end]),
        }

    return args


def is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
            return True
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "dataclass":
                return True
    return False


def parse_module_constants(tree: ast.Module):
    constants = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def format_python_default(value, constants):
    if value is None:
        return ""

    if isinstance(value, ast.Name) and value.id in constants:
        return str(constants[value.id])

    try:
        return str(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return ast.unparse(value)


def parse_python_args(path: Path):
    """Return {field_name: {line, section, default, description}} from py."""
    source = path.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    constants = parse_module_constants(tree)
    args = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not is_dataclass(node):
            continue

        fields = [
            stmt
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        for index, stmt in enumerate(fields):
            end_exclusive = (
                fields[index + 1].lineno - 1
                if index + 1 < len(fields)
                else getattr(node, "end_lineno", stmt.end_lineno)
            )
            field_source = "\n".join(source_lines[stmt.lineno - 1 : end_exclusive])
            doc_match = TRIPLE_QUOTE_RE.search(field_source)
            description = doc_match.group(2) if doc_match else ""
            args[stmt.target.id] = {
                "path": path,
                "line": stmt.lineno,
                "section": node.name,
                "default": format_python_default(stmt.value, constants),
                "description": normalize_description(description),
            }

    return args


def parse_python_files(paths):
    args = {}
    for path in paths:
        for name, metadata in parse_python_args(path).items():
            if name in args:
                first = args[name]
                raise ValueError(
                    f"duplicate Python arg {name!r}: "
                    f"{first['path']}:{first['line']} and {path}:{metadata['line']}"
                )
            args[name] = metadata
    return args


def print_group(title: str, rows):
    print(title)
    if not rows:
        print("  None")
        return

    for name, left, right in rows:
        print(f"  {name}")
        if left:
            print(f"    md: {left['section']}:{left['line']}")
        if right:
            print(f"    py: {right['path']}:{right['section']}:{right['line']}")


def print_description_mismatches(rows):
    print("Descriptions differ for common args:")
    if not rows:
        print("  None")
        return

    for name, md_arg, py_arg in rows:
        print(f"  {name}")
        print(f"    md: {md_arg['section']}:{md_arg['line']}")
        print(f"    py: {py_arg['path']}:{py_arg['section']}:{py_arg['line']}")
        print(f"    md description: {md_arg['description'] or '<empty>'}")
        print(f"    py description: {py_arg['description'] or '<empty>'}")


def display_value(value: str) -> str:
    if value == "":
        return "<empty>"
    if value != value.strip() or any(char in value for char in "\n\r\t"):
        return repr(value)
    return value


def print_default_mismatches(rows):
    print("Defaults differ for common args:")
    if not rows:
        print("  None")
        return

    for name, md_arg, py_arg in rows:
        print(f"  {name}")
        print(f"    md: {md_arg['section']}:{md_arg['line']}")
        print(f"    py: {py_arg['path']}:{py_arg['section']}:{py_arg['line']}")
        print(f"    md default: {display_value(md_arg['default'])}")
        print(f"    py default: {display_value(py_arg['default'])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check sync between configs/neox_arguments.md and Python argument "
            "source files."
        )
    )
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--py",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Python source file(s) to compare. Defaults to neox_args.py and "
            "deepspeed_args.py."
        ),
    )
    parser.add_argument(
        "--neox-only",
        action="store_true",
        help=(
            "Only compare neox_args.py and ignore NeoXArgsDeepspeed* markdown "
            "sections."
        ),
    )
    args = parser.parse_args()

    py_paths = args.py
    if py_paths is None:
        py_paths = (DEFAULT_NEOX_PY,) if args.neox_only else DEFAULT_PY_FILES

    md_args = parse_markdown_args(args.md, include_deepspeed=not args.neox_only)
    py_args = parse_python_files(py_paths)

    md_only = sorted(
        (name, md_args[name], None) for name in set(md_args) - set(py_args)
    )
    py_only = sorted(
        (name, None, py_args[name]) for name in set(py_args) - set(md_args)
    )
    description_mismatches = sorted(
        (name, md_args[name], py_args[name])
        for name in set(md_args) & set(py_args)
        if md_args[name]["description"] != py_args[name]["description"]
    )
    default_mismatches = sorted(
        (name, md_args[name], py_args[name])
        for name in set(md_args) & set(py_args)
        if md_args[name]["default"] != py_args[name]["default"]
    )

    print(f"Markdown args: {len(md_args)} ({args.md})")
    print(f"Python args:   {len(py_args)} ({', '.join(str(path) for path in py_paths)})")
    print()
    print_group("In markdown but not in Python sources:", md_only)
    print()
    print_group("In Python sources but not in markdown:", py_only)
    print()
    print_description_mismatches(description_mismatches)
    print()
    print_default_mismatches(default_mismatches)

    if md_only or py_only or description_mismatches or default_mismatches:
        return 1

    print()
    print("neox_arguments.md and Python argument sources are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

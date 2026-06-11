#!/usr/bin/env python3
"""
Check or sync configs/neox_arguments.md with argument source files.

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
IGNORED_ARGS = {"git_hash"}

SECTION_RE = re.compile(r"^##\s+(\S+)\s*$")
ARG_RE = re.compile(r"^-\s+\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*:\s*(.*)$")
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
            "type": arg_match.group(2).strip(),
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
        return format_literal_default(constants[value.id])

    try:
        literal_value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ast.unparse(value)
    return format_literal_default(literal_value)


def format_literal_default(value):
    formatted = str(value)
    if formatted != formatted.strip() or any(char in formatted for char in "\n\r\t"):
        return repr(value)
    return formatted


def format_python_type(value):
    formatted = ast.unparse(value)
    for name in ("Literal", "Union", "Optional", "List"):
        formatted = re.sub(rf"(?<![\w.]){name}\[", f"typing.{name}[", formatted)
    return formatted


def extract_python_description(field_source: str):
    doc_match = TRIPLE_QUOTE_RE.search(field_source)
    if not doc_match:
        return "", ""
    raw_description = textwrap.dedent(doc_match.group(2)).strip()
    return raw_description, normalize_description(raw_description)


def parse_python_sections(path: Path):
    """Return dataclass sections and argument metadata from a Python file."""
    source = path.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    constants = parse_module_constants(tree)
    sections = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not is_dataclass(node):
            continue

        fields = [
            stmt
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        section_args = []
        for index, stmt in enumerate(fields):
            end_exclusive = (
                fields[index + 1].lineno - 1
                if index + 1 < len(fields)
                else getattr(node, "end_lineno", stmt.end_lineno)
            )
            field_source = "\n".join(source_lines[stmt.lineno - 1 : end_exclusive])
            raw_description, description = extract_python_description(field_source)
            section_args.append(
                {
                    "name": stmt.target.id,
                    "path": path,
                    "line": stmt.lineno,
                    "section": node.name,
                    "type": format_python_type(stmt.annotation),
                    "default": format_python_default(stmt.value, constants),
                    "description": description,
                    "raw_description": raw_description,
                }
            )

        sections.append(
            {
                "name": node.name,
                "path": path,
                "line": node.lineno,
                "doc": ast.get_docstring(node, clean=True) or f"{node.name}()",
                "args": section_args,
            }
        )

    return sections


def parse_python_args(path: Path):
    """Return {field_name: {line, section, default, description}} from py."""
    args = {}
    for section in parse_python_sections(path):
        for arg in section["args"]:
            args[arg["name"]] = {
                "path": path,
                "line": arg["line"],
                "section": section["name"],
                "type": arg["type"],
                "default": arg["default"],
                "description": arg["description"],
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


def parse_python_section_files(paths):
    sections = []
    for path in paths:
        sections.extend(parse_python_sections(path))

    last_arg_by_name = {}
    for section in sections:
        for arg in section["args"]:
            last_arg_by_name[arg["name"]] = arg

    for section in sections:
        section["args"] = [
            arg for arg in section["args"] if last_arg_by_name[arg["name"]] is arg
        ]
    return sections


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


def print_type_mismatches(rows):
    print("Types differ for common args:")
    if not rows:
        print("  None")
        return

    for name, md_arg, py_arg in rows:
        print(f"  {name}")
        print(f"    md: {md_arg['section']}:{md_arg['line']}")
        print(f"    py: {py_arg['path']}:{py_arg['section']}:{py_arg['line']}")
        print(f"    md type: {display_value(md_arg['type'])}")
        print(f"    py type: {display_value(py_arg['type'])}")


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


def parse_markdown_sections(path: Path):
    lines = path.read_text().splitlines()
    section_starts = [
        (index, SECTION_RE.match(line).group(1))
        for index, line in enumerate(lines)
        if SECTION_RE.match(line)
    ]
    sections = []

    for section_index, (start, name) in enumerate(section_starts):
        end = (
            section_starts[section_index + 1][0]
            if section_index + 1 < len(section_starts)
            else len(lines)
        )
        args = []
        index = start + 1
        while index < end:
            arg_match = ARG_RE.match(lines[index])
            if not arg_match:
                index += 1
                continue

            arg_end = index + 1
            while arg_end < end:
                if ARG_RE.match(lines[arg_end]) or SECTION_RE.match(lines[arg_end]):
                    break
                arg_end += 1

            args.append(
                {
                    "name": arg_match.group(1),
                    "line": index + 1,
                    "section": name,
                    "type": arg_match.group(2).strip(),
                    "default": extract_markdown_default(lines[index + 1 : arg_end]),
                    "description": extract_markdown_description(
                        lines[index + 1 : arg_end]
                    ),
                    "start": index,
                    "end": arg_end,
                    "block": lines[index:arg_end],
                }
            )
            index = arg_end

        sections.append({"name": name, "start": start, "end": end, "args": args})

    return lines, sections


def render_arg_block(arg):
    lines = [
        f"- **{arg['name']}**: {arg['type']}",
        "",
        f"    Default = {arg['default']}",
        "",
    ]
    raw_description = arg.get("raw_description", "")
    if raw_description:
        lines.extend(
            f"    {line}" if line else "" for line in raw_description.splitlines()
        )
    else:
        lines.append("    ")
    lines.extend(["", ""])
    return lines


def render_section(section):
    lines = ["", "", f"## {section['name']}", "", section["doc"], "", ""]
    for arg in section["args"]:
        lines.extend(render_arg_block(arg))
    return lines


def should_include_section(section_name: str, include_deepspeed: bool):
    return include_deepspeed or not section_name.startswith("NeoXArgsDeepspeed")


def markdown_arg_matches_python(md_arg, py_arg):
    return (
        md_arg["type"] == py_arg["type"]
        and md_arg["default"] == py_arg["default"]
        and md_arg["description"] == py_arg["description"]
    )


def sync_markdown(path: Path, py_sections, include_deepspeed: bool):
    original = path.read_text()
    lines, md_sections = parse_markdown_sections(path)
    py_sections_by_name = {section["name"]: section for section in py_sections}
    output = []
    handled_sections = set()
    cursor = 0

    for md_section in md_sections:
        output.extend(lines[cursor : md_section["start"]])
        py_section = py_sections_by_name.get(md_section["name"])
        if py_section is None or not should_include_section(
            md_section["name"], include_deepspeed
        ):
            output.extend(lines[md_section["start"] : md_section["end"]])
            cursor = md_section["end"]
            continue

        handled_sections.add(md_section["name"])
        if md_section["args"]:
            prefix_end = md_section["args"][0]["start"]
            suffix_start = md_section["args"][-1]["end"]
            output.extend(lines[md_section["start"] : prefix_end])
        else:
            suffix_start = md_section["end"]
            output.extend(lines[md_section["start"] : md_section["end"]])

        md_args_by_name = {arg["name"]: arg for arg in md_section["args"]}
        for arg in py_section["args"]:
            md_arg = md_args_by_name.get(arg["name"])
            if md_arg and (
                arg["name"] in IGNORED_ARGS or markdown_arg_matches_python(md_arg, arg)
            ):
                output.extend(md_arg["block"])
            else:
                output.extend(render_arg_block(arg))

        output.extend(lines[suffix_start : md_section["end"]])
        cursor = md_section["end"]

    output.extend(lines[cursor:])
    for py_section in py_sections:
        if py_section["name"] in handled_sections or not should_include_section(
            py_section["name"], include_deepspeed
        ):
            continue
        output.extend(render_section(py_section))

    updated = "\n".join(output).rstrip() + "\n"
    if updated != original:
        path.write_text(updated)
        return True
    return False


def filter_ignored_args(md_args, py_args):
    md_args = dict(md_args)
    py_args = dict(py_args)
    for ignored_arg in IGNORED_ARGS:
        md_args.pop(ignored_arg, None)
        py_args.pop(ignored_arg, None)
    return md_args, py_args


def compare_args(md_args, py_args):
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
    type_mismatches = sorted(
        (name, md_args[name], py_args[name])
        for name in set(md_args) & set(py_args)
        if md_args[name]["type"] != py_args[name]["type"]
    )
    default_mismatches = sorted(
        (name, md_args[name], py_args[name])
        for name in set(md_args) & set(py_args)
        if md_args[name]["default"] != py_args[name]["default"]
    )
    return {
        "md_only": md_only,
        "py_only": py_only,
        "description_mismatches": description_mismatches,
        "type_mismatches": type_mismatches,
        "default_mismatches": default_mismatches,
    }


def has_drift(diff):
    return any(diff.values())


def print_report(md_args, py_args, py_paths, md_path, diff):
    print(f"Markdown args: {len(md_args)} ({md_path})")
    print(f"Python args:   {len(py_args)} ({', '.join(str(path) for path in py_paths)})")
    print()
    print_group("In markdown but not in Python sources:", diff["md_only"])
    print()
    print_group("In Python sources but not in markdown:", diff["py_only"])
    print()
    print_description_mismatches(diff["description_mismatches"])
    print()
    print_type_mismatches(diff["type_mismatches"])
    print()
    print_default_mismatches(diff["default_mismatches"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or sync configs/neox_arguments.md with Python argument "
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
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Update the markdown file from the Python argument source files.",
    )
    args = parser.parse_args()

    py_paths = args.py
    if py_paths is None:
        py_paths = (DEFAULT_NEOX_PY,) if args.neox_only else DEFAULT_PY_FILES

    include_deepspeed = not args.neox_only
    md_args = parse_markdown_args(args.md, include_deepspeed=include_deepspeed)
    py_args = parse_python_files(py_paths)
    md_args, py_args = filter_ignored_args(md_args, py_args)
    diff = compare_args(md_args, py_args)

    if args.sync:
        py_sections = parse_python_section_files(py_paths)
        if sync_markdown(args.md, py_sections, include_deepspeed):
            print(f"Synced {args.md} from Python argument sources.")
        else:
            print(f"{args.md} already matched the generated content.")
        md_args = parse_markdown_args(args.md, include_deepspeed=include_deepspeed)
        py_args = parse_python_files(py_paths)
        md_args, py_args = filter_ignored_args(md_args, py_args)
        diff = compare_args(md_args, py_args)

    print_report(md_args, py_args, py_paths, args.md, diff)

    if has_drift(diff):
        return 1

    print()
    print("neox_arguments.md and Python argument sources are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

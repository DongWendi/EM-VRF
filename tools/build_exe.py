"""Build EM-VRF Desktop.exe from main.py, or verify that an executable embeds main.py.

Requires Python 3.10 (the bundled runtime's version); no third-party packages.

    python tools/build_exe.py build TEMPLATE.exe OUTPUT.exe [--source main.py]
    python tools/build_exe.py verify "EM-VRF Desktop.exe" [--source main.py]

build:  TEMPLATE is any released EM-VRF Desktop.exe. Its PyInstaller bootloader, runtime
        options and PYZ library archive are copied unchanged; only the embedded 'main'
        script is replaced by main.py compiled with this Python. The PE checksum is updated.
        The output needs the release's _internal folder next to it.
verify: compiles main.py and checks that it is identical to the code object embedded in
        the executable, i.e. that the executable runs exactly this source file.
"""
import argparse
import array
import marshal
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyi_archive import CArchive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def pe_checksum(blob):
    offset = struct.unpack_from("<I", blob, 0x3C)[0] + 4 + 20 + 64
    data = bytearray(blob)
    data[offset:offset + 4] = b"\0\0\0\0"
    if len(data) % 2:
        data += b"\0"
    words = array.array("H")
    words.frombytes(bytes(data))
    total = sum(words)
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return offset, (total + len(blob)) & 0xFFFFFFFF


def compile_source(path):
    return compile(path.read_text(encoding="utf-8"), "main.py", "exec", dont_inherit=True)


def build(template, output, source):
    archive = CArchive(template)
    if archive.rebuild({}) != archive.blob:
        raise SystemExit("Template archive cannot be reproduced byte-for-byte; refusing to build.")
    blob = bytearray(archive.rebuild({"main": marshal.dumps(compile_source(source))}))
    offset, value = pe_checksum(blob)
    struct.pack_into("<I", blob, offset, value)
    output.write_bytes(bytes(blob))
    verify(output, source)


def verify(executable, source):
    archive = CArchive(executable)
    embedded = marshal.loads(archive.data(archive.find("main")))
    offset, value = pe_checksum(archive.blob)
    checksum_ok = struct.unpack_from("<I", archive.blob, offset)[0] == value
    matches = embedded == compile_source(source)
    print("%s embeds %s: %s; PE checksum valid: %s" % (executable.name, source.name, matches, checksum_ok))
    if not (matches and checksum_ok):
        raise SystemExit(1)


def main():
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Python 3.10 is required (the version of the bundled runtime).")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("template", type=Path)
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument("--source", type=Path, default=ROOT / "main.py")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("executable", type=Path)
    verify_parser.add_argument("--source", type=Path, default=ROOT / "main.py")
    args = parser.parse_args()
    if args.command == "build":
        build(args.template, args.output, args.source)
    else:
        verify(args.executable, args.source)


if __name__ == "__main__":
    main()

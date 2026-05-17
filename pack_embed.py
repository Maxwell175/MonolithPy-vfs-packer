"""Pack a MonolithPy install into a CMake-consumable embed bundle.

Output is a directory tree:

    <out>/
      MonolithPyEmbed.cmake     # `monolithpy_embed_link(<target>)` function
      include/                   # Python.h, mp_embed.h, staticinit.h, pybind11/
      lib/                       # all required .lib files on Windows, or one
                                 # aggregate libmonolithpy_bundle.a on POSIX.
      src/staticinit_stub.c      # bridges Py_BUILD_CORE-gated
                                 # Py_InitStaticModules to user TUs
      samples/                   # main.cpp + CMakeLists.txt smoke test

Usage:
    pack_embed.py [--install <monolithpy_dir>] [--out-dir <bundle_path>]
        Defaults: ./monolithpy314 and ./dist/embed_bundle/

    pack_embed.py --lipo <bundle_arm64> <bundle_x86_64> --out-dir <universal>
        macOS-only: lipos the two per-arch aggregate archives into one
        universal archive. Headers and the CMake module are copied from the
        arm64 input and normalized. Use after running pack_embed once on each
        per-arch monolithpy install.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


HERE = pathlib.Path(__file__).resolve().parent
POSIX_BUNDLE_ARCHIVE = "libmonolithpy_bundle.a"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--install",
        default=str(HERE / "monolithpy314"),
        help="MonolithPy install root (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        default=str(HERE / "dist" / "embed_bundle"),
        help="Output bundle directory (default: %(default)s)",
    )
    p.add_argument(
        "--pybind11",
        default=str(HERE / "third_party" / "pybind11"),
        help="pybind11 source checkout (header-only)",
    )
    p.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep the temporary build directory for debugging",
    )
    p.add_argument(
        "--lipo",
        nargs=2,
        metavar=("ARM64_BUNDLE", "X86_64_BUNDLE"),
        help="macOS only. Lipo two pre-built per-arch bundles into a "
             "universal bundle. --install is ignored in this mode.",
    )
    return p.parse_args()


# File extensions that exist on disk in a MonolithPy install but are dead
# weight inside the VFS: every static library is already linked into the
# final exe, and .link.json sidecars only matter at link time. Embedding
# them just inflates the binary.
_VFS_EXCLUDE_EXTS = {
    ".lib", ".obj", ".exp", ".pdb", ".pyd", ".dll",          # Windows
    ".a", ".o", ".so", ".dylib",                              # POSIX
}


_VFS_EXCLUDE_SUFFIXES = (".lib.link.json", ".lib.orig",
                         ".a.link.json", ".a.orig")
# Includes both spellings: the stdlib uses `test/` (singular, ~36 MB),
# numpy / scipy / setuptools etc. use `tests/` (plural).
_VFS_EXCLUDE_DIR_NAMES = {"tests", "test"}


def _vfs_ignore(src: str, names: list[str]) -> list[str]:
    """shutil.copytree filter: drop linker artifacts and per-package test
    fixture directories before they hit the VFS. The test directories
    (CPython's Lib/test/, numpy/tests/, scipy/tests/, etc.) carry hundreds
    of MB of fixtures not needed at runtime."""
    skipped = []
    for name in names:
        # `.lib.link.json` and `.lib.orig` don't match an .ext via splitext.
        if name.endswith(_VFS_EXCLUDE_SUFFIXES):
            skipped.append(name)
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in _VFS_EXCLUDE_EXTS:
            skipped.append(name)
            continue
        # Drop `test` / `tests` subdirectories (only when actually directories -
        # a stray file with that name would still come through).
        if name in _VFS_EXCLUDE_DIR_NAMES and os.path.isdir(os.path.join(src, name)):
            skipped.append(name)
    return skipped


def _file_hash(p: pathlib.Path) -> str:
    """sha256 of a file's contents. Used to dedup static libs that the
    upstream install duplicates across dependency_libs/<X>/lib trees."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_embed_tree(install: pathlib.Path, staging: pathlib.Path) -> None:
    """Materialize the tree we want mkembeddata to pack.

    Layout we want the VFS to answer:
      ~/lib/<stdlib>         (mapped from {install}/Lib/)
      ~/tcl/<tcl files>      (mapped from {install}/tcl/)
      /c/vfs/ssl/cert.pem    (copied straight through from {install}/Embedded/embed_data/)

    mkembeddata computes VFS paths as base_path-relative, lowercases, and
    rewrites any top-level '__relative__' segment to '~'. So:

      staging/
        __relative__/
          Lib/ -> copied from install/Lib
          tcl/ -> copied from install/tcl
        C/vfs/ssl/cert.pem -> copied from install/Embedded/embed_data/C/vfs/ssl/cert.pem

    Everything outside __relative__ lands under its absolute-path VFS slot.
    The 'C' directory here becomes '/c' at lookup time (matching Windows
    absolute-path normalization in get_virtual_path).

    Linker artifacts (.lib, .lib.link.json, .obj, .pdb, .exp, .pyd, .dll)
    are excluded - they're already linked into the final exe and would
    just bloat the embedded blob.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    relroot = staging / "__relative__"
    relroot.mkdir()

    # Copy the stdlib into the VFS under the ~/Lib/ namespace.
    # At runtime the embed binary must live at the install prefix so
    # its execfolder matches, making get_virtual_path rewrite prefix-
    # rooted paths like <prefix>/lib/python3.14/ to ~/lib/python3.14/
    # which hits these entries after lowercasing.
    src_lib = install / "Lib"
    if src_lib.is_dir():
        print(f"  staging Lib from {src_lib}")
        shutil.copytree(src_lib, relroot / "Lib", ignore=_vfs_ignore)

    # Copy tcl/ (needed for tkinter's Tcl init scripts, even if we don't use
    # them we should still make them embeddable for completeness).
    src_tcl = install / "tcl"
    if src_tcl.is_dir():
        print(f"  staging tcl from {src_tcl}")
        shutil.copytree(src_tcl, relroot / "tcl", ignore=_vfs_ignore)

    # SSL certs are stored under absolute path /c/vfs/ssl/cert.pem in the
    # prebuilt VFS - carry that forward so OpenSSL's hardcoded cert path
    # still hits an embedded file.
    src_absdata = install / "Embedded" / "embed_data"
    if src_absdata.is_dir():
        # Copy everything under embed_data/ as-is at the absolute path level.
        for sub in src_absdata.iterdir():
            dst = staging / sub.name
            if sub.is_dir():
                print(f"  staging absolute path tree {sub}")
                shutil.copytree(sub, dst, ignore=_vfs_ignore)
            else:
                shutil.copy2(sub, dst)


def _find_file(install: pathlib.Path, basename: str) -> pathlib.Path:
    """Search for a file under the install tree. The CI artifact layout may
    place files differently than a local build, so crawl if the obvious
    spot misses."""
    # Common locations in priority order.
    candidates = [
        install / basename,                            # root
        install / "bin" / basename,                    # POSIX bin/
        install / "Lib" / basename,                    # Windows Lib/
        install / "lib" / "python3.14" / basename,     # POSIX site-level
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Fall back to a glob (handles other python versions / custom layouts).
    for f in install.rglob(basename):
        if f.is_file():
            return f
    raise SystemExit(f"{basename} not found under {install}")


def _find_python(install: pathlib.Path) -> pathlib.Path:
    """Locate the monolithpy python executable."""
    # The CI artifact uses bin/python3.14; local dev builds may have python.exe
    # at the root.
    for name in ("python3.14", "python3", "python.exe"):
        try:
            return _find_file(install, name)
        except SystemExit:
            pass
    raise SystemExit(f"no python binary found under {install}")


def run_mkembeddata(install: pathlib.Path, staging: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Invoke the monolithpy python's own mkembeddata.py on the staging tree."""
    script = _find_file(install, "mkembeddata.py")
    py = _find_python(install)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  running mkembeddata ({py.name}) over {staging}")
    subprocess.check_call([str(py), str(script), str(out_dir), str(staging)])
    c_file = out_dir / "mp_embed_data.c"
    if not c_file.is_file():
        raise SystemExit("mkembeddata did not produce mp_embed_data.c")
    return c_file


def write_embed_coff(out_obj: pathlib.Path,
                     map_dat: pathlib.Path,
                     data_dat: pathlib.Path,
                     arch: str = "x86_64") -> None:
    """Hand-emit a COFF .obj that exports the four symbols mp_embed.c
    needs: nuitka_embed_map, nuitka_embed_data, nuitka_embed_map_len,
    nuitka_embed_data_len. Three .rdata sections - one per blob plus one for
    the two 32-bit lengths - keep raw-data layout simple and let the linker
    place them independently.

    Why not cl.exe? mkembeddata's mp_embed_data.c declares
    `const unsigned char nuitka_embed_data[] = { 0x.., 0x.., ... };` with
    one entry per byte. For a ~1 GB blob that's ~5 GB of preprocessor input,
    and MSVC's parser+LTCG holds the entire literal in memory (13+ GB).
    Why not assembler? Avoids requiring clang or MASM as a build dep -
    pure Python writes a valid COFF directly.

    PE/COFF spec used: Microsoft PE/COFF Specification rev 11.
    """
    import struct

    # ---- COFF format constants ----
    _coff_machines = {
        "x86_64":  0x8664,   # IMAGE_FILE_MACHINE_AMD64
        "amd64":   0x8664,
        "arm64":   0xAA64,   # IMAGE_FILE_MACHINE_ARM64
        "aarch64": 0xAA64,
    }
    machine = _coff_machines.get(arch.lower())
    if machine is None:
        raise ValueError(f"unsupported COFF arch: {arch!r}")
    IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
    IMAGE_SCN_ALIGN_8BYTES         = 0x00400000
    IMAGE_SCN_MEM_READ             = 0x40000000
    RDATA_FLAGS = (IMAGE_SCN_CNT_INITIALIZED_DATA |
                   IMAGE_SCN_ALIGN_8BYTES |
                   IMAGE_SCN_MEM_READ)
    IMAGE_SYM_CLASS_EXTERNAL = 2

    map_size = map_dat.stat().st_size
    data_size = data_dat.stat().st_size
    lens_blob = struct.pack("<II", map_size, data_size)  # two u32s

    # ---- Layout offsets ----
    file_header_size = 20
    section_header_size = 40
    num_sections = 3
    headers_size = file_header_size + num_sections * section_header_size

    sec1_off = headers_size                    # nuitka_embed_map raw data
    sec2_off = sec1_off + map_size             # nuitka_embed_data raw data
    sec3_off = sec2_off + data_size            # the two _len globals (8 bytes)
    symtab_off = sec3_off + len(lens_blob)

    # 4 symbols, 18 bytes each; no aux records.
    sym_size = 18
    num_syms = 4

    # All four symbol names exceed 8 chars - they go in the string table.
    # String table layout: 4-byte total size (incl the size field itself),
    # followed by null-terminated names. Each symbol's name field stores the
    # offset (in the table, including the leading 4-byte size field).
    sym_names = [
        b"nuitka_embed_map",
        b"nuitka_embed_data",
        b"nuitka_embed_map_len",
        b"nuitka_embed_data_len",
    ]
    string_table = bytearray(b"\x00\x00\x00\x00")  # placeholder for length
    name_offsets = []
    for n in sym_names:
        name_offsets.append(len(string_table))
        string_table += n + b"\x00"
    # Patch the size field
    struct.pack_into("<I", string_table, 0, len(string_table))

    # ---- Stream-write the .obj ----
    with open(out_obj, "wb") as out:
        # File header
        out.write(struct.pack(
            "<HHIIIHH",
            machine,
            num_sections,
            0,                  # TimeDateStamp
            symtab_off,
            num_syms,
            0,                  # SizeOfOptionalHeader
            0,                  # Characteristics
        ))

        # Section headers (3 x .rdata$mpN). The $-suffix groups them into the
        # final .rdata section, sorted alphabetically by suffix - this is the
        # standard COFF mechanism (see PE/COFF spec on `grouped sections`).
        # Using a unique base name per section avoids any ambiguity vs. having
        # three sections that all share the bare name `.rdata`.
        def section_hdr(name8: bytes, size_raw, ptr_raw):
            return struct.pack(
                "<8sIIIIIIHHI",
                name8,               # 8-byte name field
                0,                   # VirtualSize (0 for object files)
                0,                   # VirtualAddress (0 for object files)
                size_raw,            # SizeOfRawData
                ptr_raw,             # PointerToRawData
                0,                   # PointerToRelocations
                0,                   # PointerToLinenumbers
                0,                   # NumberOfRelocations
                0,                   # NumberOfLinenumbers
                RDATA_FLAGS,
            )
        out.write(section_hdr(b".rdata$a", map_size, sec1_off))
        out.write(section_hdr(b".rdata$b", data_size, sec2_off))
        out.write(section_hdr(b".rdata$c", len(lens_blob), sec3_off))

        # Section 1 raw bytes: copy map.dat (small, ~hundreds of KB).
        with open(map_dat, "rb") as f:
            shutil.copyfileobj(f, out, length=1 << 20)
        # Section 2 raw bytes: copy data.dat (large, can be GB).
        with open(data_dat, "rb") as f:
            shutil.copyfileobj(f, out, length=1 << 20)
        # Section 3 raw bytes: the two _len constants.
        out.write(lens_blob)

        # Symbol table. Each symbol's section number is 1-based.
        # Name field uses the long-name form: first 4 bytes zero, next 4 bytes
        # are the offset into the string table.
        symbol_specs = [
            (0, 1, name_offsets[0]),  # nuitka_embed_map      offset 0 in sec 1
            (0, 2, name_offsets[1]),  # nuitka_embed_data     offset 0 in sec 2
            (0, 3, name_offsets[2]),  # nuitka_embed_map_len  offset 0 in sec 3
            (4, 3, name_offsets[3]),  # nuitka_embed_data_len offset 4 in sec 3
        ]
        for value, sec_num, name_off in symbol_specs:
            out.write(struct.pack(
                "<IIIhHBB",
                0,                 # Name.Zeroes (signals long form)
                name_off,          # Name.Offset into string table
                value,             # Value (offset within section)
                sec_num,           # SectionNumber (1-based)
                0,                 # Type
                IMAGE_SYM_CLASS_EXTERNAL,
                0,                 # NumberOfAuxSymbols
            ))

        # String table
        out.write(bytes(string_table))


def write_embed_elf(out_obj: pathlib.Path,
                    map_dat: pathlib.Path,
                    data_dat: pathlib.Path,
                    arch: str = "x86_64") -> None:
    """Hand-emit an ELF64 .o that exposes the four nuitka_embed_* globals.

    Single .rodata section holds map | pad | data | pad | (map_len, data_len),
    with each blob 8-byte aligned. Four global SHT_PROGBITS / STT_OBJECT
    symbols point into it at the right offsets. SHN_UNDEF (index 0) is the
    only local symbol so .symtab.sh_info = 1.

    ELF64 reference: System V ABI, gABI rev 1.5.
    """
    import struct

    ELFCLASS64 = 2
    ELFDATA2LSB = 1
    EV_CURRENT = 1
    ELFOSABI_NONE = 0
    ET_REL = 1
    EM_X86_64 = 62
    EM_AARCH64 = 183
    SHT_NULL = 0
    SHT_PROGBITS = 1
    SHT_SYMTAB = 2
    SHT_STRTAB = 3
    SHF_ALLOC = 0x2
    STB_GLOBAL = 1
    STT_OBJECT = 1

    machines = {
        "x86_64": EM_X86_64, "amd64": EM_X86_64,
        "aarch64": EM_AARCH64, "arm64": EM_AARCH64,
    }
    e_machine = machines.get(arch.lower())
    if e_machine is None:
        raise ValueError(f"unsupported ELF arch: {arch!r}")

    map_size = map_dat.stat().st_size
    data_size = data_dat.stat().st_size
    lens_blob = struct.pack("<II", map_size, data_size)

    def _pad8(n: int) -> int:
        return (8 - (n % 8)) % 8

    map_off = 0
    map_pad = _pad8(map_size)
    data_off = map_off + map_size + map_pad
    data_pad = _pad8(data_size)
    lens_off = data_off + data_size + data_pad
    rodata_size = lens_off + len(lens_blob)

    # .strtab (symbol names): leading null byte means "no name".
    strtab = bytearray(b"\x00")
    sym_names = [b"nuitka_embed_map", b"nuitka_embed_data",
                 b"nuitka_embed_map_len", b"nuitka_embed_data_len"]
    name_offsets = []
    for n in sym_names:
        name_offsets.append(len(strtab))
        strtab += n + b"\x00"

    # .shstrtab (section names).
    shstrtab = bytearray(b"\x00")
    shstr_offsets = {}
    for sname in (".rodata", ".symtab", ".strtab", ".shstrtab"):
        shstr_offsets[sname] = len(shstrtab)
        shstrtab += sname.encode() + b"\x00"

    EHDR_SIZE = 64
    SHDR_SIZE = 64
    SYM_SIZE = 24
    num_syms = 1 + len(sym_names)        # null sym + 4 globals
    num_sections = 5                      # null, .rodata, .symtab, .strtab, .shstrtab
    e_shstrndx = 4

    rodata_file_off = EHDR_SIZE           # 64 - already 8-aligned
    symtab_file_off = rodata_file_off + rodata_size
    symtab_file_off += _pad8(symtab_file_off)
    symtab_size = num_syms * SYM_SIZE
    strtab_file_off = symtab_file_off + symtab_size
    shstrtab_file_off = strtab_file_off + len(strtab)
    shdrs_file_off = shstrtab_file_off + len(shstrtab)
    shdrs_file_off += _pad8(shdrs_file_off)

    with open(out_obj, "wb") as out:
        # e_ident[16]
        out.write(bytes([
            0x7F, ord('E'), ord('L'), ord('F'),
            ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_NONE,
            0, 0, 0, 0, 0, 0, 0, 0,
        ]))
        # rest of Elf64_Ehdr (48 bytes)
        out.write(struct.pack(
            "<HHIQQQIHHHHHH",
            ET_REL, e_machine, EV_CURRENT,
            0,                  # e_entry
            0,                  # e_phoff
            shdrs_file_off,     # e_shoff
            0,                  # e_flags
            EHDR_SIZE,          # e_ehsize
            0, 0,               # e_phentsize, e_phnum
            SHDR_SIZE, num_sections, e_shstrndx,
        ))

        # Pad to .rodata offset (no-op, EHDR is exactly 64 bytes).
        out.write(b"\x00" * (rodata_file_off - out.tell()))

        # .rodata raw: map.dat | pad | data.dat | pad | lens
        with open(map_dat, "rb") as f:
            shutil.copyfileobj(f, out, length=1 << 20)
        out.write(b"\x00" * map_pad)
        with open(data_dat, "rb") as f:
            shutil.copyfileobj(f, out, length=1 << 20)
        out.write(b"\x00" * data_pad)
        out.write(lens_blob)

        # Pad up to .symtab.
        out.write(b"\x00" * (symtab_file_off - out.tell()))

        # .symtab: null sym + 4 globals.
        # Elf64_Sym: name(4), info(1), other(1), shndx(2), value(8), size(8)
        out.write(b"\x00" * SYM_SIZE)
        st_info = (STB_GLOBAL << 4) | STT_OBJECT
        rodata_idx = 1
        sym_specs = [
            (name_offsets[0], rodata_idx, map_off,      map_size),
            (name_offsets[1], rodata_idx, data_off,     data_size),
            (name_offsets[2], rodata_idx, lens_off,     4),
            (name_offsets[3], rodata_idx, lens_off + 4, 4),
        ]
        for st_name, st_shndx, st_value, st_size in sym_specs:
            out.write(struct.pack("<IBBHQQ",
                                  st_name, st_info, 0,
                                  st_shndx, st_value, st_size))

        out.write(bytes(strtab))
        out.write(bytes(shstrtab))

        # Pad to section header table.
        out.write(b"\x00" * (shdrs_file_off - out.tell()))

        # Section headers. Elf64_Shdr is 64 bytes:
        #   name(4), type(4), flags(8), addr(8), offset(8),
        #   size(8), link(4), info(4), addralign(8), entsize(8)
        def shdr(name, type_, flags, off, size, link=0, info=0,
                 align=1, entsize=0):
            return struct.pack("<IIQQQQIIQQ",
                               name, type_, flags, 0, off, size,
                               link, info, align, entsize)

        out.write(shdr(0, SHT_NULL, 0, 0, 0))
        out.write(shdr(shstr_offsets[".rodata"], SHT_PROGBITS, SHF_ALLOC,
                       rodata_file_off, rodata_size, align=8))
        out.write(shdr(shstr_offsets[".symtab"], SHT_SYMTAB, 0,
                       symtab_file_off, symtab_size,
                       link=3, info=1, align=8, entsize=SYM_SIZE))
        out.write(shdr(shstr_offsets[".strtab"], SHT_STRTAB, 0,
                       strtab_file_off, len(strtab)))
        out.write(shdr(shstr_offsets[".shstrtab"], SHT_STRTAB, 0,
                       shstrtab_file_off, len(shstrtab)))


def write_embed_macho(out_obj: pathlib.Path,
                      map_dat: pathlib.Path,
                      data_dat: pathlib.Path,
                      arch: str = "x86_64") -> None:
    """Hand-emit a Mach-O 64 object file with the four nuitka_embed_* globals.

    One LC_SEGMENT_64 (anonymous segname) carrying a single __DATA,__const
    section that holds map | pad | data | pad | (map_len, data_len). One
    LC_SYMTAB with the four globals (Mach-O prepends `_` to C names).
    """
    import struct

    MH_MAGIC_64 = 0xFEEDFACF
    MH_OBJECT = 1
    LC_SEGMENT_64 = 0x19
    LC_SYMTAB = 0x2
    N_EXT = 0x01
    N_SECT = 0x0E
    S_REGULAR = 0x0

    cpus = {
        "x86_64": (0x01000007, 3),
        "amd64":  (0x01000007, 3),
        "arm64":  (0x0100000C, 0),
        "aarch64":(0x0100000C, 0),
    }
    cputype, cpusubtype = cpus.get(arch.lower(), (None, None))
    if cputype is None:
        raise ValueError(f"unsupported Mach-O arch: {arch!r}")

    map_size = map_dat.stat().st_size
    data_size = data_dat.stat().st_size
    lens_blob = struct.pack("<II", map_size, data_size)

    def _pad8(n: int) -> int:
        return (8 - (n % 8)) % 8

    map_off = 0
    map_pad = _pad8(map_size)
    data_off = map_off + map_size + map_pad
    data_pad = _pad8(data_size)
    lens_off = data_off + data_size + data_pad
    sect_size = lens_off + len(lens_blob)

    # macOS prepends `_` to C symbol names in the object's symtab.
    sym_names = [b"_nuitka_embed_map", b"_nuitka_embed_data",
                 b"_nuitka_embed_map_len", b"_nuitka_embed_data_len"]
    sym_offsets = [map_off, data_off, lens_off, lens_off + 4]

    strtab = bytearray(b"\x00")
    name_offsets = []
    for n in sym_names:
        name_offsets.append(len(strtab))
        strtab += n + b"\x00"
    while len(strtab) % 8 != 0:
        strtab += b"\x00"

    HDR_SIZE = 32
    SEG_CMD_BASE = 72
    SECT_SIZE = 80
    SYMTAB_CMD_SIZE = 24
    NLIST_SIZE = 16

    nsects = 1
    seg_cmdsize = SEG_CMD_BASE + nsects * SECT_SIZE
    sizeofcmds = seg_cmdsize + SYMTAB_CMD_SIZE

    cmds_off = HDR_SIZE
    raw_off = cmds_off + sizeofcmds
    raw_off += _pad8(raw_off)
    symoff = raw_off + sect_size
    symoff += _pad8(symoff)
    nsyms = len(sym_names)
    symtab_size = nsyms * NLIST_SIZE
    stroff = symoff + symtab_size
    strsize = len(strtab)

    with open(out_obj, "wb") as out:
        # mach_header_64
        out.write(struct.pack("<IIIIIIII",
                              MH_MAGIC_64, cputype, cpusubtype, MH_OBJECT,
                              2,            # ncmds: SEGMENT_64 + SYMTAB
                              sizeofcmds, 0, 0))

        # LC_SEGMENT_64 with empty segname (object-file convention).
        out.write(struct.pack("<II16sQQQQIIII",
                              LC_SEGMENT_64, seg_cmdsize,
                              b"",          # segname
                              0,            # vmaddr
                              sect_size,    # vmsize
                              raw_off,      # fileoff
                              sect_size,    # filesize
                              7, 7,         # maxprot, initprot
                              nsects, 0))

        # section_64 (sectname[16], segname[16], addr, size, offset, align,
        #             reloff, nreloc, flags, reserved1/2/3)
        out.write(struct.pack("<16s16sQQIIIIIIII",
                              b"__const", b"__DATA",
                              0, sect_size, raw_off,
                              3,            # align: log2(8) = 3
                              0, 0, S_REGULAR, 0, 0, 0))

        # LC_SYMTAB
        out.write(struct.pack("<IIIIII",
                              LC_SYMTAB, SYMTAB_CMD_SIZE,
                              symoff, nsyms, stroff, strsize))

        # Pad to raw section data.
        out.write(b"\x00" * (raw_off - out.tell()))

        with open(map_dat, "rb") as f:
            shutil.copyfileobj(f, out, length=1 << 20)
        out.write(b"\x00" * map_pad)
        with open(data_dat, "rb") as f:
            shutil.copyfileobj(f, out, length=1 << 20)
        out.write(b"\x00" * data_pad)
        out.write(lens_blob)

        # Pad to symbol table.
        out.write(b"\x00" * (symoff - out.tell()))

        # nlist_64: n_strx(4), n_type(1), n_sect(1), n_desc(2), n_value(8)
        n_type = N_EXT | N_SECT
        for st_name_off, sym_value in zip(name_offsets, sym_offsets):
            out.write(struct.pack("<IBBHQ",
                                  st_name_off, n_type, 1, 0, sym_value))

        out.write(bytes(strtab))


def _detect_target_arch(install: pathlib.Path) -> str:
    """Detect the TARGET architecture from the MonolithPy install's python
    binary, NOT the host architecture. This is critical for cross-arch
    packing (e.g. x86_64 artifact on an arm64 Mac)."""
    py = _find_python(install)
    import struct
    data = py.read_bytes()
    if len(data) < 4:
        raise SystemExit(f"cannot read python binary: {py}")
    magic = struct.unpack_from("<I", data, 0)[0]
    off = 4
    # Mach-O FAT headers are always stored big-endian: real bytes are
    # CA FE BA BE (32-bit) or CA FE BA BF (64-bit). When the leading 4
    # bytes are little-endian-unpacked, they appear as the CIGAM constants
    # 0xBEBAFECA / 0xBFBAFECA; the rest of the FAT header is still BE.
    if magic in (0xBEBAFECA, 0xBFBAFECA):  # Mach-O FAT (32/64-bit)
        off += 4  # skip narch
        cputype = struct.unpack_from(">I", data, off)[0]
    elif magic == 0xFEEDFACF:  # MH_MAGIC_64
        cputype = struct.unpack_from("<I", data, 4)[0]
    elif magic == 0xFEEDFACE:  # MH_MAGIC
        cputype = struct.unpack_from("<I", data, 4)[0]
    elif magic == 0x7F454C46:  # ELF
        e_machine = struct.unpack_from("<H", data, 18)[0]
        return {62: "x86_64", 183: "aarch64"}.get(e_machine, f"elf_machine_{e_machine}")
    else:
        import platform
        return {
            "amd64": "x86_64", "x86_64": "x86_64",
            "arm64": "arm64", "aarch64": "aarch64",
        }.get(platform.machine().lower(), platform.machine().lower())
    return {0x01000007: "x86_64", 0x0100000C: "arm64"}.get(
        cputype, f"cputype_{cputype}")


def _host_object_format() -> str:
    """Pick the object-file format for the current host."""
    if sys.platform == "win32":
        return "coff"
    if sys.platform == "darwin":
        return "macho"
    return "elf"


def write_embed_object(out_obj: pathlib.Path,
                       map_dat: pathlib.Path,
                       data_dat: pathlib.Path,
                       fmt: str | None = None,
                       arch: str | None = None) -> None:
    """Pick the host's native object format and emit the four nuitka_embed_*
    symbols. fmt/arch override host detection for cross-emission."""
    fmt = fmt or _host_object_format()
    if arch is None:
        import platform
        arch = platform.machine().lower()
        if arch in ("amd64", "x86_64"):
            arch = "x86_64"
        elif arch in ("arm64", "aarch64"):
            arch = "arm64" if sys.platform == "darwin" else "aarch64"
    if fmt == "coff":
        write_embed_coff(out_obj, map_dat, data_dat, arch)
    elif fmt == "elf":
        write_embed_elf(out_obj, map_dat, data_dat, arch)
    elif fmt == "macho":
        write_embed_macho(out_obj, map_dat, data_dat, arch)
    else:
        raise ValueError(f"unsupported object format: {fmt!r}")


def write_staticinit_stub(path: pathlib.Path) -> None:
    """Pulls Py_InitStaticModules from staticinit.h into an external symbol.

    staticinit.h's definition is `static inline`, gated behind Py_BUILD_CORE,
    so it's inaccessible from our pybind11 C++ TU. We compile this tiny shim
    as C with Py_BUILD_CORE defined, include staticinit.h, and then emit a
    non-inline wrapper with external linkage that the main can call.
    """
    path.write_text(
        """/* Bridge between Py_BUILD_CORE-gated Py_InitStaticModules and our main. */
#define Py_BUILD_CORE 1
#include <Python.h>
#include <staticinit.h>

/* Py_InitStaticModules in staticinit.h is static inline; forward it through
   a proper external symbol so C++ code that includes Python.h normally can
   link against it. */
void mp_init_static_modules(void) {
    Py_InitStaticModules();
}
""")


def resolve_libs(link_json: dict, install: pathlib.Path) -> tuple[list[str], list[str], list[str]]:
    """Rebase link.json's absolute paths onto the local install.

    CI ships link.json with paths rooted at the build's `<...>/output/`
    dir (e.g. `D:\\a\\MonolithPy\\MonolithPy\\output\\...` on Windows or
    `/home/runner/work/MonolithPy/MonolithPy/output/...` on POSIX). We
    mirror that whole subtree as `<install>` locally, so peeling the
    `output/` prefix and reattaching `<install>` recovers the local path.

    Bare lib names (no separators) and already-existing local paths pass
    through unchanged.
    """
    libs_in = link_json.get("libraries", [])
    dirs_in = link_json.get("library_dirs", [])
    flags = list(link_json.get("link_flags", []))

    def rebase(p: str) -> str:
        if not p or ("\\" not in p and "/" not in p):
            return p  # bare lib name (e.g. 'kernel32', 'pthread')
        if os.path.isabs(p) and os.path.exists(p):
            return p  # already-local; don't second-guess
        norm = p.replace("\\", "/")
        idx = norm.lower().find("/output/")
        if idx == -1:
            return p
        rel = norm[idx + len("/output/"):]
        # Reattach using install's native separator.
        return str(install.joinpath(*rel.split("/")))

    libs = []
    seen = set()
    for lib in libs_in:
        rebased = rebase(lib)
        # Drop duplicates (a path can appear multiple times in link.json).
        if rebased in seen:
            continue
        seen.add(rebased)
        libs.append(rebased)

    # Also keep the /FORCE etc. link_flags verbatim - but strip any .res that
    # doesn't exist locally. The tk_base.res is a resource only useful if you
    # embed Tk UI; we can omit silently if missing.
    cleaned_flags = []
    for f in flags:
        if f.lower().endswith(".res"):
            rebased = rebase(f)
            if os.path.isfile(rebased):
                cleaned_flags.append(rebased)
            continue
        cleaned_flags.append(f)

    lib_dirs = []
    seen_dirs = set()
    for d in dirs_in:
        rebased = rebase(d)
        if rebased in seen_dirs:
            continue
        seen_dirs.add(rebased)
        if os.path.isdir(rebased):
            lib_dirs.append(rebased)

    return libs, lib_dirs, cleaned_flags


def _ensure_link_json(install: pathlib.Path) -> None:
    """If link.json is absent, generate it by running rebuildpython.

    A pristine MonolithPy install (straight from CI) lacks link.json until
    the first pip-installed package triggers the rebuild hook. The packer
    needs it to discover which static libs the interpreter links against.
    """
    link_json = install / "link.json"
    if link_json.is_file():
        return
    py = _find_python(install)
    print("  link.json missing; running rebuildpython to generate it")
    subprocess.check_call([str(py), "-m", "rebuildpython"])


def build_bundle(install: pathlib.Path, build: pathlib.Path,
                 out_dir: pathlib.Path, pybind11_root: pathlib.Path) -> None:
    """Assemble the bundle directory: copy headers + libs into place, build
    mp_embed_data.lib (the generated VFS blob), write the CMake module + the
    sample driver."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ---- Copy headers ----
    bundle_include = out_dir / "include"
    print(f"  copying headers -> {bundle_include}")
    shutil.copytree(install / "include", bundle_include)
    shutil.copytree(pybind11_root / "include" / "pybind11",
                    bundle_include / "pybind11")

    # ---- Copy linker artifacts (libs + .res) into bundle/lib/ ----
    bundle_lib = out_dir / "lib"
    bundle_lib.mkdir()

    _ensure_link_json(install)
    link_json = json.loads((install / "link.json").read_text())
    libs, lib_dirs, link_flags = resolve_libs(link_json, install)

    # Extend lib_dirs with the install's well-known dirs. `libs/` is the
    # Windows convention; `lib/` is the POSIX one - keep both since either
    # may exist depending on the install layout.
    for extra in [install / "libs", install / "lib", install,
                  install / "tcl",
                  install / "dependency_libs" / "openssl" / "lib"]:
        if extra.is_dir() and str(extra) not in lib_dirs:
            lib_dirs.append(str(extra))

    # Filter the install's default mp_embed_data archive - we replace it
    # with the freshly generated one (with the user's full VFS blob).
    #
    # On macOS the artifact's libmp_embed.a also bundles a stale
    # mp_embed_data.o (carrying only the original SSL-cert VFS), so it
    # has to be filtered too or it shadows our fresh data at link time.
    # On Windows mp_embed.lib bundles mp_embed.obj + mp_embed_cpp.obj +
    # the zstd_objs/ that provide ZSTD_decompress et al., and there's no
    # stale data inside it - filtering it would lose every zstd symbol.
    _embed_data_basenames = {"mp_embed_data.lib", "libmp_embed_data.a"}
    if sys.platform != "win32":
        _embed_data_basenames.add("libmp_embed.a")
    libs = [l for l in libs
            if os.path.basename(l).lower() not in _embed_data_basenames]

    install_str = str(install)
    bundled_libs = []  # bundle-relative .lib paths in link order
    bundled_sys_libs = []  # bare system-lib names like "kernel32"
    # Upstream MonolithPy ships every dep into every other dep's prefix,
    # so link.json can list 30+ byte-identical copies of libcrypto.a /
    # libexpat.a / libtclstub.a across dependency_libs/<X>/lib trees.
    # Dedup by content hash so the bundle isn't bloated 30x.
    content_seen = {}  # sha256 -> bundle-relative path

    def _resolve_lib(name: str) -> pathlib.Path | None:
        if os.path.isabs(name) and os.path.isfile(name):
            return pathlib.Path(name)
        for d in lib_dirs:
            for cand in (
                pathlib.Path(d) / name,
                pathlib.Path(d) / (name + ".lib"),
                pathlib.Path(d) / (name + ".a"),
                pathlib.Path(d) / ("lib" + name + ".lib"),
                pathlib.Path(d) / ("lib" + name + ".a"),
            ):
                if cand.is_file():
                    return cand
        return None

    def _rebase_resolved(resolved: pathlib.Path) -> pathlib.Path:
        try:
            return resolved.resolve().relative_to(install_str)
        except ValueError:
            # External path (rare). Drop it directly under lib/external/.
            return pathlib.Path("external") / resolved.name

    # First pass: hash each resolved lib once and pick a canonical
    # bundle-relative path for each content hash (lex-min across all
    # paths that share the same content).
    abs_to_hash: dict[str, str] = {}
    canonical_rel: dict[str, str] = {}
    for lib in libs:
        resolved = _resolve_lib(lib)
        if resolved is None:
            continue
        abs_path = str(resolved.resolve())
        h = abs_to_hash.get(abs_path)
        if h is None:
            h = _file_hash(resolved)
            abs_to_hash[abs_path] = h
        rel_posix = _rebase_resolved(resolved).as_posix()
        if h not in canonical_rel or rel_posix < canonical_rel[h]:
            canonical_rel[h] = rel_posix

    # Second pass: walk libs in link order, dedup by content hash, copy
    # to the canonical rel path.
    for lib in libs:
        resolved = _resolve_lib(lib)
        if resolved is None:
            # System library (no .lib extension on disk). Pass through as-is
            # to the CMake target_link_libraries.
            base = pathlib.Path(lib).name
            if base not in bundled_sys_libs:
                bundled_sys_libs.append(base)
            continue
        h = abs_to_hash[str(resolved.resolve())]
        if h in content_seen:
            continue
        rel_posix = canonical_rel[h]
        dest = bundle_lib / rel_posix
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(resolved, dest)
        bundled_libs.append(rel_posix)
        content_seen[h] = rel_posix

    # ---- Generate the VFS-blob static lib in bundle/lib/ ----
    map_dat = build / "map.dat"
    data_dat = build / "data.dat"
    if not (map_dat.is_file() and data_dat.is_file()):
        raise SystemExit(
            "mkembeddata did not write map.dat / data.dat next to mp_embed_data.c")
    obj_ext = ".obj" if sys.platform == "win32" else ".o"
    embed_obj = build / f"mp_embed_data{obj_ext}"
    target_arch = _detect_target_arch(install)
    write_embed_object(embed_obj, map_dat, data_dat, arch=target_arch)

    # Locate the install's prebuilt mp_embed runtime object. build.bat emits
    # .obj on Windows; the equivalent POSIX build emits .o.
    install_runtime_obj = None
    for cand_ext in (".obj", ".o"):
        cand = install / "Embedded" / f"mp_embed{cand_ext}"
        if cand.is_file():
            install_runtime_obj = cand
            break
    if install_runtime_obj is None:
        raise SystemExit(
            f"mp_embed runtime object not found in {install / 'Embedded'}")

    embed_lib_name = ("mp_embed_data.lib" if sys.platform == "win32"
                      else "libmp_embed_data.a")
    new_embed_lib = bundle_lib / embed_lib_name

    # Pack mp_embed_data + the install's prebuilt mp_embed runtime into one
    # static lib so users only need to link this one + the rest. mp_embed
    # carries the VFS runtime (mp_fopen / mp_open / etc.); mp_embed_data
    # carries nuitka_embed_map / nuitka_embed_data + lengths.
    print(f"  combining embed VFS blob + runtime into {new_embed_lib.name}")
    _build_static_lib(install, build, new_embed_lib,
                      [embed_obj, install_runtime_obj])
    # Must come FIRST so the linker resolves nuitka_embed_* from our
    # freshly generated blob, not from the stale copy inside the
    # artifact's lib/libmp_embed.a (which only carries SSL certs).
    bundled_libs.insert(0, embed_lib_name)

    # POSIX linkers are happier when MonolithPy's many static archives are
    # grouped into one archive. It also gives macOS universal builds a single
    # library to lipo, so arch-specific package archives do not leak warnings
    # to downstream users.
    if sys.platform != "win32":
        bundled_libs = _aggregate_posix_bundle_libs(bundle_lib, bundled_libs)

    # ---- Write src/staticinit_stub.c ----
    bundle_src = out_dir / "src"
    bundle_src.mkdir()
    write_staticinit_stub(bundle_src / "staticinit_stub.c")

    # ---- Process link_flags: pull .res references out of flags into a list,
    # since CMake handles those as link sources. ----
    extra_link_flags = []
    res_files = []
    for f in link_flags:
        if f.lower().endswith(".res"):
            # Copy the .res into the bundle and reference it by relative path.
            src = pathlib.Path(f)
            dest = bundle_lib / src.name
            shutil.copy2(src, dest)
            res_files.append(dest.name)
            continue
        extra_link_flags.append(f)

    # ---- Write the CMake module ----
    write_cmake_module(out_dir, bundled_libs, bundled_sys_libs,
                       extra_link_flags, res_files)

    # ---- Write a sample main.cpp + sample CMakeLists.txt ----
    write_samples(out_dir)

    print(f"\nBundle: {out_dir}")
    _print_bundle_summary(out_dir)


def _build_static_lib(install: pathlib.Path, build: pathlib.Path,
                      out_lib: pathlib.Path, objs: list) -> None:
    """Build a static library archive from a list of object files.

    Uses the system `ar` + `ranlib` (POSIX) or `lib` (Windows). Avoids
    distutils because MonolithPy cross-arch installs may lack a working
    ranlib for the target arch."""
    out_lib.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        lib_exe = shutil.which("lib.exe")
        if lib_exe is None:
            lib_exe = shutil.which("llvm-lib.exe")
        if lib_exe is None:
            raise SystemExit("lib.exe / llvm-lib.exe not on PATH")
        subprocess.check_call(
            [lib_exe, f"/OUT:{out_lib}"] + [str(o) for o in objs])
    else:
        ar = shutil.which("ar")
        if ar is None:
            raise SystemExit("ar not on PATH")
        subprocess.check_call(
            [ar, "rcs", str(out_lib)] + [str(o) for o in objs])


def _merge_static_archives(archives: list[pathlib.Path],
                           out_archive: pathlib.Path) -> None:
    """Merge POSIX static archives into one archive."""
    if not archives:
        raise SystemExit(f"no input archives for {out_archive}")
    out_archive.parent.mkdir(parents=True, exist_ok=True)
    if out_archive.exists():
        out_archive.unlink()
    if len(archives) == 1:
        shutil.copy2(archives[0], out_archive)
        return

    if sys.platform == "darwin":
        libtool = shutil.which("libtool")
        if libtool is None:
            raise SystemExit("libtool not on PATH")
        subprocess.check_call(
            [libtool, "-static", "-o", str(out_archive)]
            + [str(a) for a in archives]
        )
        return

    ar = shutil.which("ar")
    if ar is None:
        raise SystemExit("ar not on PATH")
    with tempfile.TemporaryDirectory(prefix="pack_embed_ar_") as tmp_s:
        tmp = pathlib.Path(tmp_s)
        names = []
        for i, archive in enumerate(archives):
            name = f"lib{i}.a"
            os.symlink(archive.resolve(), tmp / name)
            names.append(name)
        script = "\n".join(
            ["CREATE out.a"]
            + [f"ADDLIB {name}" for name in names]
            + ["SAVE", "END", ""]
        )
        subprocess.run([ar, "-M"], cwd=tmp, input=script, text=True,
                       check=True)
        shutil.move(str(tmp / "out.a"), out_archive)

    ranlib = shutil.which("ranlib")
    if ranlib is not None:
        subprocess.check_call([ranlib, str(out_archive)])


def _prune_empty_dirs(root: pathlib.Path) -> None:
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    for path in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _aggregate_posix_bundle_libs(bundle_lib: pathlib.Path,
                                 bundled_libs: list[str]) -> list[str]:
    if not bundled_libs:
        return bundled_libs
    archives = [bundle_lib / rel for rel in bundled_libs]
    missing = [str(p) for p in archives if not p.is_file()]
    if missing:
        raise SystemExit(
            "cannot aggregate missing static archive(s):\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    out_archive = bundle_lib / POSIX_BUNDLE_ARCHIVE
    print(f"  aggregating {len(archives)} static libraries -> "
          f"{POSIX_BUNDLE_ARCHIVE}")
    _merge_static_archives(archives, out_archive)

    for rel in bundled_libs:
        path = bundle_lib / rel
        if path != out_archive and path.exists():
            path.unlink()
    _prune_empty_dirs(bundle_lib)
    return [POSIX_BUNDLE_ARCHIVE]


def write_cmake_module(out_dir: pathlib.Path, bundled_libs: list[str],
                       bundled_sys_libs: list[str],
                       extra_link_flags: list[str],
                       res_files: list[str]) -> None:
    """Write MonolithPyEmbed.cmake at the bundle root.

    The module exposes one function:
        monolithpy_embed_link(<target>)
    which adds the embed bundle's include dirs, links every static library +
    system library required, and forwards the linker flags MonolithPy needs
    (/FORCE, /LTCG, /NODEFAULTLIB:python3.lib on Windows; full-archive on
    POSIX so PyInit_* symbols don't get stripped).
    """
    libs_cmake = "\n".join(f'        "${{MP_BUNDLE_DIR}}/lib/{l}"' for l in bundled_libs)
    sys_libs_cmake = "\n".join(f'        "{l}"' for l in bundled_sys_libs)
    res_cmake = "\n".join(f'        "${{MP_BUNDLE_DIR}}/lib/{r}"' for r in res_files)

    # Pull -framework / FrameworkName pairs out of link_flags into a
    # separate list so the CMake module can emit them as -Wl,-framework,X
    # (single arg) — bare "-framework" / "X" pairs in target_link_options
    # get misinterpreted as filenames by the macOS Makefile generator.
    # Skip .res entries too (we already moved those into res_files).
    frameworks = []
    filtered_flags = []
    i = 0
    while i < len(extra_link_flags):
        f = extra_link_flags[i]
        if f == "-framework" and i + 1 < len(extra_link_flags):
            frameworks.append(extra_link_flags[i + 1])
            i += 2
            continue
        if f.lower().endswith(".res"):
            i += 1
            continue
        filtered_flags.append(f)
        i += 1
    flags_cmake = "\n".join(f'        "{f}"' for f in filtered_flags)
    frameworks_cmake = "\n".join(f'        "{fw}"' for fw in frameworks)

    # Find versioned header dirs under include/ (e.g. python3.14/).
    extra_include = ""
    for d in sorted((out_dir / "include").iterdir()):
        if d.is_dir() and (d / "Python.h").is_file():
            extra_include += f'\nset(MP_BUNDLE_PYTHON_INCLUDE_DIR "${{MP_BUNDLE_DIR}}/include/{d.name}")'

    module = (out_dir / "MonolithPyEmbed.cmake")
    module.write_text(f"""# Auto-generated by pack_embed.py - do not edit.
# Provides one function:
#     monolithpy_embed_link(<target>)
#
# Drop this file (and the rest of the bundle dir) into your project, then:
#     include(<bundle>/MonolithPyEmbed.cmake)
#     add_executable(my_app main.cpp)
#     monolithpy_embed_link(my_app)
#
# Your main needs to call mp_init_static_modules() once before
# Py_InitializeFromConfig() to register all statically-linked extension
# modules. See samples/main.cpp for a minimal example.

cmake_minimum_required(VERSION 3.16)

get_filename_component(MP_BUNDLE_DIR "${{CMAKE_CURRENT_LIST_DIR}}" ABSOLUTE)

set(MP_BUNDLE_INCLUDE_DIR    "${{MP_BUNDLE_DIR}}/include")
set(MP_BUNDLE_LIB_DIR        "${{MP_BUNDLE_DIR}}/lib")
set(MP_BUNDLE_SRC_DIR        "${{MP_BUNDLE_DIR}}/src")
{extra_include}

set(MP_BUNDLE_LIBS
{libs_cmake}
)

set(MP_BUNDLE_SYS_LIBS
{sys_libs_cmake}
)

set(MP_BUNDLE_LINK_FLAGS
{flags_cmake}
)

set(MP_BUNDLE_FRAMEWORKS
{frameworks_cmake}
)

set(MP_BUNDLE_RES_FILES
{res_cmake}
)

function(monolithpy_embed_link target)
    target_include_directories(${{target}} PRIVATE
        "${{MP_BUNDLE_INCLUDE_DIR}}"
        "${{MP_BUNDLE_PYTHON_INCLUDE_DIR}}"
    )

    # The staticinit stub is compiled with -DPy_BUILD_CORE so that the
    # Py_BUILD_CORE-gated Py_InitStaticModules() in <staticinit.h> becomes
    # visible. It re-exports it as mp_init_static_modules() for user code.
    target_sources(${{target}} PRIVATE
        "${{MP_BUNDLE_SRC_DIR}}/staticinit_stub.c"
    )
    set_source_files_properties(
        "${{MP_BUNDLE_SRC_DIR}}/staticinit_stub.c"
        PROPERTIES COMPILE_DEFINITIONS Py_BUILD_CORE
    )

    target_link_libraries(${{target}} PRIVATE ${{MP_BUNDLE_LIBS}}
                                              ${{MP_BUNDLE_SYS_LIBS}}
                                              ${{MP_BUNDLE_RES_FILES}})
    target_link_options(${{target}} PRIVATE ${{MP_BUNDLE_LINK_FLAGS}})

    # macOS frameworks come straight from the install's link.json -
    # passing through -Wl,-framework,X as a single linker arg avoids the
    # Makefile generator splitting "-framework"/"X" into bare filenames.
    foreach(_mp_fw IN LISTS MP_BUNDLE_FRAMEWORKS)
        target_link_libraries(${{target}} PRIVATE "-Wl,-framework,${{_mp_fw}}")
    endforeach()

    if(WIN32)
        set_property(TARGET ${{target}} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded")
    elseif(UNIX AND NOT APPLE)
        # Linux: standard libc/pthread/dl support libs.
        target_link_libraries(${{target}} PRIVATE m pthread dl util)
    endif()
endfunction()
""")


def write_samples(out_dir: pathlib.Path) -> None:
    """Write samples/main.cpp + samples/CMakeLists.txt as a runnable smoke
    test of the bundle. The sample is the same minimal embedding pattern
    we use ourselves (Py_InitializeFromConfig + PyRun_AnyFile)."""
    samples = out_dir / "samples"
    samples.mkdir()
    (samples / "main.cpp").write_text(
        """// Minimal embedded-Python entry point using the MonolithPyEmbed bundle.
//
// Build with:
//   cmake -S samples -B samples/build
//   cmake --build samples/build --config Release
// Then run samples/build/Release/embed_sample (Win) or .../embed_sample (POSIX).
//
// Usage:
//   embed_sample                   -> interactive REPL
//   embed_sample script.py [args]  -> run script.py with sys.argv = [script, ...]
//   embed_sample -c \"code\"         -> exec a string
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <iostream>
#include <string>

namespace py = pybind11;

extern \"C\" void mp_init_static_modules(void);

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    // Register statically-linked extension modules BEFORE the interpreter
    // starts importing.
    mp_init_static_modules();

    py::scoped_interpreter guard{};

    try {
        // Mirror our argv into sys.argv so user scripts see the right values.
        py::list pyargv;
        for (int i = (argc > 1 ? 1 : 0); i < argc; i++) {
#ifdef _WIN32
            pyargv.append(py::cast(std::wstring(argv[i])));
#else
            pyargv.append(py::cast(std::string(argv[i])));
#endif
        }
        if (pyargv.empty()) pyargv.append(py::cast(std::string(\"\")));
        py::module_::import(\"sys\").attr(\"argv\") = pyargv;

        if (argc >= 3 &&
#ifdef _WIN32
            std::wstring(argv[1]) == L\"-c\"
#else
            std::string(argv[1]) == \"-c\"
#endif
        ) {
            // -c \"code\": exec the string in __main__'s globals.
#ifdef _WIN32
            std::wstring ws(argv[2]);
            std::string code(ws.begin(), ws.end());  // ASCII-safe
#else
            std::string code(argv[2]);
#endif
            py::exec(code, py::module_::import(\"__main__\").attr(\"__dict__\"));
        } else if (argc > 1) {
            // Run the supplied script file with __name__ == \"__main__\".
#ifdef _WIN32
            std::wstring ws(argv[1]);
            std::string script(ws.begin(), ws.end());  // ASCII-safe
#else
            std::string script(argv[1]);
#endif
            py::dict globals;
            globals[\"__name__\"] = py::cast(\"__main__\");
            globals[\"__file__\"] = py::cast(script);
            py::eval_file(script, globals);
        } else {
            PyRun_InteractiveLoop(stdin, \"<stdin>\");
        }
    } catch (py::error_already_set& e) {
        if (e.matches(PyExc_SystemExit)) {
            try { return e.value().attr(\"code\").cast<int>(); }
            catch (...) { return 1; }
        }
        e.restore();
        PyErr_Print();
        return 1;
    }
    return 0;
}
""")
    (samples / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.16)
project(mp_embed_sample LANGUAGES CXX C)

# Pick up the bundle one directory up from this CMakeLists.txt.
include(${CMAKE_CURRENT_LIST_DIR}/../MonolithPyEmbed.cmake)

add_executable(embed_sample main.cpp)
set_target_properties(embed_sample PROPERTIES CXX_STANDARD 17)

monolithpy_embed_link(embed_sample)
""")


def _print_bundle_summary(out_dir: pathlib.Path) -> None:
    """Print sizes for top-level dirs in the bundle so the user can sanity-
    check the output."""
    def _du(p: pathlib.Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    for sub in sorted(out_dir.iterdir()):
        if sub.is_dir():
            mb = _du(sub) / (1024 * 1024)
            print(f"  {sub.name:<10} {mb:>8.1f} MB")
        else:
            kb = sub.stat().st_size / 1024
            print(f"  {sub.name:<10} {kb:>8.1f} KB")


def lipo_bundles(arm64: pathlib.Path, x86_64: pathlib.Path,
                 out: pathlib.Path) -> None:
    """Merge two per-arch macOS bundles into one universal bundle.

    Each input is expected to already contain the per-arch aggregate archive
    produced by build_bundle(). `lipo -create` combines those two archives into
    a single universal archive. This allows packages like NumPy to ship
    arch-specific dispatch archives without exposing wrong-arch static
    libraries to downstream universal links. Headers come from the arm64 input
    verbatim (identical between arches). The CMake module is copied from arm64,
    updated to link the single universal bundle archive, and stripped of
    arch-specific `-arch ...` link flags, since the consumer's CMake project
    should drive arch selection via CMAKE_OSX_ARCHITECTURES.
    """
    if shutil.which("lipo") is None:
        raise SystemExit(
            "lipo not on PATH. This subcommand only runs on macOS.")
    if out.exists():
        shutil.rmtree(out)

    # Skip arm64/lib/ at the TOP LEVEL only; nested "lib" dirs anywhere
    # else in the bundle (e.g. include/.../lib/) must survive.
    arm64_resolved = str(arm64.resolve())
    def _skip_toplevel_lib(src: str, names: list[str]) -> list[str]:
        return ["lib"] if os.path.realpath(src) == arm64_resolved else []

    print(f"  copying scaffold from {arm64}")
    shutil.copytree(arm64, out, ignore=_skip_toplevel_lib)
    (out / "lib").mkdir()

    arm_archive = arm64 / "lib" / POSIX_BUNDLE_ARCHIVE
    x86_archive = x86_64 / "lib" / POSIX_BUNDLE_ARCHIVE
    missing = [str(p) for p in (arm_archive, x86_archive) if not p.is_file()]
    if missing:
        raise SystemExit(
            "lipo requires per-arch aggregate archive(s) generated by "
            "pack_embed.py:\n" + "\n".join(f"  {p}" for p in missing)
        )
    universal_archive = out / "lib" / POSIX_BUNDLE_ARCHIVE
    print(f"  lipo'ing {POSIX_BUNDLE_ARCHIVE}")
    subprocess.check_call([
        "lipo", "-create",
        str(arm_archive), str(x86_archive),
        "-output", str(universal_archive),
    ])

    # Strip arm64-only flags from the copied CMake module so a consumer
    # CMake project's CMAKE_OSX_ARCHITECTURES governs the slice selection.
    cmake_file = out / "MonolithPyEmbed.cmake"
    if cmake_file.is_file():
        text = cmake_file.read_text()
        text = _replace_cmake_bundle_libs(text, [POSIX_BUNDLE_ARCHIVE])
        text = re.sub(r'        "-arch"\n        "[^"]+"\n', '', text)
        cmake_file.write_text(text)

    print(f"\nUniversal bundle: {out}")
    _print_bundle_summary(out)


def _replace_cmake_bundle_libs(text: str, libs: list[str]) -> str:
    replacement = "\n".join(
        f'        "${{MP_BUNDLE_DIR}}/lib/{lib}"' for lib in libs
    )
    return re.sub(
        r"(set\(MP_BUNDLE_LIBS\n)(.*?)(\n\))",
        lambda match: match.group(1) + replacement + match.group(3),
        text,
        count=1,
        flags=re.S,
    )


def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir).resolve()

    if args.lipo:
        arm64 = pathlib.Path(args.lipo[0]).resolve()
        x86_64 = pathlib.Path(args.lipo[1]).resolve()
        lipo_bundles(arm64, x86_64, out_dir)
        return

    install = pathlib.Path(args.install).resolve()
    pybind11_root = pathlib.Path(args.pybind11).resolve()

    _find_python(install)  # validate early
    if not (pybind11_root / "include" / "pybind11" / "embed.h").is_file():
        raise SystemExit(f"pybind11 headers not found at {pybind11_root}")

    build = pathlib.Path(tempfile.mkdtemp(prefix="pack_embed_"))
    print(f"build dir: {build}")

    try:
        print("[1/3] staging embed tree")
        staging = build / "embed_staging"
        stage_embed_tree(install, staging)

        print("[2/3] generating VFS blob")
        run_mkembeddata(install, staging, build)

        print("[3/3] assembling bundle")
        build_bundle(install, build, out_dir, pybind11_root)
    finally:
        if not args.keep_build:
            shutil.rmtree(build, ignore_errors=True)
        else:
            print(f"build dir retained: {build}")


if __name__ == "__main__":
    main()

# MonolithPy VFS Packer

Pack a [MonolithPy](https://github.com/Nuitka/MonolithPy) install (Python +
your `pip install`-ed packages) into a CMake-consumable bundle. Drop the
bundle into your C++ project and a single `monolithpy_embed_link(<target>)`
call gives you a self-contained executable with the entire stdlib and
site-packages tree embedded - no DLLs, no zip, no on-disk Python.

The bundle ships a hand-emitted static object file (COFF on Windows, ELF on
Linux, Mach-O on macOS) carrying the compressed VFS blob, plus every static
library MonolithPy needs.

## Quick start

```sh
# 1. Clone with submodules (pybind11 lives under third_party/pybind11).
git clone --recurse-submodules <this-repo>
# (or in an existing checkout: git submodule update --init --recursive)

# 2. Drop a MonolithPy build into ./monolithpy314/ (the result of build.bat
#    or the corresponding POSIX build).

# 3. Install whatever you want bundled into the embedded interpreter.
./monolithpy314/python -m pip install numpy scipy pandas matplotlib

# 4. Pack.
python pack_embed.py
# -> dist/embed_bundle/

# 5. Build the sample to verify.
cmake -S dist/embed_bundle/samples -B build
cmake --build build --config Release
```

The resulting executable runs scripts (`embed_sample foo.py args...`),
`-c "code"` strings, or drops into a REPL with no arguments.

## Bundle layout

```
<out>/
  MonolithPyEmbed.cmake     # exposes monolithpy_embed_link(<target>)
  include/                  # Python.h, mp_embed.h, staticinit.h, pybind11/
  lib/                      # every required .lib/.a in its original subdir,
                            # plus mp_embed_data.{lib,a} (the VFS blob +
                            # mp_embed runtime)
  src/staticinit_stub.c     # bridges Py_BUILD_CORE-gated
                            # Py_InitStaticModules() to user code
  samples/                  # main.cpp + CMakeLists.txt smoke test
```

## Using the bundle

In your project's `CMakeLists.txt`:

```cmake
include(/path/to/embed_bundle/MonolithPyEmbed.cmake)
add_executable(my_app main.cpp)
monolithpy_embed_link(my_app)
```

Your `main` should call `mp_init_static_modules()` once before initializing
the interpreter, to register every statically-linked extension's
`PyInit_*`. See `samples/main.cpp` for the canonical pattern.

## CLI

```
pack_embed.py [--install DIR] [--out-dir DIR] [--pybind11 DIR] [--keep-build]
    Defaults: ./monolithpy314, ./dist/embed_bundle, ./third_party/pybind11

pack_embed.py --lipo <ARM64_BUNDLE> <X86_64_BUNDLE> --out-dir <UNIVERSAL>
    macOS-only. Lipo the two per-arch aggregate archives into one universal
    bundle archive.
    Run pack_embed once on each per-arch MonolithPy install, then this.
```

## How it works

1. **Stage** - copy `Lib/` and `tcl/` from the install under
   `__relative__/`, plus absolute-path data (e.g. `/c/vfs/ssl/cert.pem`).
   Drop linker artifacts (`.lib`, `.obj`, `.pyd`, `.dll`, `.lib.orig`,
   `.lib.link.json`) and any `tests/` subtree (hundreds of MB of fixtures).
2. **Compress** - invoke MonolithPy's own `mkembeddata.py` to produce
   `map.dat` + `data.dat` (per-file zstd, parallelized). The result is the
   VFS blob the embedded interpreter mounts at runtime.
3. **Wrap** - hand-emit a static object file in the host's native format
   exposing the four symbols `mp_embed.c` looks for:
   `nuitka_embed_map`, `nuitka_embed_data`, `nuitka_embed_map_len`,
   `nuitka_embed_data_len`. We don't go through `cl`/`gcc`/`clang` because
   compiling an `unsigned char foo[] = { 0x.., ... }` literal of a
   ~1 GB blob exhausts >13 GB of compiler memory; emitting the object
   directly takes a few seconds.
4. **Archive** - bundle the new object file plus the install's prebuilt
   `mp_embed` runtime into a static library, then on POSIX aggregate that
   runtime archive and every other static library into
   `libmonolithpy_bundle.a`.
5. **Generate** - write the CMake module that wires the bundle archive,
   system library, and linker flag MonolithPy needs into a single
   `monolithpy_embed_link(<target>)` function.

## Requirements

- A MonolithPy install (run `build.bat` or the POSIX equivalent in the
  [Nuitka-Python](https://github.com/Nuitka/Nuitka-Python) checkout).
- pybind11 v2.13.6 headers (vendored as a submodule under
  `third_party/pybind11/`).
- Python 3.10+ to run `pack_embed.py`.

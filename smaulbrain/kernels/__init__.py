"""Native kernel interface: build + runtime dispatch.

Portable scalar C++ only (no AVX2/AVX-512 intrinsics anywhere). If a compiler
or build fails, the error is explicit and actionable; the Python scalar/numpy
reference path always remains available.
"""
import errno
import os
import sys
import hashlib
import shutil
from functools import lru_cache
from pathlib import Path

from .. import cpu as _cpu

_SO_NAME = "smaul_kernels"


def _digest_and_target(caps):
    """(compiler, flags, source, target path) for the current source, or None.

    The digest over the current ``cpu_kernels.cpp`` bytes plus the compiler
    string and flag set defines the canonical library name. Sharing the
    formula between build and load prevents a stale differently-named build
    from shadowing the fresh one.
    """
    here = Path(__file__).resolve().parent
    source = here / "cpu_kernels.cpp"
    if not source.is_file():
        return None
    configured = os.environ.get("CXX", "c++")
    compiler = shutil.which(configured) or shutil.which("c++")
    if compiler is None:
        return None
    variant_flag = "-mavx" if caps.get("has_avx", False) else "-msse2"
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC",
             "-fno-fast-math", variant_flag]
    digest = hashlib.sha256(
        source.read_bytes() + compiler.encode("utf-8") +
        " ".join(flags).encode("ascii")
    ).hexdigest()[:16]
    target = here / f"{_SO_NAME}-{variant_flag[1:]}-{digest}.so"
    return compiler, flags, source, target


def build_here(build_dir="", verbose: bool = True) -> str:
    """Compile the optional native C++ kernels to a loadable shared library."""
    import subprocess
    import tempfile

    # Report a missing source before anything else: without it the compiler
    # error below would send someone hunting for c++ on a source-less install.
    source = Path(__file__).resolve().parent / "cpu_kernels.cpp"
    if not source.is_file():
        raise RuntimeError(f"native kernel source missing: {source}")
    resolved = _digest_and_target(_cpu.detect())
    if resolved is None:
        raise RuntimeError(
            "native build requires a C++ compiler (c++/g++); "
            "the portable NumPy/Python path remains available.")
    compiler, flags, source, target = resolved

    # Compile to a temporary file, then atomically publish a complete library.
    # build_dir optionally controls scratch space, not the final load location.
    scratch = Path(build_dir).resolve() if build_dir else target.parent
    scratch.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{target.name}.{os.getpid()}.", suffix=".tmp.so", dir=scratch)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        if not target.is_file():
            command = [compiler, *flags, str(source), "-o", str(temporary)]
            result = subprocess.run(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, timeout=90)
            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.strip() or "native C++ kernel compilation failed")
            try:
                os.replace(temporary, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                # A scratch dir on another filesystem cannot be renamed into
                # place; publish with a copy instead.
                shutil.move(str(temporary), str(target))
        else:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    # try_load is memoized, so clear a cached "missing" result after a build.
    try_load.cache_clear()
    if verbose:
        print(f"Built native kernels: {target} ({variant_flag})")
    return str(target)


@lru_cache(maxsize=1)
def try_load():
    """(module, info). Never raises for missing builds; reports reason."""
    caps = _cpu.detect()
    try:
        import ctypes
        import glob
        here = os.path.dirname(os.path.abspath(__file__))
        # Resolve the library matching the CURRENT source digest first. Loading
        # whatever .so has the newest mtime once shadowed a fresh build with a
        # stale one (the stale name embeds an older digest), so the digest
        # path wins and discovery is only a fallback.
        cands = []
        advisory = None
        resolved = _digest_and_target(caps)
        if resolved is not None and resolved[3].is_file():
            cands.append(str(resolved[3]))
            all_names = [p for p in glob.glob(os.path.join(here, "smaul_kernels*.so"))
                         if ".tmp." not in os.path.basename(p)]
            if len(all_names) > 1:
                advisory = "stale sibling builds present; current source digest build used"
        if not cands:
            # Partial builds publish "<name>.so.<pid>.<rand>.tmp.so"; never let
            # those shadow a complete library.
            cands = [p for p in glob.glob(os.path.join(here, "smaul_kernels*.so"))
                     if ".tmp." not in os.path.basename(p)]
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if not cands:
            return None, {"available": False,
                          "reason": "native module not built; using portable scalar/numpy path",
                          "cpu": caps, "dispatch": _cpu.dispatch_name()}
        lib = ctypes.CDLL(cands[0])
        # Declare scalar signatures; without these every Python float argument
        # would be passed as a C double and silently misread by the callee.
        ptr_f32 = ctypes.POINTER(ctypes.c_float)
        ptr_u8 = ctypes.POINTER(ctypes.c_uint8)
        lib.rmsnorm_fwd.argtypes = [ptr_f32, ptr_f32, ptr_f32, ctypes.c_int, ctypes.c_float]
        lib.rmsnorm_fwd.restype = None
        lib.ternary_pack.argtypes = [ptr_f32, ptr_u8, ctypes.c_int, ptr_f32, ptr_f32]
        lib.ternary_pack.restype = None
        lib.packed_matvec.argtypes = [ptr_f32, ptr_u8, ptr_f32, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_float]
        lib.packed_matvec.restype = None
        lib.recurrent_step.argtypes = [ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f32,
                                       ptr_f32, ptr_f32, ctypes.c_int]
        lib.recurrent_step.restype = None
        lib.clone_perturb.argtypes = [ptr_f32, ptr_f32, ctypes.c_int, ctypes.c_uint,
                                      ctypes.c_float, ctypes.c_int]
        lib.clone_perturb.restype = None
        return lib, {"available": True, "path": cands[0], "cpu": caps,
                     "dispatch": _cpu.dispatch_name(), "advisory": advisory}
    except (OSError, AttributeError) as e:
        return None, {"available": False, "reason": f"native load failed: {e}",
                      "cpu": caps}

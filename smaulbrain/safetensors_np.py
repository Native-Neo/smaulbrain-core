"""Minimal safetensors reader/writer for NumPy arrays.

Implements the published safetensors container directly (8-byte little-endian
header length, UTF-8 JSON header, raw little-endian tensor bytes) so the runtime
does not need a compiled dependency. Only dense, C-contiguous arrays are stored,
which is exactly what the model and optimizer produce.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

_DTYPE_TO_ST = {
    np.dtype("bool"): "BOOL",
    np.dtype("uint8"): "U8",
    np.dtype("int8"): "I8",
    np.dtype("int16"): "I16",
    np.dtype("int32"): "I32",
    np.dtype("int64"): "I64",
    np.dtype("float16"): "F16",
    np.dtype("float32"): "F32",
    np.dtype("float64"): "F64",
}
_ST_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ST.items()}


def _int_product(sizes):
    result = 1
    for s in sizes:
        result *= int(s)
    return result


def _metadata_strings(metadata):
    if not metadata:
        return None
    return {str(k): str(v) for k, v in metadata.items()}


def save_file(tensors, filename, metadata=None):
    """Write ``{name: ndarray}`` to ``filename`` in safetensors format."""
    filename = os.fspath(filename)
    parent = os.path.dirname(os.path.abspath(filename))
    if parent:
        os.makedirs(parent, exist_ok=True)
    header = {}
    meta = _metadata_strings(metadata)
    if meta:
        header["__metadata__"] = meta
    payload = []
    offset = 0
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be non-empty strings")
        arr = np.asarray(value)
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        arr = np.ascontiguousarray(arr)
        dtype = _DTYPE_TO_ST.get(arr.dtype)
        if dtype is None:
            raise TypeError(f"unsupported safetensors dtype {arr.dtype} for {name}")
        nbytes = int(arr.nbytes)
        header[name] = {
            "dtype": dtype,
            "shape": [int(s) for s in arr.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        payload.append(arr.tobytes(order="C"))
        offset += nbytes
    raw_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # Pad the header so the data section starts on an 8-byte boundary.
    padding = (-(8 + len(raw_header))) % 8
    raw_header += b" " * padding
    temporary = f"{filename}.{os.getpid()}.tmp"
    try:
        with open(temporary, "wb") as handle:
            handle.write(struct.pack("<Q", len(raw_header)))
            handle.write(raw_header)
            for chunk in payload:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, filename)
        _fsync_dir(parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return filename


def _fsync_dir(directory):
    """Flush the directory entry so a rename survives a power failure."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def load_file(filename):
    """Read a safetensors file, returning ``(tensors, metadata)``."""
    filename = os.fspath(filename)
    with open(filename, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{filename}: truncated safetensors header")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > (1 << 30):
            raise ValueError(f"{filename}: invalid safetensors header length")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"{filename}: truncated safetensors header body")
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{filename}: malformed safetensors header") from exc
        data = handle.read()
    if not isinstance(header, dict):
        raise ValueError(f"{filename}: safetensors header is not an object")
    metadata = header.pop("__metadata__", None) or {}
    tensors = {}
    spans = []
    for name, spec in header.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{filename}: invalid tensor entry {name!r}")
        dtype = _ST_TO_DTYPE.get(spec.get("dtype"))
        if dtype is None:
            raise ValueError(f"{filename}: unsupported dtype {spec.get('dtype')!r}")
        raw_shape = spec.get("shape", ())
        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError(f"{filename}: tensor {name!r} shape must be a list of sizes")
        try:
            shape = tuple(int(s) for s in raw_shape)
            offsets = spec.get("data_offsets")
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                raise ValueError
            start, end = int(offsets[0]), int(offsets[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{filename}: tensor {name!r} has invalid shape or offsets") from exc
        if start < 0 or end < start or end > len(data):
            raise ValueError(f"{filename}: tensor {name!r} offsets out of range")
        expected = _int_product(shape) * dtype.itemsize
        if end - start != expected:
            raise ValueError(f"{filename}: tensor {name!r} byte length mismatch")
        for other_start, other_end, other_name in spans:
            if start < other_end and other_start < end:
                raise ValueError(
                    f"{filename}: tensor {name!r} data_offsets overlap {other_name!r}; "
                    "the safetensors spec requires non-overlapping regions")
        spans.append((start, end, name))
        arr = np.frombuffer(data[start:end], dtype=dtype).reshape(shape).copy()
        if arr.dtype == np.uint8 and spec.get("dtype") == "BOOL":
            arr = arr.astype(np.bool_)
        tensors[name] = arr
    return tensors, metadata


def load(filename):
    return load_file(filename)[0]


def save(tensors, filename, metadata=None):
    return save_file(tensors, filename, metadata=metadata)


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

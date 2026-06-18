#!/usr/bin/env python3
"""
Split a Qwen3-4B GGUF model for the pipeline-brick TP prototype.

This tool only supports raw F32/F16/BF16 tensors. Quantized tensors such as
Q4_K_M must be converted to F16/BF16 first.
"""

import argparse
import os
import re
import struct
import sys


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

QUANT_TYPES = {
    2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29,
}

OUTPUT_SPLIT_PATTERNS = [
    re.compile(r"blk\.\d+\.attn_q\.weight$"),
    re.compile(r"blk\.\d+\.attn_k\.weight$"),
    re.compile(r"blk\.\d+\.attn_v\.weight$"),
    re.compile(r"blk\.\d+\.ffn_gate\.weight$"),
    re.compile(r"blk\.\d+\.ffn_up\.weight$"),
]

INPUT_SPLIT_PATTERNS = [
    re.compile(r"blk\.\d+\.attn_output\.weight$"),
    re.compile(r"blk\.\d+\.ffn_down\.weight$"),
]

RANK0_SMALL_Q_HEADS = [4, 12, 8, 8]
RANK0_SMALL_KV_HEADS = [1, 3, 2, 2]
RANK0_SMALL_FFN = [256, 3200, 3136, 3136]
QWEN3_4B_HEAD_DIM = 128
PIPELINE_BRICK_KV_PREFIX = "pipeline_brick."


def pad(x, align):
    return (x + align - 1) // align * align


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def read_value_raw(f, vtype):
    if vtype in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        return f.read(1)
    if vtype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return f.read(2)
    if vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return f.read(4)
    if vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return f.read(8)
    if vtype == GGUF_TYPE_STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        return struct.pack("<Q", n) + f.read(n)
    if vtype == GGUF_TYPE_ARRAY:
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        raw = struct.pack("<I", etype) + struct.pack("<Q", n)
        for _ in range(n):
            raw += read_value_raw(f, etype)
        return raw
    raise ValueError(f"unknown GGUF value type {vtype}")


class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.kv = []
        self.tensors = []
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.data_start = 0
        self._read()

    def _read(self):
        f = self.f
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"{self.path} is not a GGUF file")

        version = struct.unpack("<I", f.read(4))[0]
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}")

        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            raw = read_value_raw(f, vtype)
            self.kv.append((key, vtype, raw))
            if key == "general.alignment" and vtype == GGUF_TYPE_UINT32:
                self.alignment = struct.unpack("<I", raw)[0]

        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            self.tensors.append({
                "name": name,
                "dims": dims,
                "shape": tuple(reversed(dims)),
                "dtype": dtype,
                "offset": offset,
            })

        self.data_start = pad(f.tell(), self.alignment)

    def read_tensor(self, tensor):
        n_elem = 1
        for d in tensor["dims"]:
            n_elem *= d

        dtype = tensor["dtype"]
        if dtype == GGML_TYPE_F32:
            n_bytes = n_elem * 4
        elif dtype in (GGML_TYPE_F16, GGML_TYPE_BF16):
            n_bytes = n_elem * 2
        elif dtype in QUANT_TYPES:
            raise RuntimeError(
                f"quantized tensor {tensor['name']} type={dtype} cannot be split; convert to F16/BF16 first"
            )
        else:
            raise RuntimeError(f"unsupported tensor {tensor['name']} type={dtype}")

        self.f.seek(self.data_start + tensor["offset"])
        return self.f.read(n_bytes)

    def close(self):
        self.f.close()


class GGUFWriter:
    def __init__(self, path, alignment):
        self.path = path
        self.alignment = alignment
        self.kv = []
        self.tensors = []
        self.data_size = 0
        self.data_path = f"{path}.data.tmp"
        self.data_f = open(self.data_path, "wb")

    def add_kv(self, key, vtype, raw):
        self.kv.append((key, vtype, raw))

    def add_tensor(self, name, dims, dtype, raw):
        offset = self.data_size
        pad_bytes = pad(len(raw), self.alignment) - len(raw)
        self.tensors.append((name, list(dims), dtype, offset))
        self.data_f.write(raw)
        if pad_bytes:
            self.data_f.write(b"\x00" * pad_bytes)
        self.data_size += len(raw) + pad_bytes

    def write(self):
        self.data_f.flush()
        self.data_f.close()

        with open(self.path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv)))

            for key, vtype, raw in self.kv:
                key_b = key.encode("utf-8")
                f.write(struct.pack("<Q", len(key_b)))
                f.write(key_b)
                f.write(struct.pack("<I", vtype))
                f.write(raw)

            for name, dims, dtype, offset in self.tensors:
                name_b = name.encode("utf-8")
                f.write(struct.pack("<Q", len(name_b)))
                f.write(name_b)
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", dtype))
                f.write(struct.pack("<Q", offset))

            f.write(b"\x00" * (pad(f.tell(), self.alignment) - f.tell()))
            with open(self.data_path, "rb") as data_f:
                while True:
                    chunk = data_f.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        os.remove(self.data_path)


def split_kind(name):
    if any(p.search(name) for p in OUTPUT_SPLIT_PATTERNS):
        return "output"
    if any(p.search(name) for p in INPUT_SPLIT_PATTERNS):
        return "input"
    return "copy"


def split_group(name):
    if re.search(r"blk\.\d+\.attn_q\.weight$", name):
        return "q"
    if re.search(r"blk\.\d+\.attn_k\.weight$", name):
        return "kv"
    if re.search(r"blk\.\d+\.attn_v\.weight$", name):
        return "kv"
    if re.search(r"blk\.\d+\.attn_output\.weight$", name):
        return "q"
    if re.search(r"blk\.\d+\.ffn_gate\.weight$", name):
        return "ffn"
    if re.search(r"blk\.\d+\.ffn_up\.weight$", name):
        return "ffn"
    if re.search(r"blk\.\d+\.ffn_down\.weight$", name):
        return "ffn"
    return None


def elem_size(dtype):
    if dtype == GGML_TYPE_F32:
        return 4
    if dtype in (GGML_TYPE_F16, GGML_TYPE_BF16):
        return 2
    raise RuntimeError(f"unsupported raw tensor type {dtype}")


def uniform_split_sizes(count, tp):
    if count % tp != 0:
        raise RuntimeError(f"dimension {count} is not divisible by TP={tp}")
    return [count // tp] * tp


def rank0_small_split_sizes(name, shape, split_dim, tp):
    if tp != 4:
        raise RuntimeError("rank0-small layout is defined only for TP=4")

    group = split_group(name)
    if group == "q":
        sizes = [h * QWEN3_4B_HEAD_DIM for h in RANK0_SMALL_Q_HEADS]
    elif group == "kv":
        sizes = [h * QWEN3_4B_HEAD_DIM for h in RANK0_SMALL_KV_HEADS]
    elif group == "ffn":
        sizes = list(RANK0_SMALL_FFN)
    else:
        raise RuntimeError(f"cannot choose asymmetric split for tensor {name}")

    dim_size = shape[split_dim]
    if sum(sizes) != dim_size:
        raise RuntimeError(
            f"rank0-small split for {name} sums to {sum(sizes)}, expected dimension {dim_size}"
        )
    return sizes


def split_sizes(name, shape, split_dim, tp, layout):
    if layout == "uniform":
        return uniform_split_sizes(shape[split_dim], tp)
    if layout == "rank0-small":
        return rank0_small_split_sizes(name, shape, split_dim, tp)
    raise RuntimeError(f"unsupported TP layout {layout}")


def split_raw_tensor(raw, shape, dtype, split_dim, rank, sizes):
    if len(shape) != 2:
        raise RuntimeError(f"expected a 2D tensor to split, got shape {shape}")

    rows, cols = shape
    es = elem_size(dtype)
    chunk = sizes[rank]
    offset = sum(sizes[:rank])

    if split_dim == 0:
        row_bytes = cols * es
        start = offset * row_bytes
        end = start + chunk * row_bytes
        return raw[start:end]

    if split_dim == 1:
        row_bytes = cols * es
        chunk_bytes = chunk * es
        start_col = offset * es
        out = bytearray(rows * chunk_bytes)
        dst = 0
        for row in range(rows):
            src = row * row_bytes + start_col
            out[dst:dst + chunk_bytes] = raw[src:src + chunk_bytes]
            dst += chunk_bytes
        return bytes(out)

    raise RuntimeError(f"unsupported split dim {split_dim}")


def raw_i32(value):
    return struct.pack("<i", value)


def raw_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def add_pipeline_brick_metadata(writers, tp, layout):
    if layout != "rank0-small":
        return

    for rank, writer in enumerate(writers):
        writer.add_kv("pipeline_brick.tp_layout", GGUF_TYPE_STRING, raw_string(layout))
        writer.add_kv("pipeline_brick.tp_rank", GGUF_TYPE_INT32, raw_i32(rank))
        writer.add_kv("pipeline_brick.tp_size", GGUF_TYPE_INT32, raw_i32(tp))
        writer.add_kv("pipeline_brick.tp_q_heads", GGUF_TYPE_INT32, raw_i32(RANK0_SMALL_Q_HEADS[rank]))
        writer.add_kv("pipeline_brick.tp_kv_heads", GGUF_TYPE_INT32, raw_i32(RANK0_SMALL_KV_HEADS[rank]))
        writer.add_kv("pipeline_brick.tp_ffn", GGUF_TYPE_INT32, raw_i32(RANK0_SMALL_FFN[rank]))


def split_model(input_path, tp, layout):
    if tp not in (2, 4):
        raise RuntimeError("pipeline-brick TP prototype expects --tp 2 or --tp 4")
    if layout == "rank0-small" and tp != 4:
        raise RuntimeError("--layout rank0-small requires --tp 4")

    reader = GGUFReader(input_path)
    try:
        for tensor in reader.tensors:
            if tensor["dtype"] not in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16):
                raise RuntimeError(
                    f"quantized tensor {tensor['name']} type={tensor['dtype']} cannot be split; convert to F16/BF16 first"
                )

        base, ext = os.path.splitext(input_path)
        writers = [GGUFWriter(f"{base}.tp{rank}{ext}", reader.alignment) for rank in range(tp)]
        for writer in writers:
            for key, vtype, raw in reader.kv:
                if key.startswith(PIPELINE_BRICK_KV_PREFIX):
                    continue
                writer.add_kv(key, vtype, raw)
        add_pipeline_brick_metadata(writers, tp, layout)

        n_output = 0
        n_input = 0
        n_copy = 0

        for index, tensor in enumerate(reader.tensors):
            raw = reader.read_tensor(tensor)
            kind = split_kind(tensor["name"])
            shape = list(tensor["shape"])

            if kind == "output":
                split_dim = 0
                n_output += 1
            elif kind == "input":
                split_dim = len(shape) - 1
                n_input += 1
            else:
                split_dim = -1
                n_copy += 1

            for rank, writer in enumerate(writers):
                if split_dim >= 0:
                    sizes = split_sizes(tensor["name"], shape, split_dim, tp, layout)
                    chunk = split_raw_tensor(raw, shape, tensor["dtype"], split_dim, rank, sizes)
                    new_shape = list(shape)
                    new_shape[split_dim] = sizes[rank]
                    new_dims = list(reversed(new_shape))
                else:
                    chunk = raw
                    new_dims = list(tensor["dims"])

                writer.add_tensor(tensor["name"], new_dims, tensor["dtype"], chunk)

            if (index + 1) % 50 == 0 or index + 1 == len(reader.tensors):
                print(f"[{index + 1}/{len(reader.tensors)}] output={n_output} input={n_input} copy={n_copy}")

        for writer in writers:
            writer.write()
            print(f"wrote {writer.path} ({os.path.getsize(writer.path) / (1024 * 1024):.1f} MiB)")

        print("done")
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description="Split Qwen3-4B GGUF for pipeline-brick TP")
    parser.add_argument("input", help="input F16/F32/BF16 GGUF")
    parser.add_argument("--tp", type=int, default=4, help="TP degree, 2 or 4")
    parser.add_argument("--layout", choices=("uniform", "rank0-small"), default="uniform",
                        help="TP split layout")
    args = parser.parse_args()

    try:
        split_model(args.input, args.tp, args.layout)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

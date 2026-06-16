#!/usr/bin/env python3
"""
gguf-split.py — QWEN3-4B 张量并行权重切分工具

将一个完整的 GGUF 文件按张量并行 (TP=4) 切分成 4 份。
每份只包含对应 rank 的权重大小。

切分规则：
  列并行 (split dim 1)：attn_q, attn_k, attn_v, ffn_gate, ffn_up
  行并行 (split dim 0)：attn_output, ffn_down
  全量复制：             所有 norm、embedding、output head

用法：
  python gguf-split.py qwen3-4b-F16.gguf --tp 4

输出：
  qwen3-4b-F16.tp0.gguf  (rank 0, attn_q[:, 0:640], ...)
  qwen3-4b-F16.tp1.gguf  (rank 1, attn_q[:, 640:1280], ...)
  qwen3-4b-F16.tp2.gguf  (rank 2)
  qwen3-4b-F16.tp3.gguf  (rank 3)

支持：F16, F32 类型。量化类型需先 dequantize 为 F16。

依赖：llama.cpp 自带的 gguf-py 模块
  export PYTHONPATH=llama.cpp/gguf-py:$PYTHONPATH
"""

import sys
import os
import struct
import re
import argparse
import numpy as np

# ---- 常量 ----
GGUF_MAGIC            = 0x46554747  # "GGUF"
GGUF_VERSION          = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF 值类型
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# 量化类型 (ggml_type) — 只需要区分可切分的
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_BF16 = 30
# 量化类型 (不可直接切分，需先 dequantize)
QUANT_TYPES = {2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,29}

# 按名称匹配切分策略
COLUMN_SPLIT_PATTERNS = [
    re.compile(r"blk\.\d+\.attn_q\.weight$"),
    re.compile(r"blk\.\d+\.attn_k\.weight$"),
    re.compile(r"blk\.\d+\.attn_v\.weight$"),
    re.compile(r"blk\.\d+\.ffn_gate\.weight$"),
    re.compile(r"blk\.\d+\.ffn_up\.weight$"),
]

ROW_SPLIT_PATTERNS = [
    re.compile(r"blk\.\d+\.attn_output\.weight$"),
    re.compile(r"blk\.\d+\.ffn_down\.weight$"),
]


class GGUFReader:
    """读取 GGUF 文件"""

    def __init__(self, path):
        self.f = open(path, 'rb')
        self._read_header()

    def _read_header(self):
        f = self.f
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic=0x{magic:08X})")

        self.version = struct.unpack("<I", f.read(4))[0]
        self.n_tensors = struct.unpack("<Q", f.read(8))[0]
        self.n_kv = struct.unpack("<Q", f.read(8))[0]

        # 读 KV 对
        self.kv = []
        for _ in range(self.n_kv):
            key = self._read_string()
            vtype = struct.unpack("<I", f.read(4))[0]
            self.kv.append((key, vtype, f.tell()))
            self._skip_value(vtype)

        # 读张量信息
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            self.tensors.append({
                'name': name,
                'dims': dims,        # GGUF order (内→外)
                'shape': tuple(reversed(dims)),  # numpy order (外→内)
                'dtype': dtype,
                'data_offset': offset,
            })

        # 对齐到数据区
        cur = f.tell()
        self.alignment = self._get_alignment()
        self.data_start = self._pad(cur, self.alignment)
        f.seek(self.data_start)

    def _read_string(self):
        f = self.f
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode('utf-8', errors='replace')

    def _skip_value(self, vtype):
        f = self.f
        if vtype in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
            f.read(1)
        elif vtype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
            f.read(2)
        elif vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
            f.read(4)
        elif vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
            f.read(8)
        elif vtype == GGUF_TYPE_STRING:
            f.read(struct.unpack("<Q", f.read(8))[0])
        elif vtype == GGUF_TYPE_ARRAY:
            etype = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                self._skip_value(etype)
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _get_alignment(self):
        """从 KV 中读 alignment，默认 32"""
        f = self.f
        pos = f.tell()
        for key, vtype, offset in self.kv:
            if key == "general.alignment":
                f.seek(offset)
                _ = struct.unpack("<I", f.read(4))[0]  # vtype
                val = struct.unpack("<I", f.read(4))[0]
                f.seek(pos)
                return val
        f.seek(pos)
        return GGUF_DEFAULT_ALIGNMENT

    def _pad(self, x, align):
        return (x + align - 1) // align * align

    def read_tensor_data(self, tensor_idx):
        """读取张量原始数据字节"""
        t = self.tensors[tensor_idx]
        f = self.f
        offset = self.data_start + t['data_offset']

        # 计算字节数
        n_elems = 1
        for d in t['dims']:
            n_elems *= d
        dtype = t['dtype']
        if dtype == GGML_TYPE_F32:
            n_bytes = n_elems * 4
        elif dtype == GGML_TYPE_F16:
            n_bytes = n_elems * 2
        elif dtype == GGML_TYPE_BF16:
            n_bytes = n_elems * 2
        elif dtype in QUANT_TYPES:
            # 量化类型——块大小因类型而异
            # 这里只做粗略估计，实际需要查 GGML_QUANT_SIZES 表
            raise RuntimeError(
                f"Quantized tensor '{t['name']}' (type={dtype}) "
                f"cannot be split. Convert to F16 first."
            )
        else:
            raise RuntimeError(
                f"Unknown tensor type {dtype} for '{t['name']}'"
            )

        f.seek(offset)
        data = f.read(n_bytes)
        return data, n_bytes

    def close(self):
        self.f.close()


class GGUFWriter:
    """写入 GGUF 文件"""

    def __init__(self, path, alignment=GGUF_DEFAULT_ALIGNMENT):
        self.f = open(path, 'wb')
        self.alignment = alignment
        self.kv_pairs = []           # (key, vtype, raw_value_bytes)
        self.tensor_infos = []       # (name, dims, dtype, offset, n_bytes_padded)
        self.tensor_data = b""       # 所有张量数据拼接
        self._written = False

    def add_kv_raw(self, key, vtype, raw_bytes):
        self.kv_pairs.append((key, vtype, raw_bytes))

    def add_tensor(self, name, dims, dtype, raw_data):
        """dims: GGUF 顺序（内→外）"""
        n_elems = 1
        for d in dims:
            n_elems *= d
        padded = self._pad_bytes(raw_data)
        # 计算偏移：累加之前所有张量的 padded 大小
        offset = len(self.tensor_data)
        self.tensor_infos.append((name, dims, dtype, offset, len(padded)))
        self.tensor_data += padded

    def _pad_bytes(self, data):
        pad = self._pad(len(data), self.alignment) - len(data)
        return data + b'\x00' * pad

    def _pad(self, x, align):
        return (x + align - 1) // align * align

    def write_all(self):
        """最终写入文件"""
        f = self.f

        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(self.tensor_infos)))
        f.write(struct.pack("<Q", len(self.kv_pairs)))

        # KV 对
        for key, vtype, raw_bytes in self.kv_pairs:
            key_bytes = key.encode('utf-8')
            f.write(struct.pack("<Q", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<I", vtype))
            f.write(raw_bytes)

        # 张量信息
        for name, dims, dtype, offset, _ in self.tensor_infos:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dtype))
            f.write(struct.pack("<Q", offset))

        # 对齐到数据区
        cur = f.tell()
        align_pad = self._pad(cur, self.alignment) - cur
        f.write(b'\x00' * align_pad)

        # 张量数据
        f.write(self.tensor_data)
        f.close()
        self._written = True


def is_column_split(name):
    """列并行：切最后 (最内) 维"""
    for pat in COLUMN_SPLIT_PATTERNS:
        if pat.search(name):
            return True
    return False

def is_row_split(name):
    """行并行：切最外维"""
    for pat in ROW_SPLIT_PATTERNS:
        if pat.search(name):
            return True
    return False


def split_tensor_f32(data, shape, split_dim, tp_rank, tp_size):
    """
    在指定维上切分 float32 张量。
    shape: numpy 顺序 (外→内)
    split_dim: numpy 索引 (0=最外, -1=最内)
    """
    size_on_dim = shape[split_dim]
    if size_on_dim % tp_size != 0:
        raise ValueError(
            f"Dimension {split_dim} size {size_on_dim} "
            f"not divisible by {tp_size}"
        )
    chunk = size_on_dim // tp_size
    start = tp_rank * chunk
    end = start + chunk

    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    slices = [slice(None)] * len(shape)
    slices[split_dim] = slice(start, end)
    return arr[tuple(slices)].tobytes()


def split_tensor_f16(data, shape, split_dim, tp_rank, tp_size):
    """同上，float16"""
    size_on_dim = shape[split_dim]
    if size_on_dim % tp_size != 0:
        raise ValueError(
            f"Dimension {split_dim} size {size_on_dim} "
            f"not divisible by {tp_size}"
        )
    chunk = size_on_dim // tp_size
    start = tp_rank * chunk
    end = start + chunk

    arr = np.frombuffer(data, dtype=np.float16).reshape(shape)
    slices = [slice(None)] * len(shape)
    slices[split_dim] = slice(start, end)
    return arr[tuple(slices)].tobytes()


def doit(args):
    src = GGUFReader(args.input)

    # 检查张量类型
    for t in src.tensors:
        if t['dtype'] not in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16):
            print(f"ERROR: quantized tensor '{t['name']}' (type={t['dtype']})")
            print("  Please convert to F16 first:")
            print(f"  python convert_hf_to_gguf.py ... --outtype f16")
            sys.exit(1)

    # 创建输出 writer
    base = os.path.splitext(args.input)[0]
    writers = []
    for r in range(args.tp):
        path = f"{base}.tp{r}.gguf"
        w = GGUFWriter(path, alignment=src.alignment)
        writers.append(w)
        print(f"  Output [{r}]: {path}")

    # 复制 KV 元数据（全量复制到每个输出文件）
    print(f"\nCopying {src.n_kv} KV pairs...")
    f = src.f
    for key, vtype, offset in src.kv:
        f.seek(offset)
        # 重新读 value 的原始字节
        # 先 skip type（已经知道 vtype）
        raw = b""
        if vtype in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
            raw = f.read(1)
        elif vtype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
            raw = f.read(2)
        elif vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
            raw = f.read(4)
        elif vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
            raw = f.read(8)
        elif vtype == GGUF_TYPE_STRING:
            length = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(length)
        elif vtype == GGUF_TYPE_ARRAY:
            # 数组：etype(4) + n(8) + 各元素
            etype = struct.unpack("<I", f.read(4))[0]
            n_elem = struct.unpack("<Q", f.read(8))[0]
            raw = struct.pack("<I", etype) + struct.pack("<Q", n_elem)
            for _ in range(n_elem):
                raw += _read_value_raw(f, etype)
        else:
            raise ValueError(f"Unknown vtype {vtype}")
        for w in writers:
            w.add_kv_raw(key, vtype, raw)

    # 处理张量
    print(f"\nSplitting {src.n_tensors} tensors (TP={args.tp})...")
    n_col, n_row, n_copy = 0, 0, 0

    for idx, t in enumerate(src.tensors):
        name = t['name']
        dims = t['dims']      # GGUF order (内→外)
        shape = t['shape']    # numpy order (外→内)
        dtype = t['dtype']
        data, n_bytes = src.read_tensor_data(idx)

        col = is_column_split(name)
        row = is_row_split(name)

        if col:
            # 列并行：切 GGUF 最内维 = numpy 最外维
            # 例：attn_q: GGUF dims=[2560, 2560], numpy shape=(2560, 2560)
            #     切 numpy dim 1 (= GGUF dim 0, 最内维)
            split_dim_np = len(shape) - 1  # numpy 最内维
            n_col += 1
        elif row:
            # 行并行：切 GGUF 最外维 = numpy dim 0
            # 例：attn_output: GGUF dims=[2560, 2560], numpy shape=(2560, 2560)
            #     切 numpy dim 0 (= GGUF dim 1, 最外维)
            split_dim_np = 0  # numpy 最外维
            n_row += 1
        else:
            # 全量复制
            split_dim_np = -1
            n_copy += 1

        for r in range(args.tp):
            if split_dim_np >= 0:
                # 切分
                if dtype == GGML_TYPE_F32:
                    chunk = split_tensor_f32(
                        data, shape, split_dim_np, r, args.tp)
                elif dtype in (GGML_TYPE_F16, GGML_TYPE_BF16):
                    chunk = split_tensor_f16(
                        data, shape, split_dim_np, r, args.tp)
                # 更新 shapes
                new_shape = list(shape)
                new_shape[split_dim_np] //= args.tp
                new_dims = list(reversed(new_shape))  # GGUF order
            else:
                chunk = data
                new_dims = list(dims)

            writers[r].add_tensor(name, new_dims, dtype, chunk)

        if (idx + 1) % 50 == 0 or idx == src.n_tensors - 1:
            print(f"  [{idx+1}/{src.n_tensors}] col={n_col} "
                  f"row={n_row} copy={n_copy}")

    # 写入
    print("\nWriting output files...")
    for r in range(args.tp):
        path = f"{base}.tp{r}.gguf"
        writers[r].write_all()
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{r}] {path}  ({size_mb:.1f} MB)")

    src.close()
    print("\nDone.")

    # 汇总
    print(f"\nSplit summary:")
    print(f"  Column-parallel (dim=1 split): {n_col} tensor types")
    print(f"  Row-parallel    (dim=0 split): {n_row} tensor types")
    print(f"  Full copy:                     {n_copy} tensor types")
    print(f"\nEach TP rank has 1/{args.tp} of attention/FFN weights.")
    print(f"To use with pipeline-brick:")
    print(f"  --model qwen3-4b-F16.tp0.gguf  (for NUMA 0)")
    print(f"  --model qwen3-4b-F16.tp1.gguf  (for NUMA 1)")
    print(f"  --model qwen3-4b-F16.tp2.gguf  (for NUMA 2)")
    print(f"  --model qwen3-4b-F16.tp3.gguf  (for NUMA 3)")


def _read_value_raw(f, vtype):
    """读一个 GGUF value 的原始字节"""
    if vtype == GGUF_TYPE_UINT8 or vtype == GGUF_TYPE_INT8 or vtype == GGUF_TYPE_BOOL:
        return f.read(1)
    elif vtype in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return f.read(2)
    elif vtype in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return f.read(4)
    elif vtype in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return f.read(8)
    elif vtype == GGUF_TYPE_STRING:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length)
    else:
        raise ValueError(f"Unknown vtype {vtype}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split GGUF model for tensor parallelism (QWEN3-4B)')
    parser.add_argument('input', help='Input GGUF file (F16 or F32)')
    parser.add_argument('--tp', type=int, default=4,
                        help='Tensor parallel degree (default: 4)')
    parser.add_argument('--verbose', action='store_true',
                        help='List every tensor being split')
    args = parser.parse_args()

    if args.tp not in (1, 2, 4, 8):
        print(f"Warning: TP={args.tp} not typical. Continue? [y/N] ", end='')
        if input().strip().lower() != 'y':
            sys.exit(0)

    doit(args)

import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import wave


DTYPES = {"f32": np.float32, "i16": np.int16, "u8": np.uint8}


def _convert_to_int16(block: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "f32":
        clipped = np.clip(block, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)
    if dtype_name == "i16":
        clipped = np.clip(block, -32768, 32767)
        return clipped.astype(np.int16)
    if dtype_name == "u8":
        centered = block.astype(np.float32) - 128.0
        scaled = centered * 256.0
        clipped = np.clip(scaled, -32768, 32767)
        return clipped.astype(np.int16)
    raise ValueError("unsupported dtype")


def _read_iq_pairs(raw: bytes, dtype: np.dtype) -> np.ndarray:
    if not raw:
        return np.empty((0, 2), dtype=np.float32)
    itemsize = np.dtype(dtype).itemsize
    frame_size = itemsize * 2
    n_frames = len(raw) // frame_size
    count = n_frames * 2
    arr = np.frombuffer(raw, dtype=dtype, count=count)
    return arr.reshape(-1, 2)


def convert_iq_file(
    input_path: Path,
    output_path: Path,
    rate: int,
    dtype_name: str,
    gain: float = 1.0,
    mono: bool = False,
    chunk_samples: int = 1024 * 1024,
) -> None:
    dtype = DTYPES[dtype_name]
    itemsize = np.dtype(dtype).itemsize
    frame_bytes = itemsize * 2
    size = input_path.stat().st_size
    if size % frame_bytes != 0:
        print("Warning: input size not a multiple of I/Q frame size; last partial frame will be ignored.", file=sys.stderr)

    nchannels = 1 if mono else 2

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(2)
        wf.setframerate(rate)

        with input_path.open("rb") as f:
            to_read = chunk_samples * frame_bytes
            while True:
                raw = f.read(to_read)
                if not raw:
                    break
                pairs = _read_iq_pairs(raw, dtype)
                if pairs.size == 0:
                    break
                if gain != 1.0:
                    pairs = pairs.astype(np.float32) * gain
                out16 = _convert_to_int16(pairs, dtype_name)
                if mono:
                    frames = out16[:, 0]
                    wf.writeframes(frames.tobytes())
                else:
                    wf.writeframes(out16.tobytes())


def _parse_args(argv: Tuple[str, ...]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw .iq to WAV (I->left, Q->right)")
    p.add_argument("-i", "--in", dest="infile", required=True)
    p.add_argument("-o", "--out", dest="outfile", required=True)
    p.add_argument("-r", "--rate", dest="rate", type=int, default=48000)
    p.add_argument("-d", "--dtype", dest="dtype", choices=DTYPES.keys(), default="f32")
    p.add_argument("--gain", dest="gain", type=float, default=1.0)
    p.add_argument("--mono", dest="mono", action="store_true")
    p.add_argument("--chunk", dest="chunk", type=int, default=1024 * 1024)
    return p.parse_args(list(argv))


def main(argv: Tuple[str, ...] = None) -> None:
    argv = tuple() if argv is None else tuple(argv)
    args = _parse_args(argv or tuple(sys.argv[1:]))
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists() or not infile.is_file():
        print("Input file not found.", file=sys.stderr)
        raise SystemExit(1)
    convert_iq_file(infile, outfile, args.rate, args.dtype, gain=args.gain, mono=args.mono, chunk_samples=args.chunk)


if __name__ == "__main__":
    main()

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


def convert_wav_file(
    input_path: Path,
    output_path: Path,
    dtype_name: str,
    gain: float = 1.0,
    chunk_frames: int = 1024 * 1024,
) -> None:
    dtype = DTYPES[dtype_name]

    with wave.open(str(input_path), "rb") as rf:
        nchannels = rf.getnchannels()
        sampwidth = rf.getsampwidth()
        framerate = rf.getframerate()

        with output_path.open("wb") as out:
            to_read = chunk_frames
            while True:
                raw = rf.readframes(to_read)
                if not raw:
                    break
                if sampwidth == 2:
                    in_arr = np.frombuffer(raw, dtype=np.int16)
                elif sampwidth == 1:
                    in_arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
                    in_arr = (in_arr * 256).astype(np.int16)
                elif sampwidth == 4:
                    in_arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                    in_arr = (in_arr / 2147483648.0 * 32767.0).astype(np.int16)
                else:
                    in_arr = np.frombuffer(raw, dtype=np.int16)

                if in_arr.size == 0:
                    break

                if nchannels > 1:
                    frames = in_arr.reshape(-1, nchannels)
                    if nchannels >= 2:
                        i_vals = frames[:, 0].astype(np.float32)
                        q_vals = frames[:, 1].astype(np.float32)
                    else:
                        i_vals = frames[:, 0].astype(np.float32)
                        q_vals = np.zeros_like(i_vals)
                else:
                    i_vals = in_arr.astype(np.float32)
                    q_vals = np.zeros_like(i_vals)

                if gain != 1.0:
                    i_vals = i_vals * gain
                    q_vals = q_vals * gain

                if dtype_name == "f32":
                    i_out = (i_vals / 32767.0).astype(np.float32)
                    q_out = (q_vals / 32767.0).astype(np.float32)
                    out_arr = np.empty((i_out.size * 2,), dtype=np.float32)
                    out_arr[0::2] = i_out
                    out_arr[1::2] = q_out
                    out.write(out_arr.tobytes())
                elif dtype_name == "i16":
                    i_out = np.clip(i_vals, -32768, 32767).astype(np.int16)
                    q_out = np.clip(q_vals, -32768, 32767).astype(np.int16)
                    out_arr = np.empty((i_out.size * 2,), dtype=np.int16)
                    out_arr[0::2] = i_out
                    out_arr[1::2] = q_out
                    out.write(out_arr.tobytes())
                elif dtype_name == "u8":
                    i_u8 = np.clip((i_vals.astype(np.float32) / 256.0) + 128.0, 0, 255).astype(np.uint8)
                    q_u8 = np.clip((q_vals.astype(np.float32) / 256.0) + 128.0, 0, 255).astype(np.uint8)
                    out_arr = np.empty((i_u8.size * 2,), dtype=np.uint8)
                    out_arr[0::2] = i_u8
                    out_arr[1::2] = q_u8
                    out.write(out_arr.tobytes())
                else:
                    raise ValueError("unsupported dtype")


def _parse_args(argv: Tuple[str, ...]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw .iq to WAV (I->left, Q->right) or WAV to raw .iq")
    p.add_argument("-i", "--in", dest="infile", required=True)
    p.add_argument("-o", "--out", dest="outfile", required=True)
    p.add_argument("-r", "--rate", dest="rate", type=int, default=48000)
    p.add_argument("-d", "--dtype", dest="dtype", choices=DTYPES.keys(), default="f32")
    p.add_argument("--gain", dest="gain", type=float, default=1.0)
    p.add_argument("--mono", dest="mono", action="store_true")
    p.add_argument("--chunk", dest="chunk", type=int, default=1024 * 1024)
    p.add_argument("--wav2iq", dest="wav2iq", action="store_true", help="Convert WAV input to raw interleaved IQ output (reverse)")
    return p.parse_args(list(argv))


def main(argv: Tuple[str, ...] = None) -> None:
    argv = tuple() if argv is None else tuple(argv)
    args = _parse_args(argv or tuple(sys.argv[1:]))
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists() or not infile.is_file():
        print("Input file not found.", file=sys.stderr)
        raise SystemExit(1)
    if args.wav2iq:
        convert_wav_file(infile, outfile, args.dtype, gain=args.gain, chunk_frames=args.chunk)
    else:
        convert_iq_file(infile, outfile, args.rate, args.dtype, gain=args.gain, mono=args.mono, chunk_samples=args.chunk)


if __name__ == "__main__":
    main()

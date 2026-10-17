"""
CLI video enhancer (tile/patch based) using an ONNX-exported FUnIE-GAN generator (CPU only).

Instead of resizing the whole frame to 256x256 (which destroys detail), this splits
each frame into overlapping 256x256 tiles, enhances each tile independently, and
blends them back with separable linear cross-fades that sum to exactly 1.0
(no seams / dark bands).

Usage (example):
  python cli/enhance_cli_tiles.py --input input.mp4 --output enhanced.mp4 \
      --onnx model/funie.onnx --batch-size 8 --threads 4 --overlap 32 --reencode

Dependencies:
  pip install onnxruntime opencv-python-headless numpy
"""
import argparse
import os
import subprocess
import sys
import time
import uuid
from typing import List

import numpy as np
import cv2
import onnxruntime as ort

TILE = 256


def fatal(msg, code=1):
    print("ERROR:", msg)
    sys.exit(code)


def load_session(onnx_path: str, threads: int = 4):
    if not os.path.exists(onnx_path):
        fatal(f"ONNX model not found at: {onnx_path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    try:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    except Exception as e:
        fatal(f"Failed to create ONNX Runtime session: {e}")
    return sess


def get_input_name(sess):
    names = [i.name for i in sess.get_inputs()]
    return "input" if "input" in names else names[0]


def preprocess_batch(tiles: List[np.ndarray]) -> np.ndarray:
    """BGR uint8 256x256 tiles -> (N,3,256,256) float32 in [-1,1]."""
    out = []
    for f in tiles:
        im = cv2.resize(f, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 127.5 - 1.0
        out.append(np.transpose(im, (2, 0, 1)))
    return np.stack(out, axis=0).astype(np.float32)


def postprocess_one(out: np.ndarray, shape: tuple) -> np.ndarray:
    """(3,256,256) in [-1,1] -> BGR uint8 at the tile's original size."""
    im = np.transpose(out, (1, 2, 0))
    im = ((im + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    h, w = shape
    if (h, w) != (TILE, TILE):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
    return im


def tile_starts(length: int, tile: int, stride: int) -> List[int]:
    """Start offsets covering [0, length) fully with `tile`-sized tiles.

    Starts are evenly spaced so every delta is identical, keeping overlaps
    uniform and avoiding 3-way overlap bands that a clamped final start causes.
    """

    if length <= tile:
        return [0]
    n = int(np.ceil((length - tile) / stride)) + 1
    if n < 2:
        return [0]
    step = (length - tile) / (n - 1)
    return [int(round(i * step)) for i in range(n)]


def axis_weights(starts: List[int], tile: int, index: int) -> np.ndarray:
    """
    Separable linear cross-fade weights for tile `index`.
    Uses the true geometric overlap (tile - start_delta), so adjacent tiles
    sum to exactly 1.0 across the shared region.
    """
    n = len(starts)
    w = np.ones(tile, dtype=np.float32)
    if index > 0:
        ov = tile - (starts[index] - starts[index - 1])
        if ov > 0:
            w[:ov] = np.linspace(0.0, 1.0, ov + 1)[1:]
    if index < n - 1:
        ov = tile - (starts[index + 1] - starts[index])
        if ov > 0:
            w[tile - ov:] = np.linspace(1.0, 0.0, ov + 1)[1:]
    return w


def process_video(input_path, output_path, sess, batch_size, overlap, input_name):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        fatal(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap_fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(cap_fc) if cap_fc else None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        fatal(f"Invalid input dimensions: {width}x{height}")

    overlap = max(0, min(overlap, TILE - 1))
    stride = TILE - overlap

    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
    if not out_writer.isOpened():
        cap.release()
        fatal(f"Cannot open VideoWriter for output: {output_path}")

    xs = tile_starts(width, TILE, stride)
    ys = tile_starts(height, TILE, stride)
    wx = [axis_weights(xs, TILE, i) for i in range(len(xs))]
    wy = [axis_weights(ys, TILE, j) for j in range(len(ys))]

    print(f"Input: {input_path} | FPS: {fps:.2f} | Size: {width}x{height} | "
          f"Frames: {total_frames}")
    print(f"Output: {output_path} | Tile: {TILE} | Overlap: {overlap} | "
          f"Stride: {stride} | Tiles/frame: {len(xs) * len(ys)} | "
          f"Batch: {batch_size}")

    pad_h = max(0, TILE - height)
    pad_w = max(0, TILE - width)

    processed = 0
    start_time = time.time()

    def accumulate(acc, wsum, outputs, coords, shapes, weights):
        for idx, ((y, x), (th, tw), wgt) in enumerate(
                zip(coords, shapes, weights)):
            tile = postprocess_one(outputs[idx], (th, tw))
            w = wgt[:th, :tw, None]
            acc[y:y + th, x:x + tw] += tile.astype(np.float32) * w
            wsum[y:y + th, x:x + tw] += w

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            work = (cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_REFLECT_101)
                    if (pad_h or pad_w) else frame)

            acc = np.zeros((height, width, 3), dtype=np.float32)
            wsum = np.zeros((height, width, 1), dtype=np.float32)

            batch, coords, shapes, weights = [], [], [], []
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    th, tw = min(TILE, height - y), min(TILE, width - x)
                    crop = work[y:y + th, x:x + tw]
                    if crop.shape[0] != TILE or crop.shape[1] != TILE:
                        crop = cv2.resize(crop, (TILE, TILE),
                                          interpolation=cv2.INTER_CUBIC)
                    batch.append(crop)
                    coords.append((y, x))
                    shapes.append((th, tw))
                    weights.append(np.outer(wy[j], wx[i]))

                    if len(batch) >= batch_size:
                        outputs = sess.run(None, {input_name:
 preprocess_batch(batch)})[0]
                        accumulate(acc, wsum, outputs, coords, shapes, weights)
                        batch, coords, shapes, weights = [], [], [], []

            if batch:
                outputs = sess.run(None, {input_name:
                                          preprocess_batch(batch)})[0]
                accumulate(acc, wsum, outputs, coords, shapes, weights)

            blended = acc / np.maximum(wsum, 1e-6)
            out_writer.write(blended.clip(0, 255).astype(np.uint8))

            processed += 1
            if processed % 10 == 0 or processed == 1:
                el = time.time() - start_time
                rate = processed / el if el > 0 else 0
                base = (f"Processed {processed}/{total_frames} "
                        f"({(processed / total_frames) * 100:.1f}%)"
                        if total_frames else f"Processed {processed} frames")
                print(f"{base} | {rate:.2f} fps", flush=True)

    except Exception as e:
        cap.release()
        out_writer.release()
        fatal(f"Processing failed: {e}")

    cap.release()
    out_writer.release()
    elapsed = time.time() - start_time
    fps_eff = processed / elapsed if elapsed > 0 else 0
    print(f"Done. {processed} frames in {elapsed:.2f}s ({fps_eff:.2f} fps).")
    print(f"Output: {output_path}")


def reencode_h264(src, dst, crf=18):
    """mp4v is very lossy; H.264 CRF 18 looks markedly better."""
    cmd = ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-crf", str(crf),
           "-preset", "slow", "-pix_fmt", "yuv420p", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        print(f"Re-encoded: {dst}")
    except Exception as e:
        print(f"WARNING: re-encode failed ({e}); keeping {src}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Tile-based video enhancement with ONNX FUnIE-GAN (CPU)")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", default=None,
                   help="Output path (default: enhanced-<uuid>.mp4)")
    p.add_argument("--onnx", default="model/funie.onnx", help="Path to ONNX model")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Tiles per inference batch")
    p.add_argument("--threads", type=int, default=4,
                   help="ONNX Runtime intra_op_num_threads")
    p.add_argument("--overlap", type=int, default=32,
                   help="Tile overlap in pixels (0-255)")
    p.add_argument("--reencode", action="store_true",
                   help="Re-encode result to H.264 CRF 18 for better quality")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.output or f"enhanced-{uuid.uuid4().hex}.mp4"

    if not os.path.exists(args.input):
        fatal(f"Input video not found: {args.input}")
    if args.batch_size < 1:
        fatal("--batch-size must be >= 1")
    if args.threads < 1:
        fatal("--threads must be >= 1")

    sess = load_session(args.onnx, threads=args.threads)
    input_name = get_input_name(sess)
    try:
        process_video(args.input, out_path, sess, args.batch_size,
                      args.overlap, input_name)
    except SystemExit:
        raise
    except Exception as e:
        fatal(f"Processing failed: {e}")

    final = out_path
    if args.reencode:
        final = out_path.rsplit(".", 1)[0] + "_h264.mp4"
        reencode_h264(out_path, final)

    print("Success:", final)
    sys.exit(0)


if __name__ == "__main__":
    main()

"""
CLI video enhancer using an ONNX-exported FUnIE-GAN generator (CPU only).

Usage (example):
  python cli/enhance_cli.py --input input.mp4 --output enhanced.mp4 --onnx model/funie.onnx --batch-size 8 --threads 4

Dependencies:
  pip install onnxruntime opencv-python-headless numpy

Behavior summary:
  - Loads ONNX model via onnxruntime (CPUExecutionProvider).
  - Reads input video, preserves fps and original frame size.
  - Processes frames in batches:
      * resize to 256x256 (bicubic),
      * BGR -> RGB,
      * normalize to [-1,1] via img/127.5 - 1.0,
      * arrange as NCHW float32,
      * run sess.run,
      * postprocess: (x+1)*127.5 -> uint8, RGB->BGR, resize back to original,
  - Writes enhanced video via OpenCV VideoWriter using mp4v fourcc.
  - Prints progress and exits with non-zero code on errors.
"""
import argparse
import os
import sys
import uuid
import math
import time
from typing import List

import numpy as np
import cv2
import onnxruntime as ort


def fatal(msg, code=1):
    print("ERROR:", msg)
    sys.exit(code)


def load_session(onnx_path: str, threads: int = 4):
    if not os.path.exists(onnx_path):
        fatal(f"ONNX model not found at: {onnx_path}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    # enable graph optimization
    try:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    except Exception as e:
        fatal(f"Failed to create ONNX Runtime session: {e}")
    return sess


def preprocess_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    frames: list of BGR uint8 images
    returns: numpy float32 array of shape (N,3,256,256)
    """
    out = []
    for f in frames:
        # resize to 256x256 with bicubic
        im = cv2.resize(f, (256, 256), interpolation=cv2.INTER_CUBIC)
        # BGR -> RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # to float32, normalize to [-1,1]
        im = im.astype(np.float32) / 127.5 - 1.0
        # HWC -> CHW
        im = np.transpose(im, (2, 0, 1))
        out.append(im)
    batch = np.stack(out, axis=0).astype(np.float32)
    return batch


def postprocess_batch(outputs: np.ndarray, orig_shapes: List[tuple]) -> List[np.ndarray]:
    """
    outputs: numpy array (N,3,256,256) in range [-1,1]
    orig_shapes: list of (h, w) for each frame
    returns: list of BGR uint8 images resized back to orig_shapes
    """
    results = []
    for i in range(outputs.shape[0]):
        out = outputs[i]
        # CHW -> HWC
        im = np.transpose(out, (1, 2, 0))
        # [-1,1] -> [0,255]
        im = ((im + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        # RGB -> BGR
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        h, w = orig_shapes[i]
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
        results.append(im)
    return results


def process_video_local(input_path: str, output_path: str, sess: ort.InferenceSession, batch_size: int = 8):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        fatal(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out_writer.isOpened():
        cap.release()
        fatal(f"Cannot open VideoWriter for output: {output_path}")

    print(f"Input: {input_path} | FPS: {fps:.2f} | Size: {width}x{height} | Total frames: {total_frames}")
    print(f"Output: {output_path} | Batch size: {batch_size}")

    frames_buffer = []
    shapes_buffer = []
    processed = 0
    start_time = time.time()

    # determine ONNX input name (use "input" if present, else first input)
    try:
        input_name = "input"
        # verify if input exists; fallback:
        names = [i.name for i in sess.get_inputs()]
        if input_name not in names:
            input_name = names[0]
    except Exception:
        input_name = sess.get_inputs()[0].name

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            shapes_buffer.append(frame.shape[:2])  # (h, w)
            frames_buffer.append(frame)
            if len(frames_buffer) >= batch_size:
                batch = preprocess_frames(frames_buffer)
                # run inference
                outputs = sess.run(None, {input_name: batch})[0]
                processed_batch = postprocess_batch(outputs, shapes_buffer)
                for f in processed_batch:
                    out_writer.write(f)
                processed += len(frames_buffer)
                # progress
                if total_frames:
                    pct = (processed / total_frames) * 100
                    print(f"Processed {processed}/{total_frames} frames ({pct:.1f}%)")
                else:
                    print(f"Processed {processed} frames")
                frames_buffer = []
                shapes_buffer = []
        # process leftover
        if frames_buffer:
            batch = preprocess_frames(frames_buffer)
            outputs = sess.run(None, {input_name: batch})[0]
            processed_batch = postprocess_batch(outputs, shapes_buffer)
            for f in processed_batch:
                out_writer.write(f)
            processed += len(frames_buffer)
            print(f"Processed {processed} frames (final)")

    except Exception as e:
        cap.release()
        out_writer.release()
        fatal(f"Inference/processing failed: {e}")

    cap.release()
    out_writer.release()
    elapsed = time.time() - start_time
    fps_eff = processed / elapsed if elapsed > 0 else 0
    print(f"Done. Processed {processed} frames in {elapsed:.2f}s ({fps_eff:.2f} fps). Output saved to: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Enhance video using ONNX-exported FUnIE-GAN (CPU)")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", default=None, help="Output video path (default: enhanced-<uuid>.mp4)")
    p.add_argument("--onnx", default="model/funie.onnx", help="Path to ONNX model")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for frame inference")
    p.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra_op_num_threads")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output:
        out_path = args.output
    else:
        out_path = f"enhanced-{uuid.uuid4().hex}.mp4"

    if not os.path.exists(args.input):
        fatal(f"Input video not found: {args.input}")

    sess = load_session(args.onnx, threads=args.threads)
    try:
        process_video_local(args.input, out_path, sess, batch_size=args.batch_size)
    except Exception as e:
        fatal(f"Processing failed: {e}")

    # success
    print("Success:", out_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
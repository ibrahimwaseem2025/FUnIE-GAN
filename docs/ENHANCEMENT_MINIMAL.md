# FUnIE-GAN Enhancement Minimal

Short usage instructions for exporting the model to ONNX and enhancing videos.

## Export ONNX

```bash
python tools/onnx_export.py --model-path PyTorch/models/funie_generator.pth --out model/funie.onnx
```

## Enhance Video

```bash
python cli/enhance_cli.py --input sample/input.mp4 --output sample/enhanced.mp4 --onnx model/funie.onnx --batch-size 8 --threads 4
```

## Acceptance Test

1. Export the ONNX model as shown above.
2. Run the CLI enhancer on a short sample video (e.g., `istockphoto_video_2.mp4`).
3. Verify the output file exists and is readable (e.g., `ffprobe` the output mp4).

## Options

- `--threads` (default: `4`): ONNX Runtime `intra_op_num_threads`.
- `--batch-size` (default: `8`): Number of frames processed per inference batch.
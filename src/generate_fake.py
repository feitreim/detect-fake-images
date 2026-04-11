"""Generate fake videos from real-clip captions using an open-weight T2V model.

Intended to run on a rented H100 (RunPod / Lambda). Resume-safe: skips any
clip_id whose output mp4 already exists. Writes data/fake_manifest.parquet.

Usage:
    pip install -e '.[gen]'
    python -m src.generate_fake \
        --real-manifest data/real_manifest.parquet \
        --out-dir data/fake \
        --backend hunyuan
"""
import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd
import torch


class HunyuanBackend:
    name = "hunyuan"
    default_model_id = "hunyuanvideo-community/HunyuanVideo"

    def __init__(self, model_id, dtype=torch.bfloat16):
        from diffusers import HunyuanVideoPipeline
        self.model_id = model_id
        self.pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.enable_model_cpu_offload()

    def __call__(self, prompt, num_frames, height, width, seed, num_steps, guidance):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=gen,
        )
        return out.frames[0]  # list/array of PIL images


class LTXBackend:
    name = "ltx"
    default_model_id = "Lightricks/LTX-Video"

    def __init__(self, model_id, dtype=torch.bfloat16):
        from diffusers import LTXPipeline
        self.model_id = model_id
        self.pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.enable_model_cpu_offload()

    def __call__(self, prompt, num_frames, height, width, seed, num_steps, guidance):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=gen,
        )
        return out.frames[0]


BACKENDS = {"hunyuan": HunyuanBackend, "ltx": LTXBackend}


def save_video(frames, out_path, fps):
    import av
    import numpy as np
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    first = np.asarray(frames[0])
    stream.width = first.shape[1]
    stream.height = first.shape[0]
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (arr.clip(0, 1) * 255).astype("uint8")
        vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def seed_for(clip_id, base):
    h = int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-manifest", default="data/real_manifest.parquet")
    p.add_argument("--out-dir", default="data/fake")
    p.add_argument("--manifest", default="data/fake_manifest.parquet")
    p.add_argument("--backend", choices=list(BACKENDS), default="hunyuan")
    p.add_argument("--model-id", default=None)
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None)
    args = p.parse_args()

    real = pd.read_parquet(args.real_manifest)
    if args.max_clips:
        real = real.head(args.max_clips)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    BackendCls = BACKENDS[args.backend]
    backend = BackendCls(args.model_id or BackendCls.default_model_id)

    rows = []
    start = time.time()
    for i, row in real.reset_index(drop=True).iterrows():
        real_id = row["clip_id"]
        caption = str(row["caption"]).strip()
        if not caption:
            continue
        fake_id = real_id.replace("real/", "fake/")
        out_path = out_dir / (fake_id.split("/")[-1] + ".mp4")
        seed = seed_for(real_id, args.seed_base)

        if out_path.exists():
            rows.append(_row(fake_id, out_path, caption, backend, args, seed))
            continue

        t0 = time.time()
        try:
            frames = backend(
                prompt=caption,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                seed=seed,
                num_steps=args.num_steps,
                guidance=args.guidance,
            )
            save_video(frames, out_path, args.fps)
        except Exception as e:
            print(f"[{i}] {real_id} failed: {e}")
            continue

        dt = time.time() - t0
        total = time.time() - start
        print(f"[{i+1}/{len(real)}] {real_id} -> {out_path.name}  {dt:.1f}s  (avg {total/(i+1):.1f}s)")
        rows.append(_row(fake_id, out_path, caption, backend, args, seed))

        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_parquet(args.manifest, index=False)

    pd.DataFrame(rows).to_parquet(args.manifest, index=False)
    print(f"done: {len(rows)} clips → {args.manifest}")


def _row(clip_id, mp4_path, caption, backend, args, seed):
    return {
        "clip_id": clip_id,
        "mp4_path": str(Path(mp4_path).resolve()),
        "caption": caption,
        "model_name": backend.name,
        "model_version": backend.model_id,
        "seed": int(seed),
        "guidance": float(args.guidance),
        "num_steps": int(args.num_steps),
        "num_frames": int(args.num_frames),
        "width": int(args.width),
        "height": int(args.height),
    }


if __name__ == "__main__":
    main()

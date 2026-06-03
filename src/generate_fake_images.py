"""Generate fake images from real captions, on a rented GPU.

For each real image's caption we synthesize one image with an open-weight
text-to-image model. This pairs by *content distribution* (same captions →
same scene types) rather than pixels, which is enough because every detector
feature is a single-crop statistic.

Outputs are saved as lossless PNG — the canonical source of truth. The codec
match (JPEG re-encode to mirror the real photos) happens later in
preprocess_images.py, so we can also run the raw-PNG ablation from the same files.

Backends (--backend):
    klein-9b   black-forest-labs/FLUX.2-klein-9B   (default, ~29GB bf16, 4 steps)
    klein-4b   black-forest-labs/FLUX.2-klein-4B   (~14GB, Apache-2.0, 4 steps)
    sdxl-turbo stabilityai/sdxl-turbo              (light fallback, 1 step)

Usage (on the GPU box):
    pip install "git+https://github.com/huggingface/diffusers.git" transformers accelerate safetensors sentencepiece
    python -m src.generate_fake_images --max-images 20      # dry run
    python -m src.generate_fake_images                      # full run
"""
import argparse
import hashlib
import time
import traceback
from pathlib import Path

import pandas as pd


def seed_for(image_id, base):
    h = int(hashlib.md5(str(image_id).encode()).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


class Backend:
    """A text-to-image model. Subclasses build the pipeline and define defaults."""

    model_name = ""
    model_version = ""
    default_steps = 4
    default_guidance = 1.0

    def __init__(self, device, dtype, cpu_offload):
        self.device = device
        self.dtype = dtype
        self.pipe = self.build(cpu_offload)

    def build(self, cpu_offload):
        raise NotImplementedError

    def generate(self, prompt, seed, height, width, steps, guidance):
        import torch
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        ).images[0]
        return image


class FluxKlein(Backend):
    default_steps = 4
    default_guidance = 1.0

    def __init__(self, repo, device, dtype, cpu_offload):
        self.repo = repo
        self.model_name = repo.split("/")[-1].lower()
        self.model_version = repo
        super().__init__(device, dtype, cpu_offload)

    def build(self, cpu_offload):
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(self.repo, torch_dtype=self.dtype)
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe


class SDXLTurbo(Backend):
    model_name = "sdxl-turbo"
    model_version = "stabilityai/sdxl-turbo"
    default_steps = 1
    default_guidance = 0.0

    def build(self, cpu_offload):
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_version, torch_dtype=self.dtype, variant="fp16")
        pipe.to(self.device)
        return pipe


def make_backend(name, device, dtype, cpu_offload):
    if name == "klein-9b":
        return FluxKlein("black-forest-labs/FLUX.2-klein-9B", device, dtype, cpu_offload)
    if name == "klein-4b":
        return FluxKlein("black-forest-labs/FLUX.2-klein-4B", device, dtype, cpu_offload)
    if name == "sdxl-turbo":
        return SDXLTurbo(device, dtype, cpu_offload)
    raise ValueError(f"unknown backend {name}")


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="klein-9b", choices=["klein-9b", "klein-4b", "sdxl-turbo"])
    p.add_argument("--real-manifest", default="data/real_images_manifest.parquet")
    p.add_argument("--out-dir", default="data/fake_images")
    p.add_argument("--manifest", default="data/fake_images_manifest.parquet")
    p.add_argument("--resolution", type=int, default=768,
                   help="768 keeps the fake->256 downscale ratio (3x) close to the reals' (~2.5x)")
    p.add_argument("--steps", type=int, default=None, help="override backend default")
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--cpu-offload", action="store_true", help="fit on <24GB cards")
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_parquet(args.real_manifest)

    existing, rows = set(), []
    if Path(args.manifest).exists():
        prev = pd.read_parquet(args.manifest)
        if len(prev):
            existing = set(prev["image_id"].tolist())
            rows = prev.to_dict("records")
            print(f"resuming: {len(existing)} already done")

    todo = [row for _, row in real.iterrows() if int(row["image_id"]) not in existing]
    if args.max_images:
        todo = todo[: args.max_images]
    print(f"generating {len(todo)} images ({len(existing)} done) with {args.backend}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = make_backend(args.backend, device, torch.bfloat16, args.cpu_offload)
    steps = args.steps or backend.default_steps
    guidance = args.guidance if args.guidance is not None else backend.default_guidance
    print(f"backend={backend.model_name} steps={steps} guidance={guidance} res={args.resolution}")

    start, done, fails = time.time(), 0, 0
    for row in todo:
        image_id = int(row["image_id"])
        caption = str(row.get("caption", "")).strip() or "a photograph"
        out_path = out_dir / f"{image_id}.png"
        seed = seed_for(image_id, args.seed_base)
        try:
            if not out_path.exists():
                image = backend.generate(caption, seed, args.resolution, args.resolution, steps, guidance)
                image.save(out_path)
            rows.append({
                "image_id": image_id,
                "image_path": str(out_path.resolve()),
                "caption": caption,
                "model_name": backend.model_name,
                "model_version": backend.model_version,
                "seed": int(seed),
                "steps": int(steps),
                "guidance": float(guidance),
                "width": args.resolution,
                "height": args.resolution,
            })
            done += 1
        except Exception as e:
            fails += 1
            if fails <= 10:
                traceback.print_exc()
            print(f"[fail {fails}] image {image_id}: {e}")
            continue

        if done % 10 == 0:
            rate = done / (time.time() - start)
            eta = (len(todo) - done) / rate / 60 if rate else 0
            print(f"[{done}/{len(todo)}] {rate:.2f} img/s  eta {eta:.0f}m  fails={fails}")
        if done % args.save_every == 0:
            pd.DataFrame(rows).to_parquet(args.manifest, index=False)

    pd.DataFrame(rows).to_parquet(args.manifest, index=False)
    print(f"done: {len(rows)} images → {args.manifest}  (fails: {fails})")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm


def caption_images_real(image_paths: Sequence[Path], model_name: str) -> List[str]:
    """Generate captions for images using the Florence-2 captioning model."""
    import random

    import numpy as np
    import torch
    from PIL import Image, UnidentifiedImageError
    from transformers import AutoModelForCausalLM, AutoProcessor

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    captions: List[str] = []

    for image_path in tqdm(image_paths, desc="Generating image captions"):
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                inputs = processor(
                    text="<MORE_DETAILED_CAPTION>",
                    images=image,
                    return_tensors="pt",
                )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    do_sample=False,
                )

            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )

            caption = (
                output_text[0].strip()
                if output_text and output_text[0].strip()
                else f"Image file named {image_path.stem}"
            )
            captions.append(caption)

        except UnidentifiedImageError:
            print(f"WARNING: Cannot read image: {image_path}")
            captions.append(f"Corrupted image {image_path.stem}")

        except FileNotFoundError:
            print(f"WARNING: File not found: {image_path}")
            captions.append(f"Missing image {image_path.stem}")

        except Exception as ex:
            print(f"WARNING: Failed processing {image_path}")
            print(f"Reason: {ex}")
            captions.append(f"Failed image {image_path.stem}")

    return captions


def caption_images_mock(image_paths: Sequence[Path]) -> List[str]:
    """Generate deterministic placeholder captions for offline or smoke-test runs."""
    captions: List[str] = []
    for image_path in tqdm(image_paths, desc="Generating mock captions"):
        stem = image_path.stem.replace("_", " ").replace("-", " ")
        captions.append(f"Mock caption describing {stem}".strip())
    return captions


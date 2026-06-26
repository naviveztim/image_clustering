from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from tqdm import tqdm


def caption_images_real(image_paths: Sequence[Path], model_name: str) -> List[str]:
    """Generate captions for images using the Florence-2 or similar captioning model."""
    # Keep imports local so environments that only use other modules do not pay
    # heavy startup costs or require optional ML dependencies.
    import random

    import numpy as np
    import torch
    from PIL import Image, UnidentifiedImageError
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Seed all major random sources to make caption generation as reproducible
    # as possible across runs and hardware.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the caption model (local path or HF model id) and move it to the
    # best available device.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load the paired processor that prepares both text prompt and image tensor
    # inputs in the format expected by the model.
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    captions: List[str] = []

    # Process images one by one so failures are isolated and a single bad file
    # does not stop the full batch.
    for image_path in tqdm(image_paths, desc="Generating image captions"):
        try:
            # Open image safely and normalize to RGB for consistent model input.
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                # Build model inputs from the captioning prompt and image.
                inputs = processor(
                    text="<MORE_DETAILED_CAPTION>",
                    images=image,
                    return_tensors="pt",
                )

            # Move all tensors to the same device as the model.
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            # Run deterministic inference (no gradients, no sampling) to
            # generate a caption token sequence.
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    do_sample=False,
                )

            # Decode model output and fall back to a stable placeholder if the
            # decoded text is unexpectedly empty.
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

        # Handle common file-level issues with explicit fallback captions.
        except UnidentifiedImageError:
            print(f"WARNING: Cannot read image: {image_path}")
            captions.append(f"Corrupted image {image_path.stem}")

        except FileNotFoundError:
            print(f"WARNING: File not found: {image_path}")
            captions.append(f"Missing image {image_path.stem}")

        # Keep the pipeline resilient to model/runtime errors on individual
        # files and continue processing the rest.
        except Exception as ex:
            print(f"WARNING: Failed processing {image_path}")
            print(f"Reason: {ex}")
            captions.append(f"Failed image {image_path.stem}")

    return captions

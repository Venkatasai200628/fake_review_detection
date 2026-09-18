"""
clip_embedding.py — real CLIP image encoder, replacing the placeholder descriptor.

RUN THIS LOCALLY. It cannot run in the sandbox where the dataset was built:
huggingface.co, openaipublic.azureedge.net and download.pytorch.org are all
blocked there, so no pretrained weights can be fetched.

Install (one of):
    pip install open_clip_torch torch torchvision
    pip install transformers torch torchvision      # fallback path

Model: ViT-B/32, laion2b_s34b_b79k. ~350MB of weights, runs fine on an
RTX 4060 8GB. Batch inference over ~2000 images at 1024px takes a few minutes.

Design note: the returned vector is L2-normalised, so cosine similarity is a
plain dot product and every downstream threshold keeps the same meaning. The
only thing that changes is dimensionality (512 instead of 64) and the
DISTRIBUTION of similarities -- see tune_thresholds.py, which must be run after
this.
"""

import numpy as np
from PIL import Image

_MODEL = None
_PREPROCESS = None
_DEVICE = None
_BACKEND = None


def _init():
    """Load CLIP once. Tries open_clip first, then transformers."""
    global _MODEL, _PREPROCESS, _DEVICE, _BACKEND
    if _MODEL is not None:
        return

    import torch
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to(_DEVICE).eval()
        _MODEL, _PREPROCESS, _BACKEND = model, preprocess, 'open_clip'
        print(f"[clip] open_clip ViT-B-32 laion2b on {_DEVICE}")
        return
    except ImportError:
        pass

    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    _MODEL, _PREPROCESS = model.to(_DEVICE).eval(), proc
    _BACKEND = 'transformers'
    print(f"[clip] transformers clip-vit-base-patch32 on {_DEVICE}")


def encode_images(pil_images, batch_size=32):
    """Encode a list of PIL images. Returns (n, 512) L2-normalised float32."""
    import torch
    _init()
    out = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i + batch_size]
        with torch.no_grad():
            if _BACKEND == 'open_clip':
                x = torch.stack([_PREPROCESS(im) for im in batch]).to(_DEVICE)
                v = _MODEL.encode_image(x)
            else:
                x = _PREPROCESS(images=batch, return_tensors='pt').to(_DEVICE)
                v = _MODEL.get_image_features(**x)
        v = v.float().cpu().numpy()
        out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
    return np.vstack(out)


def compute_embedding(im, dim=None):
    """
    Drop-in replacement for gen_images.compute_embedding().
    Single-image path -- prefer encode_images() for bulk work, it is far faster.
    """
    return encode_images([im])[0]


def encode_text(texts, batch_size=64):
    """
    CLIP text encoder, in the SAME embedding space as encode_images().

    This unlocks a feature the pipeline does not currently have at all: a
    cross-modal consistency score, i.e. does the review text actually describe
    the attached photograph. Several published multimodal detectors build their
    contribution on exactly this signal. See tune_thresholds.py --cross-modal.
    """
    import torch
    _init()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            if _BACKEND == 'open_clip':
                import open_clip
                tok = open_clip.tokenize(batch).to(_DEVICE)
                v = _MODEL.encode_text(tok)
            else:
                x = _PREPROCESS(text=batch, return_tensors='pt',
                                padding=True, truncation=True).to(_DEVICE)
                v = _MODEL.get_text_features(**x)
        v = v.float().cpu().numpy()
        out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))
    return np.vstack(out)

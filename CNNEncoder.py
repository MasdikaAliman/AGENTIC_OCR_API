import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np


class CNNEncoder:
    """
    CNN-based image encoder using pretrained VGG16 or ResNet50.
    Extracts L2-normalized embeddings suitable for similarity matching.
    
    Embedding dims:
      - resnet50  → 2048
      - resnet18  → 512
      - resnet101 → 2048
      - vgg16     → 4096
      - vgg19     → 4096
    """

    SUPPORTED = {
        'resnet18':  (models.resnet18,  models.ResNet18_Weights.DEFAULT,  512),
        'resnet50':  (models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
        'vgg16':     (models.vgg16,     models.VGG16_Weights.DEFAULT,     4096),
        'vgg19':     (models.vgg19,     models.VGG19_Weights.DEFAULT,     4096),
    }

    def __init__(self, model_name: str = 'resnet50', device=None):
        assert model_name in self.SUPPORTED, (
            f"model_name must be one of {list(self.SUPPORTED.keys())}"
        )
        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.embed_dim  = self.SUPPORTED[model_name][2]

        model_fn, weights, _ = self.SUPPORTED[model_name]
        base = model_fn(weights=weights)

        # ── Strip classification head, keep feature extractor only ──────────
        if model_name.startswith('resnet'):
            # Remove avgpool + fc → use AdaptiveAvgPool ourselves
            self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, C, 1, 1)
        else:
            # VGG: keep features + avgpool, replace classifier with identity
            self.backbone = nn.Sequential(
                base.features,
                base.avgpool,
                nn.Flatten(),
                base.classifier[0],   # first Linear: 25088 → 4096
                base.classifier[1],   # ReLU
                base.classifier[2],   # Dropout
                base.classifier[3],   # Linear: 4096 → 4096
                base.classifier[4],   # ReLU
                # drop final 4096→1000 classifier layer
            )

        self.backbone.eval().to(self.device)

        # ImageNet preprocessing (same as DINOv2Encoder)
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _to_pil(self, image) -> Image.Image:
        """Accept PIL Image or OpenCV BGR ndarray."""
        if isinstance(image, np.ndarray):
            return Image.fromarray(image[:, :, ::-1])  # BGR → RGB
        return image

    @torch.no_grad()
    def encode(self, image) -> np.ndarray:
        """
        image: PIL.Image or np.ndarray (BGR from OpenCV)
        returns: L2-normalized embedding — shape (embed_dim,)
          resnet → (2048,) | vgg → (4096,)
        """
        tensor = self.transform(self._to_pil(image))   # (3, 224, 224)
        tensor = tensor.unsqueeze(0).to(self.device)   # (1, 3, 224, 224)

        feat = self.backbone(tensor)                   # (1, C, 1, 1) or (1, C)
        feat = feat.flatten(1)                         # (1, embed_dim)
        feat = F.normalize(feat, p=2, dim=1)           # L2 normalize
        return feat.cpu().numpy()[0]                   # (embed_dim,)

    @torch.no_grad()
    def encode_batch(self, images: list) -> np.ndarray:
        """
        images: list of PIL.Image or np.ndarray
        returns: L2-normalized embeddings — shape (N, embed_dim)
        """
        tensors = torch.stack([
            self.transform(self._to_pil(img)) for img in images
        ]).to(self.device)                             # (N, 3, 224, 224)

        feat = self.backbone(tensors)                  # (N, C, 1, 1) or (N, C)
        feat = feat.flatten(1)                         # (N, embed_dim)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()                      # (N, embed_dim)

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """
        Cosine similarity between two embeddings.
        Both must be L2-normalized (as returned by encode).
        Returns float in [-1, 1], typically [0, 1] for natural images.
        """
        return float(np.dot(emb_a, emb_b))

    def similarity_batch(self, query: np.ndarray, refs: np.ndarray) -> np.ndarray:
        """
        query: (embed_dim,)
        refs:  (N, embed_dim)
        returns: (N,) cosine similarity scores
        """
        return refs @ query  # both L2-normalized → dot = cosine sim
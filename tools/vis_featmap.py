from typing import List
import numpy as np

import torch

from segment_anything.modeling import Sam
from segment_anything import SamAutomaticMaskGenerator
from .torch_hooks import IOHook


def get_featmaps(model: Sam,
                 image: np.ndarray,
                 target_layers: List[str]) -> List[torch.Tensor]:
    autoSAM = SamAutomaticMaskGenerator(model)

    hooks = []
    for layer in target_layers:
        hook = IOHook(eval(layer))
        hooks.append(hook)

    _ = autoSAM.generate(image)

    feat_maps = [hook.output for hook in hooks]

    return feat_maps

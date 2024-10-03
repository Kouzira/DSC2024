import torch
import torch.nn as nn
from transformers import AutoImageProcessor, EfficientNetModel
from typing import List, Optional, Tuple, Union


class ImageFeatureExtractor(nn.Modulo):
    r"""
    ImageFeatureExtractor uses efficientnet-b5 to extract features from images,
    then maps the features to fit the input size of phobert.
    """
    def __init__(self):
        super().__init__()

        self.efficient_net = EfficientNetModel.from_pretrained("google/efficientnet-b5")
        self.avg_pool2d = nn.AvgPool2d(2, ceil_mode=True)
        self.bridge_layer = nn.Linear(2048, 768)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Return a torch tensor of pooler output from final layer and flattened last hidden states.
        Shape: (Batch size, 65, 768)
        """
        
        outputs = model(input)
        last_hidden_states = outputs.last_hidden_state

        # (_, 2048, 15, 15) -> (_, 2048, 8, 8) 
        pooled_hidden_states = self.avg_pool2d(last_hidden_states)
        # (_, 2048, 8, 8) -> (_, 64, 2048) 
        pooled_hidden_states = torch.flatten(pooled_hidden_states, start_dim=2).permute(0, 2, 1)

        final_model_output = outputs.pooler_output.unsqueeze(1)

        feature = torch.cat((pooled_hidden_states, final_model_output), dim=1)
        feature = self.bridge_layer(feature)

        return feature
    
import os
from pathlib import Path
from urllib.parse import urlencode

import requests
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F


class WolfClassifier(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.feature_extractor: nn.Module = torchaudio.pipelines.HUBERT_BASE.get_model()

        self._cwd: str = os.getcwd()

        hidden_size: int = 0
        if hasattr(self.feature_extractor, 'encoder'):
            hidden_size = self.feature_extractor.encoder.transformer.layers[0].attention.k_proj.out_features
        elif hasattr(self.feature_extractor, 'model'):
            hidden_size = self.feature_extractor.model.encoder.transformer.layers[0].attention.k_proj.out_features

        self.linear: nn.Linear = nn.Linear(hidden_size, 2)
        if not Path(self._cwd).joinpath('./data/saved_weights.pth').exists():
            self.load_weights()

        self.load_state_dict(torch.load(str(Path(self._cwd).joinpath('./data/saved_weights.pth')), map_location='cpu'))

    def load_weights(self):
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
        public_key = 'https://disk.yandex.ru/d/jXBIs4C9O3fcCQ'

        final_url = base_url + urlencode({'public_key': public_key})
        response = requests.get(
            final_url,
            timeout=60,
        )
        download_url = response.json()['href']

        download_response = requests.get(
            download_url,
            timeout=60,
        )
        with open(str(Path(self._cwd).joinpath('./data/saved_weights.pth')), 'wb') as f:
            f.write(download_response.content)

    @torch.inference_mode()
    def get_embeddings(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.feature_extractor(input_tensor)[0].mean(axis=1)

        return F.normalize(embeddings)

    @torch.inference_mode()
    def get_wolf_probability(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        features = self.get_embeddings(input_tensor)

        return F.softmax(self.linear(features), dim=-1)[:, 1]

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(F.normalize(self.feature_extractor(input_tensor)[0].mean(axis=1)))

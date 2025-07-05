from pathlib import Path
from urllib.parse import urlencode

import requests
import torch
from loguru import logger
from torch import nn
from torch.nn import functional as F
from transformers import ASTForAudioClassification

root_dir: Path = Path(__file__).parent.parent


class ASTBasedClassifier(nn.Module):
    def __init__(
        self,
        yandex_disk_public_key: str,
        dump_label: str,
    ):
        super().__init__()

        self._model: nn.Module = ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
        self._model.classifier.dense = nn.Linear(
            self._model.classifier.dense.in_features,
            2,
        )

        self._yandex_disk_public_key: str = yandex_disk_public_key
        self._dump_label: str = dump_label

        if not (root_dir / 'saved_weights' / f'{dump_label.replace(" ", "_")}.pth').exists():
            logger.info(f'Weight dump for the {dump_label} was not found')

            self.load_weights()

        self.load_state_dict(
            torch.load(
                str(root_dir / 'saved_weights' / f'{dump_label.replace(" ", "_")}.pth'),
                map_location='cpu',
                weights_only=True,
            ),
        )

    def load_weights(self) -> None:
        timeout: int = 30

        logger.info(f'Loading weights for the {self._dump_label} model from yandex disk')

        base_url: str = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
        base_url = base_url + urlencode({'public_key': self._yandex_disk_public_key})

        response = requests.get(
            base_url,
            timeout=timeout,
        )
        download_url: str = response.json()['href']
        download_response = requests.get(
            download_url,
            timeout=timeout,
        )

        logger.success(f'Weights for the {self._dump_label} model were downloaded')

        with open(
            str(root_dir / 'saved_weights' / f'{self._dump_label.replace(" ", "_")}.pth'),
            'wb',
        ) as dump_file:
            dump_file.write(download_response.content)

            logger.success(f'Weights for {self._dump_label} were saved as a saved_weights/{self._dump_label.replace(" ", "_")}')

    @torch.inference_mode()
    def get_target_class_probability(
        self,
        input_tensor: torch.Tensor,  # (batch_size, n_features, seq_len)
        target_class: int,
    ) -> torch.Tensor:
        return F.softmax(
            self._model(input_tensor).logits,
            dim=-1,
        )[..., target_class]

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self._model(input_tensor).logits

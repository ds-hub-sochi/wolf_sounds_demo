import torch
from transformers import AutoFeatureExtractor


class ASTConverter:
    def __init__(
        self,
        sample_rate: int,
    ):
        self._feature_extractor: AutoFeatureExtractor = \
            AutoFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
        self._sample_rate: int = sample_rate

    def convert(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        waveform = waveform.view(-1)
        
        return self._feature_extractor(
            waveform,
            sampling_rate = self._sample_rate,
            return_tensors = 'pt',
        )['input_values']

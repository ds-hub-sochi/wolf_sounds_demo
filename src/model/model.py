from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class WoldSoundClassifierIntarface(nn.Module):
    @abstractmethod
    def get_wolf_probability(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        pass


class BaseWolfClassifier(WoldSoundClassifierIntarface):
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


WOLF_CLASSIFIER_REGISTER: dict[str, type[WoldSoundClassifierIntarface]] = {}

def wolf_classifier(cls: type[WoldSoundClassifierIntarface]) -> type[WoldSoundClassifierIntarface]:
    WOLF_CLASSIFIER_REGISTER[cls.__name__[:-19]] = cls
    return cls


@wolf_classifier
class Wav2Vec2WolfSoundClassifier(BaseWolfClassifier):
    def __init__(
        self,
    ):
        super().__init__()

        self.feature_extractor: nn.Module = torchaudio.pipelines.WAV2VEC2_BASE.get_model()

        hidden_size: int = 0
        if hasattr(self.feature_extractor, 'encoder'):
            hidden_size = self.feature_extractor.encoder.transformer.layers[0].attention.k_proj.out_features
        elif hasattr(self.feature_extractor, 'model'):
            hidden_size = self.feature_extractor.model.encoder.transformer.layers[0].attention.k_proj.out_features

        self.linear: nn.Linear = nn.Linear(hidden_size, 2)


@wolf_classifier
class HubertWolfSoundClassifier(BaseWolfClassifier):
    def __init__(
        self,
    ):
        super().__init__()

        self.feature_extractor: nn.Module = torchaudio.pipelines.HUBERT_BASE.get_model()

        hidden_size: int = 0
        if hasattr(self.feature_extractor, 'encoder'):
            hidden_size = self.feature_extractor.encoder.transformer.layers[0].attention.k_proj.out_features
        elif hasattr(self.feature_extractor, 'model'):
            hidden_size = self.feature_extractor.model.encoder.transformer.layers[0].attention.k_proj.out_features

        self.linear: nn.Linear = nn.Linear(hidden_size, 2)

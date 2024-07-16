import torch

from src.model.model import WOLF_CLASSIFIER_REGISTER, WoldSoundClassifierIntarface


class ClassifierFactory:
    def create(
        self,
        model_type: str,
        checkpoint_dir,
    ) -> type[WoldSoundClassifierIntarface]:
        model: type[WoldSoundClassifierIntarface] = WOLF_CLASSIFIER_REGISTER[model_type]()
        model.load_state_dict(
            torch.load(
                f'{checkpoint_dir}/{model_type.lower()}.pth',
                map_location=torch.device('cpu'),
            )
        )
        model.eval()

        return model

from __future__ import annotations

import shutil
import glob
import os
import pathlib

import streamlit as st
import torchaudio
from stqdm import stqdm

from src.model.model import WoldSoundClassifierIntarface
from src.model.model_factory import ClassifierFactory


# BATCH_SIZE: int = 4
MODEL_SAMPLE_RATE: int = 16000
THRESHOLD: float = 0.5
MODEL_CHECKPOINT_PATH: str = 'models/wav2vec2_base.pth'


def get_style() -> str:
    style = """
                <style>
                .block-container {
                    max-width: 70%
                }
                </style>
             """

    return style

def main():
    st.markdown(
        get_style(),
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            'Выбор модели',
            'Разметка аудио',
        ],
    )

    key: int = 1

    factory: ClassifierFactory = ClassifierFactory()

    with tabs[0]:
        model_type: str = st.selectbox(
            label='Тип модели',
            options=(
                'Wav2Vec2',
                'Hubert',
            ),
            index=None,
            key=key,
        )
        key += 1

        if model_type is not None:
            model: type[WoldSoundClassifierIntarface] = factory.create(model_type, './models')
            st.write('Подель успешно загружена')

    with tabs[1]:
        input_dir_path: str | pathlib.Path = st.text_input(
            label='Путь до директории с аудио-файлами',
            value='',
            key=key,
            help='Путь должен быть абсолютным или относительным (для текущей директории)',
        )
        key += 1

        output_dir_path: str | pathlib.Path = st.text_input(
            label='Путь до директории, в которую требуется сохранить результаты работы модели',
            value='',
            key=key,
            help='Путь должен быть абсолютным или относительным (для текущей директории)',
        )
        key += 1

        if all(
            [
                input_dir_path != '',
                output_dir_path != '',
            ],
        ):
            pathlib.Path(f'{output_dir_path}/wolf').mkdir(
                parents=True,
                exist_ok=True,
            )
            pathlib.Path(f'{output_dir_path}/no_wolf').mkdir(
                parents=True,
                exist_ok=True,
            )

            if st.button('Разметить данные'):
                all_wav_files: list[str] = []
                all_wav_files.extend(glob.glob(f'{os.path.abspath(input_dir_path)}/*.wav'))
                all_wav_files.extend(glob.glob(f'{os.path.abspath(input_dir_path)}/*/*.wav'))

                for current_file in stqdm(all_wav_files):
                    waveform, sample_rate = torchaudio.load(
                        current_file,
                        format='wav',
                        backend='ffmpeg',
                    )

                    if sample_rate != MODEL_SAMPLE_RATE:
                        waveform = torchaudio.functional.resample(
                            waveform,
                            sample_rate,
                            MODEL_SAMPLE_RATE,
                        )

                    waveform = waveform.mean(
                        dim=0,
                        keepdim=True,
                    )

                    wolf_probability: float = model.get_wolf_probability(waveform)[0]

                    filename: str = current_file.split('/')[-1]
                    
                    if wolf_probability > THRESHOLD:
                        shutil.copyfile(current_file, f'{output_dir_path}/wolf/{filename}')
                    else:
                        shutil.copyfile(current_file, f'{output_dir_path}/no_wolf/{filename}')

                st.write('Все аудио-файлы успешно размечены')


if __name__ == '__main__':
    main()

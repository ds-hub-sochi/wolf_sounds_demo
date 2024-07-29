from __future__ import annotations

import json
import pathlib
from math import ceil
from time import gmtime, strftime

import streamlit as st
import torch
import torchaudio
from stqdm import stqdm

from src import constants
from src import model


if 'MODEL_SAMPLE_RATE' not in st.session_state:
    st.session_state.MODEL_SAMPLE_RATE = constants.MODEL_SAMPLE_RATE

if 'CHUNK_SIZE' not in st.session_state:
    st.session_state.CHUNK_SIZE = constants.CHUNK_SIZE

if 'BATCH_SIZE' not in st.session_state:
    st.session_state.BATCH_SIZE = constants.BATCH_SIZE

if 'CHUNKS_WITH_WOLF_RATE' not in st.session_state:
    st.session_state.CHUNKS_WITH_WOLF_RATE = constants.CHUNKS_WITH_WOLF_RATE

if 'DEVICE' not in st.session_state:
    st.session_state.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if 'model' not in st.session_state:
    st.session_state.model = model.WolfClassifier().to(st.session_state.DEVICE)

uploaded_files = st.file_uploader(
    'Выберете wav-файлы. Суммарный вес не должен превышать 6 GB',
    accept_multiple_files=True,
    type='wav',
)

pathlib.Path('./data/processed').mkdir(
    parents=True,
    exist_ok=True,
)

markup: dict[str, list[tuple[str, str]]] = {}
with open(
    './data/processed/results.json',
    'w',
    encoding='utf-8',
) as f:
    json.dump(markup, f)


CONFIDENCE_THRESHOLD = st.number_input(
    'Минимальное значение для уверенности модели в наличии воя',
    value=None,
    help='от 0 до 1. Чем ниже значение - тем большее количество записей без воя будет помечено, как с наличием воя.'
)
if CONFIDENCE_THRESHOLD is not None:
    if st.button(label='Начать разметку'):
        for uploaded_file in stqdm(uploaded_files):
            bytes_data = uploaded_file.read()
            file_name: str = uploaded_file.name

            waveform, sample_rate = torchaudio.load(
                bytes_data,
                backend='ffmpeg',
            )

            if sample_rate != st.session_state.MODEL_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    st.session_state.MODEL_SAMPLE_RATE,
                )

            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

            audio_chunks: list[torch.Tensor] = list(
                torch.split(
                    waveform,
                    st.session_state.MODEL_SAMPLE_RATE * st.session_state.CHUNK_SIZE,
                    dim=1,
                ),
            )

            current_file_markup: list[float] = []
            duration: list[int] = []

            for start_index in range(0, len(audio_chunks), st.session_state.BATCH_SIZE):
                current_tensors = audio_chunks[start_index : start_index + st.session_state.BATCH_SIZE]
                duration.extend([st.session_state.CHUNK_SIZE] * (len(current_tensors) - 1))

                duration.append(ceil(current_tensors[-1].shape[1] / st.session_state.MODEL_SAMPLE_RATE))

                if current_tensors[-1].shape != current_tensors[0].shape:
                    padding = torch.nn.ConstantPad1d(
                        padding=(
                            0,
                            st.session_state.CHUNK_SIZE * st.session_state.MODEL_SAMPLE_RATE - current_tensors[-1].shape[1],
                        ),
                        value=0,
                    )
                    current_tensors[-1] = padding(current_tensors[-1])

                if current_tensors[-1].shape[1] < st.session_state.MODEL_SAMPLE_RATE:
                    padding = torch.nn.ConstantPad1d(
                        padding=(
                            0,
                            st.session_state.CHUNK_SIZE * st.session_state.MODEL_SAMPLE_RATE - current_tensors[-1].shape[1],
                        ),
                        value=0,
                    )
                    current_tensors[-1] = padding(current_tensors[-1])

                batch: torch.Tensor = torch.stack(
                    current_tensors,
                    dim=0,
                )
                batch = batch.squeeze(1).to(st.session_state.DEVICE)

                current_file_markup.extend(list(st.session_state.model.get_wolf_probability(batch).cpu().numpy()))

            markup[file_name] = []

            running_duration: float = 0.0
            number_of_chunks_with_noise: int = 0
            for duration_value, wolf_probability in zip(duration, current_file_markup):
                start = running_duration
                end = running_duration + duration_value

                if wolf_probability > CONFIDENCE_THRESHOLD:
                    number_of_chunks_with_noise += 1
                    markup[file_name].append(
                    (
                        strftime("%H:%M:%S", gmtime(start)),
                        strftime("%H:%M:%S", gmtime(end)),
                    ),
                )

                running_duration += duration_value

            if number_of_chunks_with_noise > len(duration) * st.session_state.CHUNKS_WITH_WOLF_RATE:
                markup[file_name] = [
                    (
                        '00:00:00',
                        strftime(
                            '%H:%M:%S',
                            gmtime(ceil(waveform.shape[1] / st.session_state.MODEL_SAMPLE_RATE))
                        )
                    )
                ]
            else:
                intervals: list[tuple[str, str]] = []

                interval_start = '00:00:00'
                interval_end = '00:00:00'
                for value in markup[file_name]:
                    if value[0] == interval_end:
                        interval_end = value[1]
                    elif interval_end != '00:00:00':
                        intervals.append((interval_start, interval_end))
                        interval_start = value[0]
                        interval_end = value[1]

                if interval_end != '00:00:00':
                    intervals.append((interval_start, interval_end))

                markup[file_name] = intervals

            with open(
                './data/processed/results.json',
                'w',
                encoding='utf-8',
            ) as f:
                json.dump(markup, f)

    with open(
        './data/processed/results.json',
        'r',
        encoding='utf-8',
    ) as f:
        st.download_button(
            label='Получить результаты разметки',
            data=f,
            file_name='results.json',
        )

import json
from math import ceil
from time import gmtime, strftime

import streamlit as st
import torch
import torchaudio
from stqdm import stqdm

from src import model
from src import constants


if "MODEL_SAMPLE_RATE" not in st.session_state:
    st.session_state.MODEL_SAMPLE_RATE = constants.MODEL_SAMPLE_RATE

if "CHUNK_SIZE" not in st.session_state:
    st.session_state.CHUNK_SIZE = constants.CHUNK_SIZE

if "BATCH_SIZE" not in st.session_state:
    st.session_state.BATCH_SIZE = constants.BATCH_SIZE

if "CONFIDENCE_THRESHOLD" not in st.session_state:
    st.session_state.CONFIDENCE_THRESHOLD = constants.CONFIDENCE_THRESHOLD

if "DEVICE" not in st.session_state:
    st.session_state.DEVICE = constants.DEVICE

if "model" not in st.session_state:
    st.session_state.model = model.WolfClassifier().to(st.session_state.DEVICE)

uploaded_files = st.file_uploader(
    "Выберете wav-файлы",
    accept_multiple_files=True,
    type="wav",
)

if st.button(label='Начать разметку'):
    markup = dict()
    with open("./data/results.json", "w") as f:
        json.dump(markup, f)

    for uploaded_file in stqdm(uploaded_files):
        bytes_data = uploaded_file.read()
        file_name: str = uploaded_file.name

        waveform, sample_rate = torchaudio.load(bytes_data)

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
            )
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

            batch = torch.stack(
                current_tensors,
                dim=0,
            )
            batch = batch.squeeze(1).to(st.session_state.DEVICE)

            current_file_markup.extend(
                list(st.session_state.model.get_wolf_probability(batch).cpu().numpy())
            )

        markup[file_name] = []
        running_duration = 0.0
        for duration_value, wolf_probability in zip(duration, current_file_markup):
            start = running_duration
            end = running_duration + duration_value

            if wolf_probability > st.session_state.CONFIDENCE_THRESHOLD:
                markup[file_name].append((f"{strftime('%H:%M:%S', gmtime(start))}", f"{strftime('%H:%M:%S', gmtime(end))}"))
            
            running_duration += duration_value
            
        with open("./data/results.json", "w") as f:
            json.dump(markup, f)

st.download_button(
   label="Получить результаты разметки",
   data=open("./data/results.json", 'r'),
   file_name="results.json",
)

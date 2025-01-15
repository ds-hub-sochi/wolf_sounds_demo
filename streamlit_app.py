from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from shutil import make_archive

import streamlit as st
import torch
import torchaudio
from stqdm import stqdm

from src import constants, feature_extractor, model, utils

root_dir: Path = Path(__file__).parent

(root_dir / 'data' / 'processed').mkdir(
    parents=True,
    exist_ok=True,
)

(root_dir / 'saved_weights').mkdir(
    parents=True,
    exist_ok=True,
)

(root_dir / 'data' / 'processed' / 'audio_with_detected_animals').mkdir(
    parents=True,
    exist_ok=True,
)

markup: dict[str, defaultdict[list[str]]] = {}
with open(
    root_dir / 'data' / 'processed' / 'timings.json',
    'w',
    encoding='utf-8',
) as timings_file:
    json.dump(markup, timings_file)

if 'MODEL_SAMPLE_RATE' not in st.session_state:
    st.session_state.MODEL_SAMPLE_RATE = constants.MODEL_SAMPLE_RATE

if 'CHUNK_SIZE' not in st.session_state:
    st.session_state.CHUNK_SIZE = constants.CHUNK_SIZE

if 'BATCH_SIZE' not in st.session_state:
    st.session_state.BATCH_SIZE = constants.BATCH_SIZE

if 'CONFIDENCE_THRESHOLD' not in st.session_state:
    st.session_state.CONFIDENCE_THRESHOLD = constants.CONFIDENCE_THRESHOLD

if 'DEVICE' not in st.session_state:
    st.session_state.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = feature_extractor.ASTConverter(st.session_state.MODEL_SAMPLE_RATE)

if 'animal_classifier' not in st.session_state:
    st.session_state.animal_classifier = model.ASTBasedClassifier(
        'https://disk.yandex.ru/d/1Jz2-F7fArielA',
        'animal_vs_no_animal',
    ).to(st.session_state.DEVICE).eval()

if 'wolf_classifier' not in st.session_state:
    st.session_state.wolf_classifier = model.ASTBasedClassifier(
        'https://disk.yandex.ru/d/S4m1-1AV-O10pQ',
        'wolf_vs_other_animal',
    ).to(st.session_state.DEVICE).eval()


uploaded_files = st.file_uploader(
    'Select wav files. The total weight must not exceed 6 GB',
    accept_multiple_files=True,
    type='wav',
)

if st.button(label='Classify'):
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

        if audio_chunks[-1].shape[-1] < st.session_state.MODEL_SAMPLE_RATE:
            audio_chunks = audio_chunks[:-1]

        current_file_markup: list[str] = []
        duration: list[int] = []

        for start_index in range(0, len(audio_chunks), st.session_state.BATCH_SIZE):
            current_tensors: list[torch.Tensor] = audio_chunks[start_index : start_index + st.session_state.BATCH_SIZE]

            duration.extend([round(_.shape[-1] / st.session_state.MODEL_SAMPLE_RATE) for _ in current_tensors])

            features: list[torch.Tensor] = [st.session_state.feature_extractor.convert(wav) for wav in current_tensors]

            difference: int = features[0].shape[1] - features[-1].shape[1]
            if difference != 0:
                padding: torch.nn.ZeroPad2d = torch.nn.ZeroPad2d((0, difference, 0, 0))
                features[-1] = padding(features[-1])

            batch: torch.Tensor = torch.cat(
                features,
                dim=0,
            ).to(st.session_state.DEVICE)

            animal_probability: torch.Tensor = st.session_state.animal_classifier.get_target_class_probability(
                batch,
                0,
            ).cpu()

            animal_indices: torch.Tensor = (animal_probability > st.session_state.CONFIDENCE_THRESHOLD).nonzero().view(-1)

            animal_batch: torch.Tensor = torch.index_select(
                batch.to('cpu'),
                dim=0,
                index=animal_indices,
            ).to(st.session_state.DEVICE)

            wolf_probability: torch.Tensor = st.session_state.wolf_classifier.get_target_class_probability(
                animal_batch,
                0,
            ).cpu()

            wolf_indices: torch.Tensor = (wolf_probability > st.session_state.CONFIDENCE_THRESHOLD).nonzero().view(-1)
            wolf_original_indices: torch.Tensor = torch.index_select(
                animal_indices,
                dim=0,
                index=wolf_indices,
            )

            batch_markup: list[str] = ['no animals' for _ in range(len(current_tensors))]

            for index in animal_indices:
                batch_markup[index] = 'other animal'

            for index in wolf_original_indices:
                batch_markup[index] = 'wolf'

            current_file_markup.extend(batch_markup)

        markup[file_name] = utils.format_markup(
            durations=duration,
            labels=current_file_markup,
        )

        if len(markup[file_name]) != 0:
            torchaudio.save(
                root_dir / 'data' / 'processed' / 'audio_with_detected_animals' / f'{file_name}',
                waveform,
                sample_rate=st.session_state.MODEL_SAMPLE_RATE,
                format='wav',
                backend='ffmpeg',
            )

    with open(
        root_dir / 'data' / 'processed' / 'timings.json',
        'w',
        encoding='utf-8',
    ) as timings_file:
        json.dump(markup, timings_file)

    make_archive(
        root_dir / 'data' / 'results',
        'zip',
        root_dir / 'data' / 'processed',
    )

    st.write(f'Classification was done; {len(uploaded_files)} files were classified')

    with open(
        root_dir / 'data' / 'results.zip',
        'rb',
    ) as zip_result:
        st.download_button(
            label='Get classification results',
            data=zip_result,
            help='You will get an archive with audio files where animal were detected and also a file with timings',
            file_name='results.zip',
        )

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import torch

from src.model import ASTBasedClassifier


def format_markup(
    durations: list[int],
    labels: list[str],
) -> defaultdict[list[str]]:
    """
    This function formats markup so it can be better understanded.
    Essentially, it collapses adjacent moments of time of one class into one single interval like:

    if durations = [30, 30, 30, 30], labels = ['wolf', 'wolf', 'other animal', 'wolf'] then the result will be
    {
        'wolf': 0:00:00-0:01:00, 0:01:30-0:02:00,
        'other animal': 0:01:00-0:01:30,
    }

    Args:
        durations (list[int]): list of chunks' durations in second
        labels (list[str]): list of chunks's classes

    Returns:
        defaultdict[list[str]]: mapping with time intervals related to each class
    """

    mapping: defaultdict[list[str]] = defaultdict(list)

    current_label: str = labels.pop(0)
    current_label_start_timing: int = 0

    running_duration: int = durations.pop(0)

    for duration, label in zip(durations, labels):
        if label != current_label:
            mapping[current_label].append(
                f"{str(timedelta(seconds=current_label_start_timing))}" + \
                "-" + \
                f"{str(timedelta(seconds=running_duration))}"
            )

            current_label_start_timing = running_duration
            current_label = label
        
        running_duration += duration

    mapping[current_label].append(
        f"{str(timedelta(seconds=current_label_start_timing))}" + \
        "-" + \
        f"{str(timedelta(seconds=running_duration))}"
    )

    return mapping

@torch.inference_mode()
def get_animal_indices(
    animal_vs_no_animal_model: ASTBasedClassifier,
    wolf_vs_other_animal_model: ASTBasedClassifier,
    threshold: float,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function takes batch and return a tuple of indices of its tensors where animal were detected 
    and indices of its tensors where wolf's howl were detected

    Args:
        animal_vs_no_animal_model (ASTBasedClassifier): model that classifies any animal appearance
        wolf_vs_other_animal_model (ASTBasedClassifier): model that classifies wolf howl appearance
        threshold (float): confidence threshold for the classification
        batch (torch.Tensor): batch of features to classify

    Returns:
        tuple[torch.Tensor, torch.Tensor]: indices of tensors with animals detected,
                                           indices of tensors with wolf howl detected
    """
    device: torch.device = next(animal_vs_no_animal_model.parameters()).device

    animal_probability: torch.Tensor = animal_vs_no_animal_model.get_target_class_probability(
        batch,
        0,
    ).cpu()

    animal_indices: torch.Tensor = (animal_probability > threshold).nonzero().view(-1)

    animal_batch: torch.Tensor = torch.index_select(
        batch.to('cpu'),
        dim=0,
        index=animal_indices,
    ).to(device)

    wolf_probability: torch.Tensor = wolf_vs_other_animal_model.get_target_class_probability(
        animal_batch,
        0,
    ).cpu()

    wolf_indices: torch.Tensor = (wolf_probability > threshold).nonzero().view(-1)
    wolf_original_indices: torch.Tensor = torch.index_select(
        animal_indices,
        dim=0,
        index=wolf_indices,
    )

    return animal_indices, wolf_original_indices

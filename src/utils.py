from __future__ import annotations

from collections import defaultdict
from datetime import timedelta


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

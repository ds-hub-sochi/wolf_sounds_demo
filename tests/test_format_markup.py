from collections import defaultdict

from src import utils


def test_from_dockstring():
    durations: list[int] = [30, 30, 30, 30]
    labels: list[str] = ['wolf', 'wolf', 'other animal', 'wolf']

    formatted_markup: defaultdict[list[str]] = utils.format_markup(
        durations,
        labels,
    )

    assert len(list(formatted_markup.keys())) == 2

    assert len(formatted_markup['wolf']) == 2
    assert len(formatted_markup['other animal']) == 1

    assert formatted_markup['other animal'][0] == '0:01:00-0:01:30'

    assert formatted_markup['wolf'][0] == '0:00:00-0:01:00'
    assert formatted_markup['wolf'][1] == '0:01:30-0:02:00'


def test_all_intervals_have_the_same_class():
    durations: list[int] = [30, 30, 30]
    labels: list[str] = ['wolf', 'wolf', 'wolf']

    formatted_markup: defaultdict[list[str]] = utils.format_markup(
        durations,
        labels,
    )

    assert len(list(formatted_markup.keys())) == 1

    assert len(formatted_markup['wolf']) == 1

    assert formatted_markup['wolf'][0] == '0:00:00-0:01:30'


def test_diffenets_adjacent_classes():
    durations: list[int] = [30, 30, 30, 30, 30]
    labels: list[str] = ['wolf', 'other animal', 'wolf', 'other animal', 'wolf']

    formatted_markup: defaultdict[list[str]] = utils.format_markup(
        durations,
        labels,
    )

    assert len(list(formatted_markup.keys())) == 2

    assert len(formatted_markup['wolf']) == 3
    assert len(formatted_markup['other animal']) == 2

    assert formatted_markup['wolf'][0] == '0:00:00-0:00:30'
    assert formatted_markup['wolf'][1] == '0:01:00-0:01:30'
    assert formatted_markup['wolf'][2] == '0:02:00-0:02:30'

    assert formatted_markup['other animal'][0] == '0:00:30-0:01:00'
    assert formatted_markup['other animal'][1] == '0:01:30-0:02:00'


def test_without_interval_overlapping():
    durations: list[int] = [30, 30, 30, 30, 30]
    labels: list[str] = ['wolf', 'other animal', 'wolf', 'other animal', 'wolf']

    formatted_markup: defaultdict[list[str]] = utils.format_markup(
        durations,
        labels,
    )

    assert len(list(formatted_markup.keys())) == 2

    assert len(formatted_markup['wolf']) == 3
    assert len(formatted_markup['other animal']) == 2

    assert formatted_markup['wolf'][0] == '0:00:00-0:00:30'
    assert formatted_markup['wolf'][1] == '0:01:00-0:01:30'
    assert formatted_markup['wolf'][2] == '0:02:00-0:02:30'

    assert formatted_markup['other animal'][0] == '0:00:30-0:01:00'
    assert formatted_markup['other animal'][1] == '0:01:30-0:02:00'


def test_all_unique_classes_appearance():
    durations: list[int] = [30, 30, 30]
    labels: list[str] = ['wolf', 'other animal', 'no animals']

    formatted_markup: defaultdict[list[str]] = utils.format_markup(
        durations,
        labels,
    )

    assert len(list(formatted_markup.keys())) == 3

    assert len(formatted_markup['wolf']) == 1
    assert len(formatted_markup['other animal']) == 1
    assert len(formatted_markup['no animals']) == 1

    assert formatted_markup['wolf'][0] == '0:00:00-0:00:30'

    assert formatted_markup['other animal'][0] == '0:00:30-0:01:00'

    assert formatted_markup['no animals'][0] == '0:01:00-0:01:30'

import numpy as np

from src.preprocessing.dataset_builder import split_windows_by_source


def test_source_split_is_deterministic_stratified_and_grouped():
    windows = []
    for label, prefix in ((0, "adl"), (1, "fall")):
        for source_index in range(10):
            source_name = f"{prefix}-{source_index:02d}"
            for _ in range(3):
                windows.append((np.zeros((2, 4), dtype=np.float32), label, source_name))

    train, val, test = split_windows_by_source(iter(windows))
    repeated_split = split_windows_by_source(iter(windows))

    assert [len(partition) for partition in (train, val, test)] == [
        len(partition) for partition in repeated_split
    ]
    for partition in (train, val, test):
        assert {window[1] for window in partition} == {0, 1}

    source_sets = [
        {window[2] for window in partition}
        for partition in (train, val, test)
    ]
    assert source_sets[0].isdisjoint(source_sets[1])
    assert source_sets[0].isdisjoint(source_sets[2])
    assert source_sets[1].isdisjoint(source_sets[2])


def test_source_split_keeps_singleton_sources_in_training():
    windows = [
        (np.zeros((2, 4), dtype=np.float32), 0, "only-normal-source"),
        (np.zeros((2, 4), dtype=np.float32), 1, "only-fall-source"),
    ]

    train, val, test = split_windows_by_source(windows)

    assert len(train) == 2
    assert val == []
    assert test == []

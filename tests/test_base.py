from pyGRNN.base import combs, countX, list_combs, recursive_combination_excluder


def test_countX_counts_occurrences():
    assert countX([1, 2, 2, 3, 2], 2) == 3
    assert countX(["a", "b"], "c") == 0


def test_list_combs_includes_empty_and_full_set():
    result = list_combs((1, 2))
    assert () in result
    assert (1, 2) in result
    assert len(result) == 2**2  # every subset, including the empty one


def test_combs_excludes_the_empty_set():
    result = list(combs((1, 2)))
    assert () not in result
    assert (1, 2) in result
    assert len(result) == 2**2 - 1


def test_recursive_combination_excluder_keeps_only_supersets_of_excludes():
    excludes = [{1}]
    result = list(recursive_combination_excluder([1, 2, 3], 2, excludes))
    assert all(1 in comb for comb in result)
    assert (2, 3) not in result

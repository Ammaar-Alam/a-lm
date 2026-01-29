from alm.rlvr.reward import dense_int_reward, exact_int_reward, extract_int


def test_extract_int() -> None:
    assert extract_int("Final: 42") == "42"
    assert extract_int("no numbers here") is None
    assert extract_int("a=1 b=2") == "2"


def test_exact_int_reward() -> None:
    assert exact_int_reward("42", "42") == 1.0
    assert exact_int_reward("41", "42") == 0.0


def test_dense_int_reward() -> None:
    assert dense_int_reward("42", "42") == 1.0
    assert dense_int_reward("41", "42") == 0.5
    assert dense_int_reward("40", "42") == 1.0 / 3.0

import pytest

from asr.antiloop import strip_repetition_loops as strip


@pytest.mark.parametrize("text,expected,flag", [
    ("shock de-o-200 joules. " + "I'm sorry " * 40, "shock de-o-200 joules.", True),
    ("Okay so happy 1 mg IV push. " + "I'm sorry, but " * 30, "Okay so happy 1 mg IV push.", True),
    ("I'm sorry " * 30, "", True),
    ("push push push push push", "", True),
    ("gave one one milligram", "gave one one milligram", False),
    ("BP 120 over 80, heart rate 110, Sat 94.", "BP 120 over 80, heart rate 110, Sat 94.", False),
    ("Narcan 2 mg intranasal, sorry, 4 mg IV.", "Narcan 2 mg intranasal, sorry, 4 mg IV.", False),
])
def test_loops_are_cut_and_normal_speech_untouched(text, expected, flag):
    assert strip(text) == (expected, flag)
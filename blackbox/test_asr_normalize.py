from eval.score import _norm_words, _raw_words


def test_raw_metric_keeps_number_and_wording_differences():
    assert _raw_words("2.5 mg IV!") == ["2.5", "mg", "iv"]
    assert _raw_words("two and a half milligrams intravenous") != _raw_words("2.5 mg IV")


def test_normalized_metric_folds_numbers_units_routes_and_punctuation():
    assert _norm_words("Midazolam two and a half milligrams IV.") == _norm_words("midazolam 2.5 mg intravenous")
    assert _norm_words("one fifty micrograms intramuscular") == _norm_words("150 mcg IM")
    assert _norm_words("point four millilitres IO") == _norm_words("0.4 ml intraosseous")
    assert _norm_words("fentanyl 50 mcg intranasal") == _norm_words("fentanyl 50 micrograms nasal")
    assert _norm_words("two hundred joules") == _norm_words("200 J")
    assert _norm_words("five migs intravenous push") == _norm_words("5 mg IVP")
    assert _norm_words("500 cc by mouth") == _norm_words("500 ml PO")


def test_normalization_does_not_rewrite_drug_identity():
    assert _norm_words("morphine 4 mg") != _norm_words("midazolam 4 mg")

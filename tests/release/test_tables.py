from pathlib import Path

import pytest

from appl_release.tables import _format, exp1, exp2


ROOT = Path(__file__).resolve().parents[2]


def test_exp1_recomputed_main_table_matches_paper():
    rows = {row["method"]: row for row in exp1(ROOT)}
    assert rows["B0_vanilla_dp"]["mean_OOD"] == pytest.approx(37.86458333333333)
    assert rows["B1_rule_prior"]["mean_OOD"] == pytest.approx(47.291666666666664)
    assert rows["q1"]["mean_OOD"] == pytest.approx(67.91666666666667)
    assert rows["q3"]["mean_OOD"] == pytest.approx(85.20833333333334)
    assert rows["q4"]["mean_OOD"] == pytest.approx(92.39583333333333)
    assert _format(rows["q4"]["N20_OOD"]) == "93.13"


def test_exp2_recomputed_main_table_matches_paper_and_omits_vla():
    rows = {row["method"]: row for row in exp2(ROOT)}
    expected = {
        "dp": (10.0, None, None),
        "single_prior": (10.0, None, None),
        "without_interface_information": (20.0, 65.0, "2/16"),
        "without_prior_information": (45.0, 72.5, "3/16"),
        "rule4": (22.5, 52.5, "1/16"),
        "without_verification": (52.5, 67.5, "7/16"),
        "appl": (50.0, 92.5, "8/16"),
    }
    assert set(rows) == set(expected)
    for method, result in expected.items():
        assert (rows[method]["motion_percent"], rows[method]["task_percent"], rows[method]["composition"]) == result

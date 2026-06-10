from __future__ import annotations

import pytest

from medmissingai.data.modality_settings import all_nonempty_settings, parse_settings, setting_id


def test_all_nonempty_settings_for_four_modalities():
    settings = all_nonempty_settings(("t1", "t1ce", "t2", "flair"))

    assert len(settings) == 15
    assert ("t1",) in settings
    assert ("t1", "t1ce", "t2", "flair") in settings


def test_parse_custom_settings():
    settings = parse_settings("t1+t2,t1ce+flair", ("t1", "t1ce", "t2", "flair"))

    assert settings == [("t1", "t2"), ("t1ce", "flair")]
    assert setting_id(settings[0]) == "t1+t2"


def test_parse_settings_rejects_unknown_modality():
    with pytest.raises(ValueError, match="Unknown modality"):
        parse_settings("t1+adc", ("t1", "t1ce", "t2", "flair"))


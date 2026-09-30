from pathlib import Path
from unittest.mock import Mock

import myosuite_init


def test_no_myoapi(monkeypatch, tmp_path):
    monkeypatch.setattr(myosuite_init, "simhive_path", str(tmp_path / "simhive"))
    fetch_git = Mock()
    monkeypatch.setattr(myosuite_init, "fetch_git", fetch_git)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    myosuite_init.fetch_simhive()

    fetch_git.assert_not_called()
    assert not (tmp_path / "simhive" / "simhive-version").exists()


def test_yes_myoapi(monkeypatch, tmp_path):
    simhive_path = tmp_path / "simhive"
    asset_path = simhive_path / "myo_model"
    target_xml = asset_path / "myoskeleton" / "myoskeleton.xml"
    target_xml.parent.mkdir(parents=True)
    target_xml.write_text("<mujoco/>")

    monkeypatch.setattr(myosuite_init, "simhive_path", str(simhive_path))
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    fetch_git = Mock(return_value=str(asset_path))
    monkeypatch.setattr(myosuite_init, "fetch_git", fetch_git)

    myosuite_init.fetch_simhive()

    fetch_git.assert_called_once()
    assert target_xml.is_file()
    assert (simhive_path / "simhive-version").read_text() == "2.5.0"

import json
from pathlib import Path

from sevcap.fireworks import extract_json
from sevcap.io_contract import ResultWriter, atomic_write_json, find_output_dir, list_videos
from sevcap.sampler import probe_duration, sample_keyframes


def test_extract_json_lenient():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("prefix [[1,2],[3]] suffix") == [[1, 2], [3]]


def test_result_writer_per_clip_and_combined(tmp_path):
    w = ResultWriter(tmp_path)
    w.write("clip1", {"formal": "f", "sarcastic": "s",
                      "humorous_tech": "ht", "humorous_non_tech": "hnt"})
    w.write("clip2", {"formal": "f2", "sarcastic": "s2",
                      "humorous_tech": "ht2", "humorous_non_tech": "hnt2"},
            meta={"stage": "sev-verified"})

    per = json.loads((tmp_path / "clip1.json").read_text())
    assert per["captions"]["formal"] == "f"
    combined = json.loads((tmp_path / "captions.json").read_text())
    assert len(combined["results"]) == 2
    assert combined["results"][1]["verification"]["stage"] == "sev-verified"

    # overwrite is atomic and replaces the record
    w.write("clip1", {"formal": "better", "sarcastic": "s",
                      "humorous_tech": "ht", "humorous_non_tech": "hnt"})
    combined = json.loads((tmp_path / "captions.json").read_text())
    assert len(combined["results"]) == 2
    assert json.loads((tmp_path / "clip1.json").read_text())["captions"]["formal"] == "better"


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"ok": True})
    assert json.loads(p.read_text()) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_find_output_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    assert find_output_dir() == Path(str(tmp_path / "out"))


def test_sampler_on_synthetic_clip(synthetic_clip, tmp_path):
    assert 34 <= probe_duration(synthetic_clip) <= 36
    frames = sample_keyframes(synthetic_clip, n_frames=6, workdir=str(tmp_path))
    assert 3 <= len(frames) <= 8
    ts = [f.t for f in frames]
    assert ts == sorted(ts)
    for f in frames:
        assert Path(f.path).stat().st_size > 0
        assert len(f.b64()) > 100


def test_list_videos(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.webm").write_bytes(b"x")
    vids = list_videos(tmp_path)
    assert [v.name for v in vids] == ["a.mp4", "c.webm"]

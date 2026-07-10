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
    w = ResultWriter(tmp_path, task_order=["clip1", "clip2"])
    w.write("clip1", {"formal": "f", "sarcastic": "s",
                      "humorous_tech": "ht", "humorous_non_tech": "hnt"})
    w.write("clip2", {"formal": "f2", "sarcastic": "s2",
                      "humorous_tech": "ht2", "humorous_non_tech": "hnt2"},
            meta={"stage": "sev-verified"})

    per = json.loads((tmp_path / "clip1.json").read_text())
    assert per["task_id"] == "clip1"
    assert per["captions"]["formal"] == "f"
    assert "verification" not in per

    per2 = json.loads((tmp_path / "clip2.json").read_text())
    assert per2["verification"]["stage"] == "sev-verified"

    harness = json.loads((tmp_path / "results.json").read_text())
    assert isinstance(harness, list)
    assert len(harness) == 2
    assert harness[0]["task_id"] == "clip1"
    assert "verification" not in harness[0]
    assert harness[1]["captions"]["formal"] == "f2"

    legacy = json.loads((tmp_path / "captions.json").read_text())
    assert legacy["results"][0]["clip"] == "clip1"

    # overwrite is atomic and replaces the record
    w.write("clip1", {"formal": "better", "sarcastic": "s",
                      "humorous_tech": "ht", "humorous_non_tech": "hnt"})
    harness = json.loads((tmp_path / "results.json").read_text())
    assert len(harness) == 2
    assert json.loads((tmp_path / "clip1.json").read_text())["captions"]["formal"] == "better"


def test_load_tasks_and_harness_schema(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "v1",
                    "video_url": "https://example.com/a.mp4",
                    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
                }
            ]
        )
    )
    from sevcap.io_contract import discover_tasks, load_tasks

    loaded = load_tasks(tasks_path)
    assert loaded[0].task_id == "v1"
    assert discover_tasks(tmp_path)[0].video_url.endswith("a.mp4")


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

from sevcap.config import (
    CLIP_DURATION_MAX_S,
    CLIP_DURATION_MIN_S,
    LONG_CLIP_THRESHOLD_S,
    clip_profile,
)


def test_clip_profile_endpoints():
    short = clip_profile(CLIP_DURATION_MIN_S)
    long = clip_profile(CLIP_DURATION_MAX_S)

    assert short.duration_s == CLIP_DURATION_MIN_S
    assert long.duration_s == CLIP_DURATION_MAX_S
    assert short.n_frames == 6
    assert long.n_frames == 12
    assert short.upgrade_timeout_s < long.upgrade_timeout_s
    assert short.long_form is False
    assert long.long_form is True


def test_clip_profile_midpoint():
    mid = clip_profile(75.0)
    assert mid.n_frames == 9
    assert mid.long_form is (75.0 >= LONG_CLIP_THRESHOLD_S)


def test_clip_profile_clamps():
    assert clip_profile(5.0).duration_s == CLIP_DURATION_MIN_S
    assert clip_profile(999.0).duration_s == CLIP_DURATION_MAX_S

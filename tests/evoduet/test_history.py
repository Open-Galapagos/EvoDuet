"""Tests for verbalize_history (evolutionary history -> prompt text)."""

from skydiscover.evoduet.history import verbalize_history


def _prog(**kw):
    base = dict(
        id="p1",
        solution="def f(): pass",
        language="python",
        metrics={"combined_score": 0.3642, "validity": 1.0, "sum_radii": 0.9598},
        iteration_found=0,
    )
    base.update(kw)
    return base


def test_verbalize_history_dict_of_programs():
    out = verbalize_history({"": [_prog()]})  # the shape the layer passes
    assert "### Program 1 (iteration 0, combined_score: 0.3642)" in out
    assert "Score breakdown:" in out
    assert "- validity: 1.0000" in out and "- sum_radii: 0.9598" in out
    assert "def f(): pass" not in out
    assert "combined_score" in out and "Program(" not in out  # no raw repr


def test_verbalize_history_groups_by_island_label():
    out = verbalize_history({"island_a": [_prog()], "island_b": [_prog(id="p2")]})
    assert "## island_a" in out and "## island_b" in out


def test_verbalize_history_empty_and_string_and_none():
    assert verbalize_history(None) == "(none)"
    assert verbalize_history({}) == "(none)"
    assert verbalize_history({"": []}) == "(none)"
    assert verbalize_history("already rendered") == "already rendered"


def test_verbalize_history_keeps_scaffold_rendered_history_verbatim():
    """A pre-rendered scaffold history is never truncated or re-rendered."""
    rendered = "## Previous Attempts\n\n### Attempt 1\n- Changes: " + "x" * 30_000

    assert verbalize_history(rendered) == rendered
    assert verbalize_history(rendered, max_chars=100) == rendered


def test_verbalize_history_flat_list():
    out = verbalize_history([_prog(), _prog(id="p2")])
    assert "### Program 1" in out and "### Program 2" in out


def test_verbalize_history_renders_whole_only_when_asked():
    programs = [
        {
            "iteration_found": i,
            "metrics": {"combined_score": i / 100},
            "metadata": {"changes": "c" * 200},
        }
        for i in range(200)
    ]

    out = verbalize_history(programs, max_chars=None)

    assert len(out) > 40_000 and "truncated" not in out
    assert "### Program 200" in out
    assert "truncated" in verbalize_history(programs)  # the default keeps the 12,000 cap
    assert len(verbalize_history(programs)) == 12_000


def test_verbalize_history_prefers_higher_is_better_proxy_and_has_total_cap():
    program = _prog(
        solution="secret implementation" * 10_000,
        metadata={"changes": "reworked\n```unsafe``` " + "z" * 1_000},
        metrics={
            "loss": 0.2,
            "evoduet_score": -0.2,
            "long_metric": "x" * 10_000,
        },
    )

    out = verbalize_history([program], max_chars=500)

    assert "evoduet_score: -0.2000 (higher is better)" in out
    assert "secret implementation" not in out
    assert "Changes: reworked ~~~unsafe~~~" in out
    assert "```unsafe```" not in out
    assert len(out) <= 500

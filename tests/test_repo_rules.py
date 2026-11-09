"""Static checks on the workflows and the tree: manual triggers (plus the one approved daily schedule), minimal
permissions, no third-party actions, secrets only from the environment, and nothing in the tree that is not
generic."""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF = os.path.join(ROOT, ".github", "workflows")


def _read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def _workflows():
    return {n: _read(os.path.join(WF, n)) for n in sorted(os.listdir(WF))}


def test_two_workflows():
    assert sorted(_workflows()) == ["backfill.yml", "daily.yml"]


def test_triggers_permissions_actions():
    for name, text in _workflows().items():
        live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        on_block = live.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        assert "workflow_dispatch:" in on_block, name
        banned = ["pull_request", "pull_request_target", "workflow_run", "issue_comment", "push:",
                  "repository_dispatch", "issues:"]
        if name != "daily.yml":
            banned.append("schedule:")  # only the daily append runs on a schedule
        for trig in banned:
            assert trig not in on_block, (name, trig)
        assert re.search(r"^permissions:\n  contents: read$", live, re.M), name
        assert "uses:" not in live, name
        assert "set -x" not in live, name
        assert "runs-on: ubuntu-24.04" in live and "latest" not in live, name


def test_daily_schedule_is_the_approved_one():
    text = _workflows()["daily.yml"]
    assert re.search(r"^  schedule:\n    - cron: \"13 11 \* \* \*\"$", text, re.M)
    assert len(re.findall(r"cron:", text)) == 1
    assert not re.search(r"^\s*schedule:", _workflows()["backfill.yml"], re.M)
    # the keepalive job stays: scheduled workflows of a public repo are disabled after 60 idle days
    assert "\n  keepalive:\n" in text and "actions/workflows/daily.yml/enable" in text


def test_secrets_only_in_env_blocks_and_environment_jobs():
    allowed = {"S3_ENDPOINT", "S3_BUCKET", "S3_KEY_ID", "S3_SECRET"}
    for name, text in _workflows().items():
        for m in re.finditer(r"\$\{\{\s*secrets\.([A-Z0-9_]+)\s*\}\}", text):
            assert m.group(1) in allowed, (name, m.group(1))
            line = text[text.rfind("\n", 0, m.start()) + 1:m.end()]
            assert re.match(r"^\s+(S3_ENDPOINT|S3_BUCKET|S3_KEY_ID|S3_SECRET): \$\{\{", line), line
        jobs = re.split(r"\n  (?=[a-z]+:\n)", text.split("\njobs:\n", 1)[1])
        for job in jobs:
            if "secrets." in job:
                assert re.search(r"\n    environment: archive\n", "\n" + job), (name, job[:40])


def test_inputs_never_interpolated_into_run_text():
    for name, text in _workflows().items():
        for block in re.findall(r"run: \|\n((?:\s{10,}.*\n)+)", text):
            assert "${{" not in block, name
        for line in re.findall(r"run: (.*)", text):
            assert "${{" not in line, (name, line)


def test_catalog_job_runs_before_the_lanes():
    for name, text in _workflows().items():
        assert "\n  catalog:\n" in text and "LANE: catalog" in text, name
        for job in ("noaa", "iem"):
            block = text.split(f"\n  {job}:\n", 1)[1].split("\n  ", 1)[0]
            assert "catalog" in block and block.startswith("    needs:"), (name, job)


def test_contact_from_repo_variable():
    for name, text in _workflows().items():
        assert "CONTACT_EMAIL: ${{ vars.CONTACT_EMAIL }}" in text, name


def test_lock_is_hash_pinned():
    lock = _read(os.path.join(ROOT, "requirements.lock"))
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^ \\]+)", lock, re.M)
    assert len(pins) >= 5
    for pkg, _ in pins:
        block = lock.split(f"{pkg}==", 1)[1].split("\n\n", 1)[0]
        assert "--hash=sha256:" in block, pkg
    req = _read(os.path.join(ROOT, "requirements.in"))
    assert {line.split("=")[0].strip() for line in req.splitlines() if line and not line.startswith("#")} == \
        {"numpy", "pygrib", "pyarrow", "zstandard", "boto3"}


def test_tree_is_generic():
    # the terms are split so that this file itself does not contain them
    terms = ["storm" "auditor", "supa" "base", "con" "nor", "wind" "-rnd", "workers" "[.]dev", "hz" "_",
             "weather" "-lab", "weather" "lab"]
    bad = re.compile("|".join(terms), re.I)
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".venv", ".pytest_cache")]
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                text = _read(p)
            except UnicodeDecodeError:
                continue
            assert not bad.search(text), p

"""Check transcribed aggregates, never infer model results or edit documents."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def table(path, first_headers):
    rows = []
    collecting = False
    for line in (ROOT / path).read_text().splitlines():
        if not line.startswith("|"):
            if collecting:
                break
            continue
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        if not collecting:
            collecting = cells[: len(first_headers)] == first_headers
        elif not all(re.fullmatch(r"[-: ]+", cell) for cell in cells):
            rows.append(cells)
    assert collecting and rows, (path, first_headers)
    return rows


def success(value):
    match = re.fullmatch(r"(\d+)/(\d+) \((\d+\.\d+)%\)", value)
    assert match, value
    count, planned = map(int, match.group(1, 2))
    assert match[3] == f"{100 * count / planned:.1f}"
    return count, planned


@pytest.fixture
def figures(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "report_figures", ROOT / "docs/figures/render.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    captured = {}

    def capture(name, title, subtitle, rows, series, footnote, **kwargs):
        captured[name] = rows

    monkeypatch.setattr(renderer, "architecture", lambda: None)
    for name in ("vertical_bars", "grouped_success_bars", "bars"):
        monkeypatch.setattr(renderer, name, capture)
    renderer.main()  # All file-writing rendering functions are replaced above.
    return captured


@pytest.mark.parametrize("document", ["README.md", "REPORT.md"])
def test_condition_tables_match_figure_counts_and_rounding(document, figures):
    rows = table(document, ["Role hints", "Binding", "Request-only retrieval"])
    plotted = figures["04-condition-success.svg"]
    assert len(rows) == 4
    assert len(plotted) == 3
    for arm, row in enumerate(rows):
        assert (
            tuple(row[:2])
            == [("Off", "Off"), ("Off", "On"), ("On", "Off"), ("On", "On")][arm]
        )
        for condition, value in enumerate(row[2:]):
            assert success(value) == (plotted[condition][2][arm], plotted[condition][1])


def test_supporting_condition_counts_reconcile_with_main_tables():
    main = table("REPORT.md", ["Role hints", "Binding", "Request-only retrieval"])
    details = table("docs/study-details.md", ["Role hints", "Binding", "Condition"])
    pooled = table("REPORT.md", ["Hints", "Binding", "Success"])
    assert len(details) == 12 and len(pooled) == 4
    for arm in range(4):
        group = details[arm * 3 : (arm + 1) * 3]
        for condition, row in enumerate(group):
            counts = list(map(int, row[3:6]))
            assert sum(counts) == 123
            assert counts[0] == success(main[arm][condition + 2])[0]
            assert row[6] == f"{100 * counts[0] / 123:.1f}%"
        totals = [sum(int(row[i]) for row in group) for i in range(3, 6)]
        assert totals == list(map(int, pooled[arm][2:5]))
        assert sum(totals) == 369
        assert pooled[arm][5] == f"{100 * totals[0] / 369:.1f}%"
    assert sum(int(row[5]) for row in details) == 118


def normalize(value):
    return " ".join(value.lower().replace("-", " ").split())


def test_task_tables_match_figure_counts_and_rounding(figures):
    main = table("REPORT.md", ["Task category"])
    details = table("docs/study-details.md", ["Capability", "Hints off, binding off"])
    plotted = figures["05-task-success.svg"]
    assert len(main) == len(details) == len(plotted) == 7
    for row, supporting, (label, planned, counts) in zip(
        main, details, plotted, strict=True
    ):
        assert normalize(row[0]) == normalize(supporting[0]) == normalize(label)
        assert row[1:] == supporting[1:]
        assert [success(value) for value in row[1:]] == [
            (count, planned) for count in counts
        ]
    assert sum(row[1] for row in plotted) == 369


def test_extraction_counts_percentages_and_figures_match(figures):
    details = table("docs/study-details.md", ["Group", "TP", "FP", "FN"])
    by_group = {row[0]: tuple(map(int, row[1:4])) for row in details}

    def metrics(group):
        tp, fp, fn = by_group[group]
        return 100 * tp / (tp + fp), 100 * tp / (tp + fn), 200 * tp / (2 * tp + fp + fn)

    for row in details:
        assert row[4:] == [f"{value:.2f}%" for value in metrics(row[0])]
    types = ["PERSON", "DATE", "EVENT", "TOPIC"]
    for column in range(3):
        assert sum(by_group[k][column] for k in types) == by_group["Overall"][column]
        assert (
            sum(by_group[k][column] for k in ["Clean", "Typo", "Simulated ASR"])
            == by_group["Overall"][column]
        )
    for row in table("REPORT.md", ["Entity type", "Reference entities"]):
        group = "Overall" if row[0] == "Overall (micro)" else row[0]
        tp, _, fn = by_group[group]
        assert int(row[1]) == tp + fn
        assert row[2:] == [f"{value:.2f}%" for value in metrics(group)]
    for row in table("REPORT.md", ["Input version", "Requests"]):
        assert int(row[1]) == 41
        assert row[2:] == [f"{value:.2f}%" for value in metrics(row[0])]
    for filename in ["02-entity-types.svg", "03-extraction-noise.svg"]:
        for label, values in figures[filename]:
            assert values == pytest.approx(metrics(label))


def test_public_document_local_links_and_anchors_exist():
    paths = [
        ROOT / name
        for name in [
            "README.md",
            "REPORT.md",
            "DATA-LICENSE.md",
            "THIRD-PARTY-NOTICES.md",
            "SECURITY.md",
        ]
    ]
    paths += list((ROOT / "docs").glob("*.md"))
    for path in paths:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if re.match(r"^[a-z]+:", link):
                continue
            filename, _, anchor = link.partition("#")
            target = path.parent / filename if filename else path
            assert target.exists(), (path.name, link)
            if anchor and target.suffix == ".md":
                headings = re.findall(r"^#+ (.+)$", target.read_text(), re.MULTILINE)
                slugs = {
                    re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
                    for h in headings
                }
                assert anchor in slugs, (path.name, link)


def test_report_has_clean_image_alt_text_and_no_proposal_markers():
    text = (ROOT / "REPORT.md").read_text()
    assert "%%" not in text
    alts = re.findall(r"!\[([^\]]*)\]\(", text)
    assert len(alts) == 6 and all(alts)
    assert not any(re.search(r"\|\d+$", alt) for alt in alts)

"""Render report figures from transcribed, already-measured aggregate values.

Run with Python 3.11: python docs/figures/render.py. No model or research data
is loaded. See docs/study-details.md for counts and evidence availability.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK = "#17263d"
MUTED = "#526579"
GRID = "#dae2e9"
BLUE = "#215b91"
TEAL = "#008477"
GOLD = "#b56313"
PURPLE = "#745293"
WIDTH = 960
MARGIN = 36


def text(x, y, value, size=16, color=INK, anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(str(value))}</text>'
    )


def rect(x, y, width, height, color, rx=0):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" fill="{color}"/>'


def line(x1, y1, x2, y2, color=GRID):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"/>'


def heading(title, subtitle):
    return text(MARGIN, 42, title, 26, weight="bold") + text(
        MARGIN, 70, subtitle, 16, MUTED
    )


def legend(series):
    """Keep each key beside its label instead of using fixed-width slots."""
    content = ""
    x = MARGIN
    for label, color in series:
        content += rect(x, 96, 14, 14, color, 2)
        content += text(x + 22, 109, label, 16)
        # Conservative Arial label width, followed by a 28 px inter-key gap.
        x += 22 + len(label) * 9 + 28
    return content


def extraction_metrics(tp, fp, fn):
    """Derive display percentages from the published diagnostic counts."""
    return 100 * tp / (tp + fp), 100 * tp / (tp + fn), 100 * 2 * tp / (2 * tp + fp + fn)


def svg(name, width, height, title, description, content):
    source = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{escape(title)}</title>'
        f'<desc id="desc">{escape(description)}</desc>'
        f"{rect(0, 0, width, height, '#ffffff')}{content}</svg>\n"
    )
    (OUT / name).write_text(source, encoding="utf-8")


def architecture():
    content = heading(
        "From request to grounded decision",
        "The original request is retained in all three conditions",
    )
    boxes = [
        (36, 112, 184, "Request +", "eligible records"),
        (268, 112, 184, "Evidence", "retrieval"),
        (500, 112, 184, "Local Qwen", "decision"),
        (732, 112, 184, "Validation +", "saved result"),
    ]
    for x, y, w, line1, line2 in boxes:
        content += rect(x, y, w, 80, "#edf3f8", 8)
        content += text(x + w / 2, y + 32, line1, 18, INK, "middle", "bold")
        content += text(x + w / 2, y + 58, line2, 18, INK, "middle")
    for x in (244, 476, 708):
        content += text(x, 161, "→", 28, BLUE, "middle")
    content += rect(268, 260, 184, 68, "#e2f3ee", 8)
    content += text(360, 288, "XLM-R extractor", 18, TEAL, "middle", "bold")
    content += text(360, 312, "optional", 16, TEAL, "middle")
    content += line(360, 256, 360, 204, TEAL)
    content += line(360, 204, 355, 212, TEAL)
    content += line(360, 204, 365, 212, TEAL)
    content += text(376, 236, "Informs retrieval", 15, TEAL)
    content += line(456, 294, 592, 294, TEAL)
    content += line(592, 294, 592, 204, TEAL)
    content += line(592, 204, 587, 212, TEAL)
    content += line(592, 204, 597, 212, TEAL)
    content += text(484, 318, "Annotations may", 15, TEAL)
    content += text(484, 340, "also go to Qwen", 15, TEAL)
    content += text(732, 246, "Optional recipient", 16, PURPLE, weight="bold")
    content += text(732, 268, "binding", 16, PURPLE, weight="bold")
    content += text(732, 294, "After generation;", 16, MUTED)
    content += text(732, 318, "does not write the draft.", 16, MUTED)
    content += text(
        MARGIN,
        384,
        "Output: draft  ·  clarification  ·  unsupported    |    No messages are sent",
        16,
        MUTED,
    )
    svg(
        "01-architecture.svg",
        WIDTH,
        408,
        "Messaging assistant architecture",
        "Request and records pass through retrieval, Qwen and validation; optional extraction informs retrieval and annotations, and optional binding acts after generation.",
        content,
    )


def vertical_bars(name, title, subtitle, rows, series, footnote, maximum=100):
    """Short category names fit beneath columns without a wide label gutter."""
    left, right, top, bottom = 84, 924, 156, 396
    height = 488
    content = heading(title, subtitle) + legend(series)
    for tick in range(0, maximum + 1, 20):
        y = bottom - (bottom - top) * tick / maximum
        content += line(left, y, right, y)
        content += text(left - 12, y + 5, f"{tick}%", 14, MUTED, "end")
    group_width = (right - left) / len(rows)
    bar_width = 32 if len(series) == 3 else 48
    gap = 24
    cluster_width = len(series) * bar_width + (len(series) - 1) * gap
    for i, (label, values) in enumerate(rows):
        center = left + (i + 0.5) * group_width
        content += text(center, bottom + 32, label, 16, anchor="middle")
        for j, ((_, color), value) in enumerate(zip(series, values, strict=True)):
            x = center - cluster_width / 2 + j * (bar_width + gap)
            bar_height = (bottom - top) * value / maximum
            content += rect(x, bottom - bar_height, bar_width, bar_height, color)
            content += text(
                x + bar_width / 2,
                bottom - bar_height - 10,
                f"{value:.1f}%",
                14,
                anchor="middle",
            )
    content += text(MARGIN, height - 20, footnote, 14, MUTED)
    svg(name, WIDTH, height, title, subtitle + " " + footnote, content)


def bars(name, title, subtitle, rows, series, footnote, maximum=100):
    """Long task and arm labels remain easier to scan beside horizontal bars."""
    left, right, top = 224, 892, 148
    group_height = 88 if len(series) == 3 else 64
    bottom = top + len(rows) * group_height - 20
    height = bottom + 84
    content = heading(title, subtitle) + legend(series)
    for tick in range(0, maximum + 1, 20):
        x = left + (right - left) * tick / maximum
        content += line(x, top - 8, x, bottom)
        content += text(x, bottom + 28, f"{tick}%", 14, MUTED, "middle")
    for i, (label, values) in enumerate(rows):
        y = top + i * group_height
        label_lines = label.split(" · ") if len(series) == 3 else [label]
        center = y + ((len(series) - 1) * 22 + 16) / 2
        for k, label_line in enumerate(label_lines):
            content += text(
                left - 16,
                center + 5 + (k - (len(label_lines) - 1) / 2) * 22,
                label_line,
                16,
                anchor="end",
            )
        for j, ((_, color), value) in enumerate(zip(series, values, strict=True)):
            yy = y + j * 22
            bar_width = (right - left) * value / maximum
            if value:
                content += rect(left, yy, bar_width, 16, color)
            else:
                # A zero-width bar is invisible; retain a colored zero marker.
                content += line(left, yy, left, yy + 16, color)
            content += text(left + bar_width + 8, yy + 13, f"{value:.1f}%", 14)
    content += text(MARGIN, height - 20, footnote, 14, MUTED)
    svg(name, WIDTH, height, title, subtitle + " " + footnote, content)


def grouped_success_bars(name, title, subtitle, rows, series, footnote, maximum=100):
    """Compare all four arms in one plot, with success percentages on the y-axis."""
    left, right, top, bottom = 100, 924, 194, 524
    height = 672
    content = heading(title, subtitle)
    descriptions = [subtitle]
    for i, (label, color) in enumerate(series):
        x = left + (i % 2) * (right - left) / 2
        y = 98 + (i // 2) * 28
        content += rect(x, y, 14, 14, color, 2)
        content += text(x + 22, y + 13, label, 15)
    for tick in range(0, maximum + 1, 20):
        y = bottom - (bottom - top) * tick / maximum
        content += line(left, y, right, y)
        content += text(left - 12, y + 5, f"{tick}%", 14, MUTED, "end")
    content += (
        f'<g transform="translate(28 {(top + bottom) / 2}) rotate(-90)">'
        + text(0, 0, "Semantic success (%)", 15, MUTED, "middle")
        + "</g>"
    )
    group_width = (right - left) / len(rows)
    gap = 4 if len(rows) > 4 else 12
    bar_width = min(40, (group_width * 0.8 - (len(series) - 1) * gap) / len(series))
    cluster_width = len(series) * bar_width + (len(series) - 1) * gap
    for i, (label, planned, counts) in enumerate(rows):
        center = left + (i + 0.5) * group_width
        plain_label = label.replace("\n", " ")
        for k, label_line in enumerate(label.split("\n")):
            content += text(
                center, bottom + 30 + k * 20, label_line, 15, anchor="middle"
            )
        content += text(center, bottom + 78, f"n = {planned}", 12, MUTED, "middle")
        for j, ((arm, color), successes) in enumerate(zip(series, counts, strict=True)):
            value = 100 * successes / planned
            x = center - cluster_width / 2 + j * (bar_width + gap)
            bar_height = (bottom - top) * value / maximum
            description = f"{plain_label}, {arm}: {successes}/{planned} ({value:.1f}%)."
            content += f"<g><title>{escape(description)}</title>"
            if successes:
                content += rect(x, bottom - bar_height, bar_width, bar_height, color)
            else:
                content += line(x, bottom, x + bar_width, bottom, color)
            # Narrow groups omit the percent suffix; the y-axis supplies the unit.
            content += text(
                x + bar_width / 2,
                bottom - bar_height - 9,
                f"{value:.1f}" if len(rows) > 4 else f"{value:.1f}%",
                11 if len(rows) > 4 else 14,
                anchor="middle",
            )
            content += "</g>"
            descriptions.append(description)
    content += text(MARGIN, height - 20, footnote, 14, MUTED)
    svg(name, WIDTH, height, title, " ".join(descriptions), content)


def main():
    architecture()
    vertical_bars(
        "02-entity-types.svg",
        "Entity extraction by type",
        "Exact span and type matching; micro-aggregated across 123 requests.",
        [
            ("PERSON", extraction_metrics(171, 31, 28)),
            ("DATE", extraction_metrics(9, 0, 9)),
            ("EVENT", extraction_metrics(34, 16, 23)),
            ("TOPIC", extraction_metrics(34, 68, 35)),
        ],
        [("Precision", BLUE), ("Recall", TEAL), ("F1", GOLD)],
        "Source: retained extraction diagnostic; see docs/study-details.md.",
    )
    vertical_bars(
        "03-extraction-noise.svg",
        "Entity extraction under noise",
        "Exact span and type matching; 41 requests per variant.",
        [
            ("Clean", extraction_metrics(93, 26, 22)),
            ("Typo", extraction_metrics(88, 37, 27)),
            ("Simulated ASR", extraction_metrics(67, 52, 46)),
        ],
        [("Precision", BLUE), ("Recall", TEAL), ("F1", GOLD)],
        "ASR errors are simulated in text; all three groups represent the same scenarios.",
    )
    condition_labels = [
        "Request-only\nretrieval",
        "Entity-informed\nretrieval",
        "Entity-informed retrieval\n+ annotations",
    ]
    arms = [
        ("Hints off, binding off", BLUE),
        ("Hints off, binding on", TEAL),
        ("Hints on, binding off", GOLD),
        ("Hints on, binding on", PURPLE),
    ]
    condition_successes = [(23, 26, 38), (21, 30, 42), (47, 49, 40), (49, 49, 44)]
    grouped_success_bars(
        "04-condition-success.svg",
        "Semantic success by condition",
        "All four hint/binding arms; 123 planned requests per condition.",
        [
            (label, 123, tuple(counts[i] for counts in condition_successes))
            for i, label in enumerate(condition_labels)
        ],
        arms,
        "n = planned requests per arm. Exact success counts are in the report tables.",
        maximum=60,
    )
    task_counts = [
        ("Core\nsingle-hop", 36, (19, 20, 32, 34)),
        ("Identity\nambiguity", 63, (30, 27, 44, 43)),
        ("Third-party\nattendance", 81, (15, 17, 41, 40)),
        ("Temporal\nprecision", 54, (17, 21, 17, 21)),
        ("Multi-hop\nbook", 72, (5, 6, 2, 3)),
        ("Policy\nsafety", 36, (1, 2, 0, 1)),
        ("Missing\nfact", 27, (0, 0, 0, 0)),
    ]
    grouped_success_bars(
        "05-task-success.svg",
        "Semantic success by task category",
        "All four hint/binding arms; three conditions pooled within each category.",
        task_counts,
        arms,
        "n = planned cells per arm; success values are percentages. Exact counts are in the report.",
    )
    bars(
        "06-arm-diagnostics.svg",
        "Technical and mechanical diagnostics",
        "All three conditions pooled; 369 planned cells per arm.",
        [
            ("Hints off · binding off", (86.2, 56.9, 50.9)),
            ("Hints off · binding on", (86.2, 56.9, 53.1)),
            ("Hints on · binding off", (90.0, 74.3, 69.1)),
            ("Hints on · binding on", (90.0, 74.3, 70.2)),
        ],
        [
            ("Technical validity", BLUE),
            ("Correct action", TEAL),
            ("Mechanical success", GOLD),
        ],
        "These checks are not semantic task success; binding reuses the same generations.",
    )


if __name__ == "__main__":
    main()

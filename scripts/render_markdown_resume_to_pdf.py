from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0
LEFT = 0.65
TOP = 10.45
BOTTOM = 0.65


def parse_markdown(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            blocks.append(("blank", ""))
            continue
        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
        elif line.startswith("### "):
            blocks.append(("subheading", line[4:].strip()))
        elif line.startswith("## "):
            blocks.append(("heading", line[3:].strip()))
        elif line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
        else:
            blocks.append(("paragraph", line))
    return blocks


def style_for(kind: str) -> tuple[float, float, str]:
    if kind == "title":
        return 20.0, 1.45, "bold"
    if kind == "heading":
        return 13.0, 1.15, "bold"
    if kind == "subheading":
        return 11.5, 1.1, "bold"
    if kind == "bullet":
        return 9.5, 1.08, "normal"
    if kind == "paragraph":
        return 9.5, 1.08, "normal"
    return 9.0, 0.8, "normal"


def wrap_block(kind: str, content: str) -> list[str]:
    if kind == "blank":
        return [""]

    width_map = {
        "title": 60,
        "heading": 68,
        "subheading": 78,
        "paragraph": 92,
        "bullet": 86,
    }
    width = width_map.get(kind, 92)
    prefix = ""
    subsequent = ""
    if kind == "bullet":
        prefix = "- "
        subsequent = "  "
    wrapped = textwrap.wrap(
        content,
        width=width,
        initial_indent=prefix,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [prefix.rstrip()]


def new_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_WIDTH)
    ax.set_ylim(0, PAGE_HEIGHT)
    ax.axis("off")
    return fig, ax, TOP


def render_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8")
    blocks = parse_markdown(text)

    with PdfPages(output_path) as pdf:
        fig, ax, y = new_page(pdf)

        for kind, content in blocks:
            font_size, line_factor, font_weight = style_for(kind)
            lines = wrap_block(kind, content)
            line_height = font_size / 72.0 * line_factor
            needed_height = line_height * max(len(lines), 1)

            if y - needed_height < BOTTOM:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
                plt.close(fig)
                fig, ax, y = new_page(pdf)

            for line in lines:
                if kind == "blank":
                    y -= 0.10
                    continue
                x = LEFT + (0.08 if kind == "bullet" else 0.0)
                ax.text(
                    x,
                    y,
                    line,
                    fontsize=font_size,
                    fontweight=font_weight,
                    fontfamily="DejaVu Sans",
                    va="top",
                    ha="left",
                    color="#111111",
                )
                y -= line_height

            if kind == "title":
                y -= 0.08
            elif kind in {"heading", "subheading"}:
                y -= 0.03
            else:
                y -= 0.02

        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a simple markdown resume to PDF.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_markdown_to_pdf(args.input, args.output)


if __name__ == "__main__":
    main()

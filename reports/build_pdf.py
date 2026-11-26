"""Markdown report -> PDF with the layout of the course reports.

    python reports/build_pdf.py reports/final_report.md report.pdf

Needs the Python package `markdown` and Chrome or Chromium (used headless to print the page).
The HTML is written next to the Markdown file, so its relative image links work, and removed afterwards.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 13mm 14mm 14mm 14mm; }
* { box-sizing: border-box; }
body { font-family: Inter, "Helvetica Neue", Arial, sans-serif; font-size: 8.6pt; line-height: 1.38; color: #1b1f24; margin: 0; }
h1 { font-size: 15.5pt; margin: 0 0 3pt 0; letter-spacing: -0.2pt; }
h2 { font-size: 10.8pt; margin: 11pt 0 3pt 0; padding-top: 5pt; border-top: 0.6pt solid #c9ced6; break-after: avoid; }
h3 { font-size: 9.2pt; margin: 7pt 0 2pt 0; break-after: avoid; }
p { margin: 0 0 4.5pt 0; }
ul, ol { margin: 0 0 4.5pt 0; padding-left: 13pt; }
li { margin-bottom: 1.5pt; }
a { color: #1f5fa8; text-decoration: none; }
code { font-family: "JetBrains Mono", "DejaVu Sans Mono", Menlo, monospace; font-size: 7.6pt; background: #f1f3f5; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 5pt 0; font-size: 7.9pt; break-inside: avoid; }
th { text-align: left; border-bottom: 0.9pt solid #1b1f24; padding: 2.2pt 5pt 2.2pt 0; font-weight: 600; vertical-align: bottom; }
td { border-bottom: 0.4pt solid #d5d9df; padding: 2pt 5pt 2pt 0; vertical-align: top; }
.meta { color: #59616b; margin-bottom: 6pt; }
.small { font-size: 7.4pt; color: #59616b; margin-top: -1pt; }
.links { border: 0.6pt solid #c9ced6; border-radius: 3pt; padding: 5pt 8pt 2pt 8pt; margin: 5pt 0 6pt 0; font-size: 8pt; }
.box { border-left: 2.2pt solid #1f5fa8; background: #f4f7fb; padding: 4pt 8pt 1pt 8pt; margin: 4pt 0 6pt 0; }
.two { display: flex; gap: 10pt; align-items: flex-start; margin: 3pt 0; break-inside: avoid; }
.two > div { flex: 1; }
.fig { break-inside: avoid; margin: 3pt 0; }
.fig img, img { max-width: 100%; }
.pagebreak { break-before: page; }
"""

CHROME = ["google-chrome", "chromium", "chromium-browser", "chrome",
          "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]


def main(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    body = markdown.markdown(source.read_text(encoding="utf-8"),
                             extensions=["tables", "md_in_html", "attr_list", "sane_lists"])
    html = source.with_suffix(".print.html")
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{source.stem}</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    chrome = next((c for c in CHROME if shutil.which(c) or Path(c).exists()), None)
    if chrome is None:
        raise SystemExit(f"Chrome or Chromium not found; open {html} in a browser and print it to PDF instead")
    try:
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={target}", html.as_uri()], check=True, capture_output=True, timeout=180)
    finally:
        html.unlink(missing_ok=True)
    print(f"{target} written")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])

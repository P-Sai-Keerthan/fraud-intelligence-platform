"""Builds reports/PROJECT_AUDIT_AND_TECHNICAL_DOCUMENTATION.{md,pdf} from tools/main_template.md.

usage: python build_pdf.py <project_root>
Needs: markdown, pygments, pypdf, playwright (Chromium).  The Markdown file is the single source: the PDF is rendered from the very text written to it.
"""
import base64
import html
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import date
from pathlib import Path

import markdown
from pypdf import PdfReader
from playwright.sync_api import sync_playwright

ROOT = Path(sys.argv[1]).resolve()
REPORTS = ROOT / "reports"
rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
today = date.today().isoformat()

# ------------------------------------------------------------------ assemble the Markdown source
text = (REPORTS / "tools" / "main_template.md").read_text()
text = text.replace("@@DATE@@", today).replace("@@REV@@", rev)
text = re.sub(r"@@FILE:([^@]+)@@", lambda m: (REPORTS / m.group(1)).read_text().strip(), text)
inv = [line.split() for line in (REPORTS / "evidence/output/test_inventory_after_fixes.txt").read_text().splitlines() if line.strip()]
rows = ["| Test module | Tests |", "|---|---|"] + [f"| `{name.replace('tests/', '')}` | {n} |" for n, name in inv]
rows.append(f"| **Total** | **{sum(int(n) for n, _ in inv)}** |")
text = text.replace("@@TESTINV@@", "\n".join(rows))

head_re = re.compile(r"^# (.+)$", re.M)
no_code = re.sub(r"```.*?```", "", text, flags=re.S)                  # "# ..." lines inside code blocks are not headings
chapters = [m.group(1) for m in head_re.finditer(no_code)][1:]          # first H1 is the title
toc_md = "## Contents\n\n" + "\n".join(f"- {c}" for c in chapters) + "\n"
md_final = text.replace("@@TOC@@", toc_md)
(REPORTS / "PROJECT_AUDIT_AND_TECHNICAL_DOCUMENTATION.md").write_text(md_final)

# ------------------------------------------------------------------ Markdown -> HTML
MD = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "attr_list", "sane_lists"],
                       extension_configs={"codehilite": {"guess_lang": False, "css_class": "hl"}})
cover_md, rest_md = md_final.split(toc_md, 1)
slug = lambda s: "ch-" + re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def render(md_text):
    MD.reset()
    return MD.convert(md_text)


def embed_images(h):
    def sub(m):
        path = REPORTS / m.group(1)
        return f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"'
    h = re.sub(r'src="(assets/[^"]+\.png)"', sub, h)
    return h.replace("<p><em>Figure", '<p class="caption"><em>Figure')


def with_ids(h):
    return re.sub(r"<h1>(.+?)</h1>", lambda m: f'<h1 id="{slug(html.unescape(re.sub("<[^>]+>", "", m.group(1))))}">{m.group(1)}</h1>', h)


body = embed_images(with_ids(render(rest_md)))
cover = render(cover_md)
PYG = subprocess.run([sys.executable, "-m", "pygments", "-S", "default", "-f", "html", "-a", ".hl"], capture_output=True, text=True).stdout
CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.45; color: #1b1b1a; }
h1 { font-size: 17pt; color: #0e4f78; border-bottom: 2px solid #0e4f78; padding-bottom: 3px; margin: 22px 0 10px; break-after: avoid; }
h2 { font-size: 12.5pt; color: #0e4f78; margin: 16px 0 6px; break-after: avoid; } h3 { font-size: 10.5pt; margin: 12px 0 4px; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 8.3pt; break-inside: auto; }
tr { break-inside: avoid; } thead { display: table-header-group; }
th { background: #e8eff5; text-align: left; } th, td { border: 1px solid #c9d1d8; padding: 3px 5px; vertical-align: top; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; background: #f1f1ee; padding: 0 2px; border-radius: 2px; }
pre { background: #f7f7f4; border: 1px solid #deded8; border-radius: 3px; padding: 6px 8px; overflow: hidden; white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.8pt; }
img { max-width: 100%; display: block; margin: 8px auto 2px; break-inside: avoid; } p.caption { text-align: center; color: #52514e; font-size: 8.4pt; margin-bottom: 10px; }
.cover { padding-top: 70mm; } .cover h1 { font-size: 26pt; border: 0; margin-bottom: 0; } .cover h2 { font-size: 15pt; color: #52514e; margin-top: 4px; }
.cover table { width: 80%; margin-top: 30mm; font-size: 9.4pt; } .pagebreak { break-after: page; }
.toc li { list-style: none; margin: 3px 0; display: flex; } .toc ul { padding-left: 0; } .toc .t { flex: 1; border-bottom: 1px dotted #999; margin: 0 6px 3px; } .toc .p { min-width: 22px; text-align: right; }
"""


def build_toc(pages):
    items = "".join(f'<li><span>{html.escape(c)}</span><span class="t"></span><span class="p">{pages.get(c, "")}</span></li>' for c in chapters)
    return f'<div class="toc"><h2>Contents</h2><ul>{items}</ul></div>'


def doc(pages):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>Project Audit and Technical Documentation</title><style>{CSS}{PYG}</style></head><body>'
            f'<div class="cover pagebreak">{cover}</div><div class="pagebreak">{build_toc(pages)}</div>{body}</body></html>')


FOOTER = ("<div style='font-size:7.5px;color:#777;width:100%;padding:0 16mm;display:flex;justify-content:space-between'>"
          f"<span>Explainable Fraud Intelligence Platform - audit and technical documentation - revision {rev}</span>"
          "<span>page <span class='pageNumber'></span> of <span class='totalPages'></span></span></div>")


def render_pdf(pages, out):
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
        pg = b.new_page()
        pg.set_content(doc(pages), wait_until="load")
        pg.pdf(path=str(out), format="A4", print_background=True, display_header_footer=True, header_template="<div></div>", footer_template=FOOTER,
               margin={"top": "18mm", "bottom": "20mm", "left": "16mm", "right": "16mm"})
        b.close()


def locate(pdf_path):
    pages, reader = {}, PdfReader(str(pdf_path))
    texts = [(pg.extract_text() or "") for pg in reader.pages]
    for c in chapters:
        for i in range(2, len(texts)):                     # skip cover and contents
            if c in re.sub(r"\s+", " ", unicodedata.normalize("NFKC", texts[i])):
                pages[c] = i + 1
                break
    return pages


out = REPORTS / "PROJECT_AUDIT_AND_TECHNICAL_DOCUMENTATION.pdf"
tmp = Path(tempfile.gettempdir()) / "audit_report_pass1.pdf"       # first pass only, to find the page of each chapter
render_pdf({}, tmp)
pages = locate(tmp)
render_pdf(pages, out)
pages2 = locate(out)
missing = [c for c in chapters if c not in pages2]
print(f"pages: {len(PdfReader(str(out)).pages)} | chapters located: {len(pages2)}/{len(chapters)} | toc stable: {pages == pages2} | missing: {missing}")

"""Body word count for report.md per the coursework counting rules:
counts body prose only -- excludes YAML, headings, images, figure
captions, sidenote references, raw LaTeX blocks, tables (and table
captions), and everything from '# Appendix' on.
"""

import re
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "report.md").read_text()
src = re.sub(r"^---\n.*?\n---\n", "", src, count=1, flags=re.S)  # YAML
src = re.sub(r"\n# Appendix.*$", "", src, flags=re.S)  # appendix
src = re.sub(r"```\{=latex\}.*?```", "", src, flags=re.S)  # raw blocks
src = re.sub(r"\^\[.*?\]", "", src, flags=re.S)  # sidenotes
lines = [
    l
    for l in src.splitlines()
    if not l.startswith("#")
    and not l.startswith("![")
    and not l.startswith("|")
    and not l.startswith("Table:")
    and not l.startswith(": ")
]
text = "\n".join(lines).replace("*", "").replace("`", "")
print(len([t for t in text.split() if re.search(r"[A-Za-z0-9]", t)]))

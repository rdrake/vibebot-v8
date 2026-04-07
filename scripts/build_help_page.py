#!/usr/bin/env python3
"""Build the HTML help page from the command registry.

Used by the GitHub Pages deployment workflow to generate _site/index.html.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Add plugin source to path so we can import the registry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "llm" / "src"))

from llm.plugin import COMMAND_REGISTRY  # noqa: E402

_CATEGORY_LABELS = {"generation": "Generation", "memory": "Memory", "utility": "Utility"}

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Bot Commands</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    padding: 3rem 1.5rem;
    background: #fff;
    color: #1a1a2e;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif;
    font-size: 16px;
    line-height: 1.7;
    max-width: 720px;
    margin: 0 auto;
}
h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 2.5rem;
}
h2 {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}
.cmd {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.15s;
}
.cmd:hover { box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.cmd-header {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.command {
    font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a2e;
}
.param {
    font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
    font-size: 0.85rem;
    color: #9ca3af;
}
.cmd p {
    color: #4b5563;
    font-size: 0.925rem;
    margin-bottom: 0.5rem;
}
.examples {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}
.examples code {
    font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
    font-size: 0.8rem;
    background: #f3f4f6;
    color: #374151;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
}
.features {
    list-style: none;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.features li {
    font-size: 0.9rem;
    color: #4b5563;
    padding-left: 1.25rem;
    position: relative;
}
.features li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0.55rem;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #d1d5db;
}
.note {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #6b7280;
}
@media (max-width: 600px) {
    body { padding: 1.5rem 1rem; }
    .features { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<h1>LLM Bot Commands</h1>
<p class="subtitle">AI-powered IRC bot commands using LiteLLM.</p>
"""

_HTML_FOOT = """
<h2>Features</h2>
<ul class="features">
<li>Volatile memory for follow-up questions</li>
<li>Non-volatile memory across conversations</li>
<li>Vision support with image URLs</li>
<li>Syntax-highlighted code output</li>
<li>Spontaneous channel participation</li>
<li>Multi-provider AI via LiteLLM</li>
</ul>

<h2>Configuration</h2>
<div class="note">
Configuration is managed by the bot operator via Limnoria's config system.
Commands require the appropriate capability (e.g., <code>llm.ask</code>).
</div>

</body>
</html>"""


def _build_commands_html() -> str:
    """Build the command sections from the registry."""
    sections: list[str] = []
    for category in ("generation", "memory", "utility"):
        cmds = [c for c in COMMAND_REGISTRY if c.category == category]
        if not cmds:
            continue
        sections.append(f"<h2>{_CATEGORY_LABELS[category]}</h2>")
        for cmd in cmds:
            escaped_args = html.escape(cmd.args)
            examples = "".join(f"<code>{html.escape(ex)}</code>" for ex in cmd.examples)
            sections.append(
                f'<div class="cmd">'
                f'<div class="cmd-header">'
                f'<span class="command">%{cmd.name}</span> '
                f'<span class="param">{escaped_args}</span>'
                f"</div>"
                f"<p>{html.escape(cmd.description)}</p>"
                f'<div class="examples">{examples}</div>'
                f"</div>"
            )
    return "\n".join(sections)


def main() -> None:
    """Generate _site/index.html from the command registry."""
    out_dir = Path("_site")
    out_dir.mkdir(exist_ok=True)

    page = _HTML_HEAD + _build_commands_html() + _HTML_FOOT
    out_file = out_dir / "index.html"
    out_file.write_text(page, encoding="utf-8")
    print(f"Wrote {out_file} ({len(page)} bytes)")


if __name__ == "__main__":
    main()

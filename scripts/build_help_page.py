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
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 20px;
    background: #272822;
    color: #f8f8f2;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
}
h1 { color: #f8f8f2; margin-bottom: 0.5em; }
h2 { color: #a6e22e; margin-top: 1.5em; border-bottom: 1px solid #49483e; padding-bottom: 0.3em; }
h3 { color: #66d9ef; margin-top: 1.2em; }
code {
    font-family: 'SF Mono', 'Fira Code', Consolas, 'Liberation Mono', monospace;
    font-size: 14px;
    background: #1e1e1e;
    padding: 2px 6px;
    border-radius: 3px;
}
pre {
    background: #1e1e1e;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 1em 0;
}
pre code { padding: 0; background: none; }
.command { color: #e6db74; font-weight: bold; }
.param { color: #fd971f; }
.example { color: #75715e; font-style: italic; }
ul { margin: 0.5em 0; padding-left: 1.5em; }
li { margin: 0.3em 0; }
a { color: #66d9ef; }
.note {
    background: #3e3d32;
    border-left: 3px solid #a6e22e;
    padding: 10px 15px;
    margin: 1em 0;
    border-radius: 0 6px 6px 0;
}
@media (max-width: 600px) {
    body { padding: 15px; }
    pre { padding: 12px; font-size: 13px; }
}
</style>
</head>
<body>
<h1>LLM Bot Commands</h1>
<p>AI-powered IRC bot commands using LiteLLM.</p>
"""

_HTML_FOOT = """
<h2>Features</h2>
<ul>
<li><strong>Volatile Memory</strong> &ndash; Recent exchanges for natural follow-up questions (cleared by <code>%forget</code>, expires after timeout)</li>
<li><strong>Non-volatile Memory</strong> &ndash; Facts the bot remembers about you across conversations (managed by <code>%memories</code>)</li>
<li><strong>Vision Support</strong> &ndash; Include image URLs in <code>%ask</code> for image analysis</li>
<li><strong>Syntax Highlighting</strong> &ndash; Generated code is displayed with full highlighting</li>
<li><strong>Spontaneous Participation</strong> &ndash; The bot may occasionally join channel conversations (when enabled)</li>
<li><strong>Multi-Provider</strong> &ndash; Supports various AI providers via LiteLLM</li>
</ul>

<h2>Configuration</h2>
<div class="note">
Configuration is managed by the bot operator via Limnoria's config system.
Commands require the appropriate capability (e.g., <code>llm.ask</code>).
</div>

<p>Key settings include:</p>
<ul>
<li><strong>Model selection</strong> &ndash; Different models for ask/code/draw commands</li>
<li><strong>System prompts</strong> &ndash; Customize bot personality per command</li>
<li><strong>Context settings</strong> &ndash; Configure volatile memory limits</li>
</ul>

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
            sections.append(
                f'<h3><code class="command">%{cmd.name}</code> '
                f'<span class="param">{escaped_args}</span></h3>'
            )
            sections.append(f"<p>{html.escape(cmd.description)}</p>")
            example_lines = "\n".join(
                f'<span class="example">{html.escape(ex)}</span>' for ex in cmd.examples
            )
            sections.append(f"<pre><code>{example_lines}</code></pre>")
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

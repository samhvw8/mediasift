---
type: Decision
title: Distribution - how people install framesift and its agent skill
description: framesift ships as a Python package installed from GitHub (uv or mise pypi:), and the repository doubles as a Claude Code plugin marketplace that carries the skill and a guarded prompt hook.
tags: [install, distribution, plugin, skill]
generated:
  by: claude-code/opus-5.5
  at: "2026-10-16T10:00:00Z"
resource: .claude-plugin/marketplace.json
sources:
  - id: mise-pypi
    resource: https://mise.jdx.dev/dev-tools/backends/pypi.html
    title: mise pypi backend (pipx is an alias); GitHub shorthand installs the default branch when there are no releases
  - id: cc-marketplace
    resource: https://code.claude.com/docs/en/plugins/marketplace-reference
    title: Claude Code marketplace reference; "./" as a plugin source means the marketplace root
  - id: skills-cli
    resource: https://github.com/vercel-labs/skills
    title: npx skills add OWNER/REPO, which also reads .claude-plugin manifests
---

# CLI

framesift is a Python package with no prebuilt binaries, so it installs from the GitHub repository:

| tool | command | tested 2026-10-16 |
|---|---|---|
| uv | `uv tool install git+https://github.com/samhvw8/framesift` | installed in a clean tool dir; `see` sensed the fixture in 11.7 s and downloaded the 22 MB tagger and the 40 MB speaker model |
| mise | `mise use -g pypi:samhvw8/framesift`[^mise-pypi] | installed in 18 s; `pipx:` gives the same tool. `mise up` keeps the old commit ("no latest version found": no GitHub releases); `mise install -f` moves to the newest commit. `uv tool upgrade` follows new commits |
| mise | `mise use -g ffmpeg yt-dlp` | ffmpeg 9.0.2 (conda backend), yt-dlp (github backend) |

mise's `github:` backend needs release binaries, so it does not apply to framesift. There is no PyPI
release; installing from the repository keeps one source of truth.

# Agent skill

The repository root is both a Claude Code plugin and its marketplace[^cc-marketplace]
(`.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` with `"source": "./"`). The plugin
carries `skills/framesift/` and `hooks/hooks.json`. The hook runs
`command -v framesift && framesift hook || true`, so it is silent when the CLI is missing. It takes
about 80 ms on a prompt without a video. `npx skills add samhvw8/framesift`[^skills-cli] installs the
same skill for other agents; it needs strict YAML, so the skill's `description` is quoted (a bare `: ` inside it failed to parse). `plugin.json` has no `version`, so installed plugins follow the latest commit instead of waiting for a version bump; `claude plugin validate` warns about that, by choice. The skill's first section tells the agent how to suggest an install
when `framesift` is missing; it does not install anything itself.

[^mise-pypi]: mise pypi backend docs.
[^cc-marketplace]: Claude Code marketplace reference.
[^skills-cli]: vercel-labs/skills README.

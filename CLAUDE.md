# Project context

Before doing any work in this repo, read `.kiro/context.md` — it holds the current state of the project (architecture, in-progress work, decisions) and is the primary source of truth for Claude sessions here, alongside README.md for user-facing info.

It is Claude's responsibility to keep `.kiro/context.md` accurate and up to date as work happens in this repo: update it when architecture changes, features land, or decisions are made, so a new chat can pick up full context from this file alone. **Keep it short** — see its own §0 for the length rule; don't let entries grow back into essays.

**If the session is on a new machine** (data root missing, MATLAB not shared), read `.kiro/context.md` §15.8 first: it lists what lives outside git, the pipeline order, and how to attach to the user's MATLAB.

**If the chat is dedicated to thesis writing (editing `thesis/roberto_rocco_master_thesis/`)**, also read `thesis/master_thesis_rules.md` first — PRISMA Lab's official style/structure checklist, which governs that file. All thesis content (source, bibliography, figures, cited-paper notes) lives under `thesis/`. Ignore this for ordinary code work in this repo.

## Orchestration

When the active session model is Opus, dispatch subtasks (Agent tool calls) on Sonnet rather than inheriting Opus — Opus stays the orchestrator, Sonnet does the token-heavy subagent work.

## Commit & Push Discipline

- Commit messages: **one line, imperative mood, <72 chars**. No body unless the user asks for one.
- Do not narrate the commit/push process step by step — run it, then report the result in ≤2 sentences.
- Never re-read a file immediately after editing/writing it "to verify" — trust the tool; Edit/Write already error loudly on failure.
- Stage only the files actually changed for the task at hand — never `git add -A`/`git add .`.
- After push, give the sync command block (§14 of context.md) and stop — no summary of the diff unless asked.

## Code Comment Style

- Comments are **1 line only**: short, explicative, minimal.
- No history: no dates, no "previously X now Y", no references to past bugs, tuning steps, commits, or sessions. That belongs in the commit message, not the file.
- Each method gets exactly 1 explanatory line before/as its docstring.
- Comment only the non-obvious: hidden invariants, subtle workarounds, hard-to-read formulas, load-bearing observations. Well-named code needs no comment restating what it does.
- Write for a reader new to the project: comments must convey the fundamental meaning of the architecture, not internal narrative.
- The platform is **TRIAGo** — use that name in comments even where files/Docker say TIAGO.

## Sibling repo: haption_teleoperation

`~/exchange/ros2-ws/src/haption_teleoperation` drives the Haption haptic device and is the teleoperation half of the same pipeline this repo's QP controller/shared-autonomy stack serves; they share a live ROS2 topic interface, so a change on one side routinely requires a matching change on the other. If a task touches the teleop loop, grasp state machine, blending, or force feedback, use the `haption-interface` skill for the full topic map, or read `../haption_teleoperation/.kiro/context.md` directly.

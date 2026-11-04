---
name: reyu-workflow
description: The Reyu spec workflow for this repository — map, fix, check, accept. Use when the user asks about the repo's specs, wants changes made through specs, or asks what work was done.
---

# Reyu workflow

This repository is mapped with Reyu: every module has a `.ryu` spec under `.reyu/specs/`, drafted from the code, judged by a checker, accepted only by the human. The specs are the visible layer of this codebase — edit the spec, and the code follows.

## Commands (the `reyu` CLI)

- `reyu what` — what the journal says was checked/done recently. Use for "where was I" / "what changed".
- `reyu map .` (add `--view`) — re-ingest the repo and draft specs for unmapped modules. Never re-drafts ACCEPTED specs.
- `reyu check <spec.ryu>` — run the checker on a spec. FAIL output includes the invariant and a counterexample state; ERROR means the checker could not parse it. Always show the user the actual verdict text.
- `reyu fix <spec.ryu>` — repair loop: the local model proposes a fixed spec using the checker's error; a diff is shown; the user must approve. Never run with `--yes` unless the user explicitly said to apply without asking.
- `reyu accept <spec.ryu>` — flips DRAFT to ACCEPTED. **Never run this on your own.** Acceptance is the human's word. Ask, show the spec, wait for explicit approval.
- `reyu build <spec.ryu> --app <dir>` — regenerate a module's implementation from its spec; judged by tests + checker; installs only with `--force` after a pass, and only with the user's approval.
- `reyu tests <spec.ryu>` — emit the pytest suite the spec implies (walked states, boundary probes, counterexample witnesses; each case cites its invariant). Run it after accepting a spec, and again inside any fix loop: the emitted reds name the exact promise a change breaks. If the installed CLI errors on `reyu tests`, quote the error and rely on the repo suite — never hand-write the file the command would emit or caption your own tests with its name.


## The precision flow (task work)

When the user describes a task to do (not just a question to answer), work it through the task surface — four kinds of facts, each treated differently:

1. **Draft the task**: run `reyu task "<the user's words>"`. The draft's interface block carries the task surface — `# goal:`, one `# bucket:` line per dimension (PROMISED the words asked for, DISCOVERED what the repo already shows, ASSUMED what the draft had to assume, UNDEFINED what the words did not cover), `# baseline:`. Read the printed task id, goal, and baseline back to the user.
2. **Ask the boundary question, verbatim**: run `reyu task boundary <id>` and relay the printed question to the user EXACTLY as printed — one proposed dimension per line, answered with ONE WORD per line, in order (`out` / `promoted` / `assumed`). Journal the answers with `reyu task boundary <id> <word>...`. Never choose the answers yourself; a `promoted` answer may surface a fact conflict — show it to the user before working.
3. **Cite facts by id when working**: run `reyu retrieve "<what you are about to do>" --task <id>` and work from the rendered context. Cite the id of any item you use (`[P1]`, `f_...`). P-items are contract — they must hold when you are done. UNDEFINED dimensions never enter your context: they are out of scope until promoted.
4. **Surface ASSUMED at accept**: before `reyu accept`, list every ASSUMED item still in the task's context ("I assumed X — confirm?") and get the user's word on each. Assumptions surface at accept-time, never silently.

## Finding code: grep first, read after

Search before you read. Locating code by opening files one at a time burns turns and context; a targeted grep finds the exact file and line in one call.

- Locate with content search first: `grep -rn "pattern" path/` through the bash tool — or the `search`/`grep` tool when it is available (it does the same thing natively).
- Only then read — around the hit lines, not whole files.
- Search for specs the same way: `grep -rn "module name" .reyu/specs/` finds the spec that governs the code you are about to touch.

This is measured, not taste. In the Real-PR re-run (2026-10-29, 12 cells, both arms), agents given grep reached the fix file exactly as fast as before but paid far less for context — targeted greps replaced scroll-through reads — and on the one task that passed, the pass landed earlier: turn 8 → 6 without a spec, turn 8 → 4 with one. Grep to find; read to understand.

## The fix loop (promise → red → edit → green)

When changing code that a spec governs: run the module's emitted tests (`reyu tests <spec>` once, then pytest) alongside the repo suite. A red emitted test names the promise your change broke — read its `# implies:` citation, fix exactly that, rerun. Measured in the V8 experiment (2026-11-01): agents given the emitted fact-tests converted a task six prior runs could not — the reds shrank 3→2→0 across three submits and the hidden suite went green the same turn. When a fix is hard, show the user the loop state (which promises are red) instead of narrating.

## Rules


1. The spec files in `.reyu/specs/` are the source of truth for intent. Before changing code, find the module's spec with grep and read it.
2. When a spec fails the checker, quote the counterexample to the user in plain words before proposing anything.
3. ACCEPTED specs are never edited by you without the user's explicit instruction.
4. Map before drafting: if a module has no spec, run `reyu map .` (or draft for the specific module) rather than writing a spec by hand.
5. After any accepted change, suggest `reyu page` so the user can see the record.
6. Task work goes through the task surface: draft, boundary, cited context, assumptions surfaced. Never work an UNDEFINED dimension, and never let an assumption ride silently to accept.

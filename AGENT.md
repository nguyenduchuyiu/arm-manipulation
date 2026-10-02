# AGENTS.md

# Working Env
uv env named ".venv"

## Files and directoy management

* Delete temporary test, preparation, and execution code after use unless the user asks to keep it.

* Keep the repository root focused on the overall arm manipulation project and shared assets, controllers, and environments.
* Put task-specific code, scenes, data tools, and docs in a top-level folder for that task (for example, `memory_occlusion/`), organized into subfolders by role.

## Working style

Write code for a research prototype: correct, sufficient, minimal, and clean.

Prioritize:

1. Correct behavior.
2. Minimal implementation.
3. Clear structure.
4. Easy debugging.
5. Fast iteration.

Do not write production-style abstractions unless explicitly requested.

## Code rules

* Implement only what is necessary for the requested task.
* Keep the solution as small as possible.
* Prefer simple functions over complex classes.
* Prefer explicit code over clever abstractions.
* Do not add unnecessary configuration layers.
* Do not add unnecessary CLI flags.
* Do not add unnecessary environment-variable handling.
* Do not add unnecessary logging frameworks.
* Do not add unnecessary dependency injection.
* Do not add backward compatibility unless explicitly requested.
* Do not add broad fallback paths.
* Do not silently ignore errors.
* Do not use try/except unless there is a concrete reason.
* Do not catch generic exceptions just to continue running.
* Fail fast when required files, inputs, topics, models, or configs are missing.
* Keep error messages short and actionable.

## Research prototype standard

The code should be good enough to run experiments, debug results, and modify quickly.

It does not need to be:

* production-ready,
* highly configurable,
* enterprise-grade,
* fully generalized,
* compatible with every possible setup,
* optimized before correctness is proven.

It should be:

* readable,
* deterministic when possible,
* easy to run,
* easy to inspect,
* easy to delete or rewrite later.

## Scope control

Make the smallest change that solves the task.

Do not rewrite unrelated files.
Do not redesign the whole architecture unless asked.
Do not introduce new patterns just for cleanliness.
Do not add features that were not requested.
Do not preserve old behavior unless it is still needed.

When modifying existing code:

* keep the current style unless it is clearly broken,
* remove dead code when obvious,
* avoid large refactors,
* explain only the important changes.

## Fallback policy

Avoid fallback chains like:

* try A, if fail try B, if fail try C,
* silently replace missing input with defaults,
* continue with degraded behavior without telling the user,
* auto-detect many cases when one explicit case is enough.

For a research prototype, prefer:

* one clear expected path,
* one clear config,
* one clear failure mode.

If something is missing or invalid, raise an error and say exactly what is missing.

## Dependencies

Use existing dependencies when they make the implementation simpler, clearer, or more reliable.

Prefer well-maintained libraries for standard functionality instead of reimplementing solved problems.

Add a new dependency when it clearly reduces code complexity, avoids fragile custom code, or is necessary for the task.

Avoid adding large frameworks for small utilities.

Do not add dependencies just for style, abstraction, or premature optimization.

When adding a dependency, keep its usage direct and minimal.

## Comments

Add comments only when they explain non-obvious research logic, math, robotics assumptions, or experiment-specific choices.

Do not comment obvious Python syntax.

## Output after coding

After making changes, report:

1. Files changed.
2. What changed.
3. How to run or test it.
4. Any important limitation.

## Long running rule
Dont read all logs but save it to a log file and grep if needed.
For every long-running command in tmux, stream its main stdout and stderr to the tmux window while saving the same output to a log file (for example, `set -o pipefail; command 2>&1 | tee -a "$log_file"`). Do not redirect the only live output away from tmux. When several stages share a run, keep their current progress visible in a clearly named tmux window.
Use `tqdm` for every long-running code path so progress is visible in tmux and saved in the run log.

Keep the explanation concise.

## Server usage rules

Use `vishc-server-1` only for workloads that need its compute or Linux/CUDA environment. Keep code authoring, documentation, lightweight inspection, data analysis, and small smoke tests on the local machine whenever possible.

Update code on local machine and then git push to remote, and then git pull on server to save edit time.

Use one workspace per server: `/home/hoang.pm/duchuy/arm-manipulation` on `vishc-server-1`, and `/mnt/disk1/backup_user/25thanh.tk/arm-manipulation` on `vishc-server-2`. Do not create additional experiment worktrees. Name project branches, runs, and artifacts after `memory_occlusion`. Keep names of reference models only in source attribution and checkpoint provenance.

### Resources

* Default to one GPU. The user authorized up to three GPUs for the additional `vishc-server-2` run. Set `CUDA_VISIBLE_DEVICES` explicitly and use only GPUs with enough free memory.
* Do not use more than 8 CPU cores or 64 GB RAM.
* Run long jobs inside the `huy` tmux session so they survive SSH disconnects.
* Before starting a long run, check the selected GPU, available disk space, output paths, and whether another copy of the job is already running.
* Save stdout and stderr to a log file and show the main log live in tmux. Inspect progress with targeted `tail`, `grep`, or process/GPU checks instead of repeatedly reading the full log.

### Storage

* Keep the server repository at `/home/hoang.pm/duchuy/arm-manipulation` limited to source code and small runtime metadata.
* Store datasets, checkpoints, logs, videos, export shards, and other heavy artifacts under `/mnt/disk1/backup_user/hoang.pm/huy/arm-manipiulation`.
* On `vishc-server-2`, read the existing dataset/checkpoints from that shared mount; write new artifacts under `/mnt/disk1/backup_user/25thanh.tk/memory_occlusion`. Its SSH account is `25thanh.tk`, so do not write to the other account's artifact directories.
* Use these standard subdirectories: `datasets/`, `checkpoints/`, `logs/`, and `scratch/`.
* Treat the 100 GB `/home` account limit as hard. Check both `du -xsh /home/hoang.pm` and `df -h /home /mnt/disk1` before a run that may create substantial output.
* Never place a large dataset, checkpoint, cache, raw video collection, or temporary export physically under `/home`.
* Use fresh, experiment-specific output directories and fail if they already exist. Do not silently overwrite a previous run.
* Delete temporary shards, abandoned exports, debug dumps, and obsolete intermediate checkpoints after the final artifact has been verified. Retain the final checkpoint and any checkpoint explicitly needed for comparison or recovery.
* Do not delete or modify another user's directories, shared caches, environments, or processes.

### Local and server responsibilities

* Develop and review code locally first. Run a syntax check or small smoke test locally before starting an expensive server job.
* Treat the local repository as the source of truth. Prefer committing/pushing locally and pulling on the server; avoid editing the same source file independently in both places.
* Send only the code and configuration required to run the experiment. Do not use the server as a documentation archive or general workspace.
* Run GPU training, large-scale simulation collection, full dataset export, and heavy closed-loop evaluation on the server.
* Keep research documentation only in local `docs/research/`. The canonical report is `docs/research/REPORT.md`.
* After each experiment, copy important small outputs back to the local experiment folder: configuration, audit JSON, `summary.json`, `results.jsonl`, selected plots/images/videos, and the useful final log excerpt.
* For a large final checkpoint, keep it on `/mnt`, record its exact path and checksum in the local report, and copy it to the local machine only when it is needed there.

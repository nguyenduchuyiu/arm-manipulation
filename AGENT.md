# AGENTS.md

# Working Env
use uv venv

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

## Experiment names and descriptions

* Keep experiment-specific code inside that experiment's own directory.
* Keep only genuinely reusable implementation in shared repository modules. An experiment may import shared code, but its own variants must stay within its experiment directory; do not build chains of per-experiment packages importing other experiment packages. Record the shared dependencies and their source hashes for reproducibility.
* This organization rule applies to future work. Do not relocate old experiments as part of a new run unless the user requests cleanup; preserve their saved paths and reproducibility.
* Use short experiment names such as `e01_delta`, `e02_joint`, `e03_freeze`, and `e04_vit`. Use the next unused numeric ID for a new experiment; put dates and detailed settings in metadata rather than in the directory name. This user preference supersedes the older requirement to prefix every run with `memory_occlusion`.
* Keep related tests under the experiment as `test_k10`, `test_k25`, etc.; use a short numeric suffix for repetitions and never overwrite earlier results. Keep run, log, and tmux names short and consistent.
* Every experiment output folder, including a standalone test folder, must contain a short `README.md`: experiment purpose/change, actual training configuration (data, source weights, frozen/trainable modules, epochs, batch, learning rate, action representation), and actual test configuration (checkpoint/step, scenes, H/K, flow steps, noise seed, frame budget). Record status/results and link exact configuration files. Mark tests as planned, incomplete, or not run when applicable.


## Experiment isolation and code reuse

- Use a dedicated Git branch/worktree for experiments requiring isolated code changes.
- Keep experiment-specific code in `experiments/e###_name/code/`; do not mix variants into shared modules.
- Start new experiments from `main` or a chosen previous commit.
- Reuse code selectively via `git restore`, `cherry-pick`, or copying specific files. Never import directly from another experiment.
- Do not modify completed experiments when developing new ones.
- Move code into shared modules only when it is genuinely reusable across experiments.
- Commit code before running and record the exact Git commit hash and reused code sources in the experiment README.
- Keep datasets, checkpoints, logs, and temporary outputs out of Git.

## Long running rule
Dont read all logs but save it to a log file and grep if needed.
Run each experiment in a separate named window of the user's attached tmux session (currently `ssh_tmux`), so the user can switch to its live logs. Do not put runs only in separate detached sessions. Leave the current window selected, set `remain-on-exit on` for the experiment window, and report its name.
For every long-running command in tmux, stream its main stdout and stderr to the tmux window while saving the same output to a log file (for example, `set -o pipefail; command 2>&1 | tee -a "$log_file"`). Do not redirect the only live output away from tmux. When several stages share a run, keep their current progress visible in a clearly named tmux window.
Use `tqdm` for every long-running code path so progress is visible in tmux and saved in the run log.

Keep the explanation concise.

### Local and server responsibilities
* Keep research documentation only in local `docs/research/`. The canonical report is `docs/research/REPORT.md`.
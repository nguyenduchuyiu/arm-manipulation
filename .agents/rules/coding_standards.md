# Project Rules & Coding Standards

## Working Style

Write code for a research prototype: correct, sufficient, minimal, and clean.

Prioritize:
1. Correct behavior.
2. Minimal implementation.
3. Clear structure.
4. Easy debugging.
5. Fast iteration.

Do not write production-style abstractions unless explicitly requested.

## Code Rules

* Implement only what is necessary for the requested task.
* Keep the solution as small as possible.
* Prefer simple functions over complex classes.
* Prefer explicit code over clever abstractions.
* Do not add unnecessary configuration layers.
* Do not add unnecessary CLI flags.
* Do not add unnecessary environment-variable handling.
* Do not add unnecessary logging frameworks.
* Do not add broad fallback paths.
* Do not silently ignore errors.
* Fail fast when required files, inputs, models, or configs are missing.
* Keep error messages short and actionable.

## Robotics & Simulation Standards

* Preserve exact physical units (meters, radians, seconds) in MuJoCo XMLs and environment code.
* Keep camera positions, FOVs, and orientations fixed between training and evaluation unless domain randomization is requested.
* Keep wrist camera views clean, un-occluded by site geometry, and positioned in front of physical bracket meshes.
* Keep clean verification scripts in `scripts/` and avoid cluttering the project with temporary debug outputs.

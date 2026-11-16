# Repo conventions (for Claude Code sessions)

This repo is a **toolbox**: many small, independent tools, one per top-level folder.
The owner is not a professional programmer. Keep instructions plain and copy-pasteable.

## Layout rules
- Each project/chat gets **one top-level folder named for its purpose in ALL CAPS with
  underscores**, e.g. `PROFILE_TRACER/`, `NOISEMSG_MODEM/`. No spaces.
- Everything for that tool lives inside its folder:
  `README.md`, `.gitignore`, `pyproject.toml`, the Python package (lowercase name,
  e.g. `profiler/`), `tests/`, and any data or scripts.
- **Never** put a project's files at the repo root, and **never** create a nested `.git`
  (no submodules). Don't edit other tools' folders unless asked.
- The root `README.md` has a table with one row per tool (folder, one-line purpose,
  command). Add or update your tool's row; don't rewrite the rest.

## Python packaging (uv)
- Use a `pyproject.toml` per folder (copy `PROFILE_TRACER/pyproject.toml` as a template):
  dependencies in `[project] dependencies`, pytest in `[dependency-groups] dev`,
  a command in `[project.scripts]`, and the hatchling build backend.
- Don't add `requirements.txt` files. Don't tell the owner to make or activate virtual environments.
- How the owner runs things: `uv tool install --editable ./FOLDER` then `<command> ...`,
  or `cd FOLDER && uv run <command> ...`.
- Run tests with `cd FOLDER && uv run pytest -q` before committing.
- Commit `uv.lock` if uv creates one.
- Choose a command name that doesn't clash with the others in the root README table.

## Git
- Work on a `claude/...` branch from the latest `main`, then open a pull request to `main`.
- Don't commit large personal data (photos, scans, zips) unless the owner asks; use the
  folder's `.gitignore`.

## Moving an old chat's work into this layout
If a branch was made before this layout existed and has files at the repo root:
1. `git fetch origin main` and rebase or merge onto it (resolve root `README.md` /
   `requirements.txt` / `tests/` conflicts in favour of the folder layout).
2. `git mv` that project's files into its new ALL-CAPS folder (keeping history).
3. Add the folder's `pyproject.toml`, `README.md`, and `.gitignore`. Fold the old
   `requirements.txt` into `pyproject.toml` and delete it.
4. Add a row to the root README table, run the tests with uv, push, and open a pull request.

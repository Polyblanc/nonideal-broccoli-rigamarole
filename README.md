# nonideal-broccoli-rigamarole
We all make mistakes sometimes, this one included.

A toolbox of small programs made with Claude Code. **Each tool lives in its own
ALL-CAPS folder** with its own README, `.gitignore`, `pyproject.toml` and tests.
Open a folder's README to see what the tool does and how to use it.

| Folder | What it does | Command |
|---|---|---|
| [PROFILE_TRACER](PROFILE_TRACER/) | Photo of a part / PCB / tool / spring → outline or centerline as SVG or DXF, with click-to-pick, lasso, zoom and two-point scale | `profiler` |
| [VIDEO_SPLAT](VIDEO_SPLAT/) | Single-camera video → per-frame 3D Gaussian splats (SHARP) rendered with gsplat. Needs an NVIDIA GPU for rendering | `video-splat` |

## One-time setup on your computer

1. **Install git**: <https://git-scm.com/downloads> (macOS: run `git --version` and it offers to install).
2. **Install uv**, which installs Python and every tool's packages for you:
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

   Then close and reopen the terminal.
3. **Download this repo** into a folder you'll remember:
   ```bash
   git clone https://github.com/Polyblanc/nonideal-broccoli-rigamarole.git
   cd nonideal-broccoli-rigamarole
   ```
4. **Install a tool as a command** you can type from anywhere:
   ```bash
   uv tool install --editable ./PROFILE_TRACER
   ```
   (Repeat for each folder you want. `--editable` means updates from `git pull` show up
   without reinstalling, unless a tool's dependencies changed; then rerun with `--force`.)

You do **not** need to create or activate virtual environments: uv does that behind the
scenes. It downloads each package once into a shared cache, so ten tools that all use
OpenCV don't download OpenCV ten times.

## Getting new tools / updates later

```bash
cd nonideal-broccoli-rigamarole
git pull
uv tool install --editable ./NEW_FOLDER     # only for a new tool
```

## Trying a tool without installing it

```bash
cd PROFILE_TRACER
uv run profiler serve
```

## How changes get here
Every Claude Code chat works on its own branch (`claude/...`) and opens a pull request.
Merging the pull request on GitHub puts the tool into `main`. Branches are temporary;
once a branch is merged, it can be deleted. The conventions that Claude Code sessions
follow are in [CLAUDE.md](CLAUDE.md).

Other files at the top level (`index.html`, `LICENSE`, `.github/`) predate the toolbox.

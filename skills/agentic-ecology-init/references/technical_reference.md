# Initialization Technical Reference

Detailed instructions, code templates, and configuration schemas for
initializing a local `uv`-managed agentic ecology workspace.

## 1. Project Initialization & Directory Structure

To initialize a new project workspace, execute `uv init --python 3.12 --bare`
in the root directory. This bootstraps the project environment, pins Python 3.12
in `.python-version`, generates `.gitignore` (ignoring `.venv`), and initializes
version control tracking (which is required by dynamic versioning tools and
package managers) without generating unwanted boilerplate code.

```bash
# Initialize uv project with Python 3.12 (if not already initialized)
uv init --python 3.12 --bare

# Scaffold required agentic ecology subdirectories
mkdir -p .agents agent_workspace data databases
```

A fully initialized agentic ecology project directory follows this structure:

```text
my_ecology_project/
├── .agents/
│   ├── AGENTS.md             # Workspace safety rules and import gotchas
│   └── skills.json           # Antigravity skill linking config
├── agent_workspace/          # Sandboxed scratchpad for agent code and UI apps
├── data/                     # Local raw dataset files (audio recordings, images)
├── databases/                # Hoplite SQLite/USearch vector databases
├── .gitignore                # Git/VCS ignore list (ignores .venv)
├── .python-version           # Target Python version pinning
├── pyproject.toml            # Python dependency definitions
└── uv.lock                   # Pinned dependency lockfile
```

--------------------------------------------------------------------------------

## 2. Skills Configuration (`.agents/skills.json`)

Antigravity uses `.agents/skills.json` to discover skills located in external or
shared repositories.

### Configuration Schema

Create `.agents/skills.json` inside the target workspace with the following
content:

```json
{
  "entries": [
    {
      "path": "/path/to/agentic_ecology/skills"
    }
  ]
}
```

### Path Resolution Options

*   **Absolute Path**: `"/Users/<username>/.../agentic_ecology/skills"`
*   **Home-Relative Path**: `"~/agentic_ecology/skills"` (resolves relative
    to the user's home directory).
*   **Workspace-Relative Path**: `"../agentic_ecology/skills"` (resolves
    relative to the workspace root).

> [!TIP]
> If the user wants the skills available across all workspaces on their machine,
> they can instead register the entries in their global Antigravity configuration
> at `~/.gemini/config/skills.json`.

--------------------------------------------------------------------------------

## 3. Dependency Configuration (`pyproject.toml` and `uv.lock`)

### Copying Configurations

Copy the reference files [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock)
from the `agentic_ecology` repository into the target workspace root. This
replaces the skeleton `pyproject.toml` created by `uv init` with the complete
dependency configuration and pinned lockfile:

```bash
cp /path/to/agentic_ecology/skills/agentic-ecology-init/references/pyproject.toml .
cp /path/to/agentic_ecology/skills/agentic-ecology-init/references/uv.lock .
```

### Adjusting `pyproject.toml`

When initializing a new project, modify the `name` field in `pyproject.toml` to
reflect the user's project name while keeping the required dependencies and
constraints intact:

```toml
[project]
name = "my_ecology_project"
version = "0.1.0"
description = "Agentic tools for ecological modelling"
requires-python = ">=3.12"

dependencies = [
    "perch-hoplite[tf,jax,onnx]",
    "speciesnet",
]

[tool.uv]
package = false
constraint-dependencies = [
    "numba >= 0.59.0",
    "torch < 2.13",
]

[tool.uv.sources]
perch-hoplite = { git = "https://github.com/google-research/perch-hoplite.git" }
```

--------------------------------------------------------------------------------

## 4. Workspace Guidelines (`AGENTS.md`)

Copy the reference [AGENTS.md](AGENTS.md) into the target project root or
`.agents/AGENTS.md` to preserve essential workspace rules (such as macOS
TensorFlow/PyArrow deadlock prevention, Linux PyTorch/TensorFlow import order,
and SQL log suppression filters):

```bash
cp /path/to/agentic_ecology/skills/agentic-ecology-init/references/AGENTS.md .agents/AGENTS.md
```

--------------------------------------------------------------------------------

## 5. Synchronizing the Environment

Run `uv sync` to create the virtual environment (`.venv`) and install the
pinned dependencies:

```bash
uv sync
```

--------------------------------------------------------------------------------

## 6. Verifying the Setup

Copy the reference verification script [verify_env.py](verify_env.py) into
`agent_workspace/` and execute it via `uv run python` to confirm that all core
libraries are available and import safely without deadlocks:

```bash
cp /path/to/agentic_ecology/skills/agentic-ecology-init/references/verify_env.py agent_workspace/
uv run python agent_workspace/verify_env.py
```

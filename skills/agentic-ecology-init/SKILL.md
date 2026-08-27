---
name: agentic-ecology-init
description: >-
  Initializes a local uv-managed project directory for agentic ecology workloads.
  Sets up Python dependencies using reference pyproject.toml and uv.lock, configures
  workspace rules, and creates a local skills.json linking to agentic ecology skills.
---

# Agentic Ecology Project Initialization Skill

Use this skill when you need to initialize a fresh, local project directory for
ecological modeling workflows (such as bioacoustics or camera trap analysis)
without cloning the entire `agentic_ecology` repository into the user's project
workspace.

## Workflow Overview

Follow these sequential steps to set up the local workspace:

1.  **Identify Target Working Directory & Repository Location**:
    *   Confirm the root directory of the user's project workspace.
    *   Locate the reference `agentic_ecology` repository on the local machine
        (or retrieve its location from the user if not obvious).
2.  **Initialize Project & Scaffolding**:
    *   Ensure the target project directory exists.
    *   Run `uv init --python 3.12 --bare` in the project directory if it has
        not already been initialized. This pins `.python-version` to 3.12,
        generates `.gitignore`, and initializes version control tracking.
    *   Create standard working subdirectories:
        *   `agent_workspace/`: Sandboxed folder for agent scripts, server
            runners, and generated files.
        *   `databases/`: Destination folder for Hoplite vector databases.
        *   `data/`: Destination folder for raw datasets (e.g., audio, images).
        *   `.agents/`: Configuration folder for agent skills and local rules.
3.  **Copy Reference Dependency Configurations**:
    *   Copy the reference `pyproject.toml` and `uv.lock` from
        `agentic_ecology/skills/agentic-ecology-init/references/` into the target project root.
    *   Adjust the package name in `pyproject.toml` to match the user's project
        name if desired, keeping all core dependencies (`perch-hoplite`,
        `speciesnet`), constraints, and build configurations intact.
4.  **Configure Skill Discovery (`skills.json`)**:
    *   Create `.agents/skills.json` in the user's project directory.
    *   Add an entry pointing to the `agentic_ecology/skills` directory
        so Antigravity automatically discovers downstream skills (`agentic-ecology-bioacoustics`,
        `agentic-ecology-camera-traps`, `agentic-ecology-colab-cli`, `agentic-ecology-storage`,
        `agentic-ecology-ui`).
5.  **Install Workspace Rules (`AGENTS.md`)**:
    *   Copy the reference `AGENTS.md` from
        `agentic_ecology/skills/agentic-ecology-init/references/` into the project root or
        `.agents/AGENTS.md` to establish standard Agentic Ecology safety
        guidelines (macOS dynamic library deadlock rules, Linux
        PyTorch/TensorFlow import order rules, and SQL log suppression filters).
6.  **Synchronize Environment with `uv`**:
    *   Execute `uv sync` from the target project root to create the local
        virtual environment (`.venv`) and install all pinned dependencies.
7.  **Verify Environment Setup**:
    *   Run a verification command via `uv run python` to confirm that key
        libraries (`perch_hoplite`, `speciesnet`, `soundfile`, `tensorflow`)
        import cleanly.

## Technical Reference

For detailed command options, directory layout specifications, `skills.json`
schema examples, and verification code snippets, see:

*   [Initialization Technical Reference](references/technical_reference.md)

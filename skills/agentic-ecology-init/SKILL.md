---
name: agentic-ecology-init
description: >-
  Initializes a local uv-managed project directory for agentic ecology workloads.
  Sets up Python dependencies using reference pyproject.toml and uv.lock, configures
  workspace rules, and ensures Agentic Ecology skills are discoverable.
---

# Agentic Ecology Project Initialization Skill

Use this skill when you need to initialize a fresh, local project directory for
ecological modeling workflows (such as bioacoustics or camera trap analysis)
without cloning the entire `agentic_ecology` repository into the user's project
workspace.

## Workflow Overview

Follow these sequential steps to set up the local workspace:

1.  **Identify Target Working Directory**:
    *   Confirm the root directory of the user's project workspace.
2.  **Initialize Project & Scaffolding**:
    *   Ensure the target project directory exists.
    *   Run `uv init --python 3.12 --no-readme && rm main.py` in the project
        directory if it has not already been initialized. This pins
        `.python-version` to 3.12, generates `.gitignore`, initializes version
        control tracking, and removes the placeholder entrypoint.
    *   Install the following skills locally with `npx skills add`:
        *    `google-deepmind/agentic_ecology` (all skills).
        *    `googlecolab/google-colab-cli` (`colab-operator` skill).
        *    `googleworkspace/cli` (`gws-shared` and `gws-drive-upload` skills).
    *   Create standard working subdirectories:
        *   `agent_workspace/`: Sandboxed folder for agent scripts, server.
        *   `databases/`: Destination folder for Hoplite vector databases.
        *   `data/`: Destination folder for raw datasets (e.g., audio, images).
3.  **Copy Reference Dependency Configurations**:
    *   Overwrite the generated `pyproject.toml` and copy `uv.lock` from
        this skill's `references/` directory into the target project root.
    *   Adjust the package name in `pyproject.toml` to match the user's project
        name if desired, keeping all core dependencies (`perch-hoplite`,
        `speciesnet`), constraints, and build configurations intact.
5.  **Install Workspace Rules (`AGENTS.md`)**:
    *   Copy the reference `AGENTS.md` from this skill's `references/` directory
        into the appropriate location in the project workspace to establish
        standard Agentic Ecology guidelines (compute assessment and offloading
        protocols, macOS dynamic library deadlock rules, Linux
        PyTorch/TensorFlow import order rules, and SQL log suppression filters).
6.  **Synchronize Environment with `uv`**:
    *   Execute `uv sync` from the target project root to create the local
        virtual environment (`.venv`) and install all pinned dependencies.
7.  **Verify Environment Setup**:
    *   Run a verification command via `uv run python` to confirm that key
        libraries (`perch_hoplite`, `speciesnet`, `soundfile`, `tensorflow`)
        import cleanly.

## Technical Reference

For detailed command options, directory layout specifications, and verification
code snippets, see:

*   [Initialization Technical Reference](references/technical_reference.md)

---
name: agentic-ecology-storage
description: >-
  Provides guidelines and reference implementations for uploading local data
  (such as audio recordings or datasets) to cloud storage systems, focusing on
  Google Drive via rclone as the primary target.
---

# Storage Skill

This skill provides patterns for transferring local datasets (such as raw audio
recordings or Hoplite databases) to Google Drive using `rclone`, so they can be
processed remotely (e.g., in a Google Colab notebook).

## Workflow Overview

Follow these sequential steps when a user needs to migrate local datasets to
Google Drive:

1.  **Verify rclone Installation:** Check if `rclone` is installed on the user's
    system by running `which rclone`. If not, provide the user with installation
    instructions.
2.  **Verify or Bootstrap Configuration:**
    *   Check if the user has a configured Google Drive remote by running
        `rclone listremotes`.
    *   If no remote is configured, do not ask the user to manually walk through
        the interactive configuration. Instead, **propose running `rclone config
        create gdrive drive` directly**. This command creates a remote named
        `gdrive` and opens the OAuth approval page in their browser
        automatically.
3.  **Execute Transfer:** Run the upload command using `rclone copy` or `rclone
    sync` to transfer the local dataset or folders to the remote Google Drive
    destination.
4.  **Retrieve Folder ID (Optional):** Once uploaded, get the remote folder
    details or direct sharing link if needed for downstream remote processing.

## Technical Reference

For installation steps, configuration guides, and command examples, see:

*   [Storage Technical Reference](references/technical_reference.md)

> [!IMPORTANT] To comply with repository environment rules, any files you plan
> to use, test, or modify MUST first be copied into the `agent_workspace/`
> directory. Do not run or write them directly in `skills/...` to ensure
> all executions remain inside the workspace.

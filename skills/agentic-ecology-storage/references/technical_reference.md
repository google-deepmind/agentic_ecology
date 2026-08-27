# Storage Integration Technical Reference (rclone)

Instructions and reference commands for the agent to install, configure, and
execute Google Drive uploads using `rclone`.

--------------------------------------------------------------------------------

## 1. Installation

If `rclone` is not installed on the system (verify using `which rclone`),
propose the appropriate installation command to the user based on their
operating system:

### Ubuntu / Debian

```bash
sudo apt update && sudo apt install -y rclone
```

### macOS (Homebrew)

```bash
brew install rclone
```

### Windows (Winget)

```powershell
winget install rclone.rclone
```

--------------------------------------------------------------------------------

## 2. Configuration Bootstrapping

To create a Google Drive remote, do not instruct the user to run the multi-step
interactive wizard (`rclone config`). Instead, **run the following command on
the workstation to initialize the remote and launch the OAuth authorization
flow**:

```bash
rclone config create gdrive drive
```

### Flow Execution:

*   **Interactive Browser Flow:** This command will attempt to open a browser
    window on the user's local machine for Google Sign-In and OAuth consent.
*   **Headless/Remote SSH Flow:** If the browser cannot be opened automatically,
    the command outputs a URL. Copy this URL and present it to the user.
    Instruct the user to open it in a browser, authenticate, and paste the
    resulting authorization code back into the terminal.

--------------------------------------------------------------------------------

## 3. Uploading Datasets

Execute or propose the following commands to transfer files or directories to
Google Drive. Use the remote name configured (defaults to `gdrive:`).

### Copying a Directory

Use this command to upload a local directory to Google Drive. Already uploaded
files on the remote will be skipped:

```bash
rclone copy /path/to/local/dataset gdrive:my_dataset_folder
```

### Synchronizing a Directory

Use this command to ensure the remote directory matches the local directory
exactly. Note that this will **delete** files on the remote that do not exist
locally:

```bash
rclone sync /path/to/local/dataset gdrive:my_dataset_folder --interactive
```

### Recommended Performance Flags

When running transfers, append these flags to optimize performance:

*   `--transfers=N`: Controls parallel file uploads (default is 4). Adjust based
    on workstation network bandwidth.
*   `--checkers=N`: Controls parallel file checks (default is 8).
*   `-P` or `--progress`: Displays real-time transfer progress and throughput
    statistics.

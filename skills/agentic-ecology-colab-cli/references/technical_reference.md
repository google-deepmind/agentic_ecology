# Colab CLI Technical Reference

Reference guide for installation commands, authentication setups, and remote
session management commands.

--------------------------------------------------------------------------------

## 1. Installation

Install the package globally or run it on-demand:

### Installation via uv

```bash
uv tool install google-colab-cli
```

### On-Demand Invocation (If path conflicts exist)

```bash
uv tool run --from google-colab-cli colab <command>
```

--------------------------------------------------------------------------------

## 2. Authentication Scopes

The CLI uses Google Application Default Credentials (ADC) by default. If you
encounter authorization errors (`401` or `403`), re-login specifying the
required scopes:

```bash
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

Alternatively, force OAuth2 client authentication:

```bash
colab --auth=oauth2 <command>
```

--------------------------------------------------------------------------------

## 3. Remote Session Commands

Use these exact command structures. Replace `my_session` with your target
session name.

### Troubleshooting & Execution Quirks

#### 1. Argument Parser SystemExit 2 (Jupyter Kernel Conflict)

*   **Issue:** When executing a script via `colab exec -f script.py`, the code
    runs inside a Jupyter/IPython kernel. If the script uses `argparse`, it
    parses `sys.argv`, which contains internal Jupyter kernel arguments (like
    `-f /root/.../kernel-xxx.json`), causing the script to exit with error
    code 2.
*   **Workaround:** Detect if the script is executing inside a Jupyter/IPython
    kernel and parse empty arguments if so:

    ```python
    import sys
    is_jupyter = (
        any(arg.endswith(".json") for arg in sys.argv)
        or (len(sys.argv) > 0 and "ipykernel" in sys.argv[0])
    )
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    ```

#### 2. OSError: Read-only file system on Kaggle/Colab Mounts

*   **Issue:** When running inside an environment with pre-mounted Kaggle models
    (like Colab VMs), `kagglehub.model_download` resolves to a read-only
    directory `/kaggle/input/`. Since model code (such as `SpeciesNetDetector`)
    attempts to download and write additional weights into the model directory,
    it fails with a read-only filesystem error.
*   **Workaround:** Copy the model directory from `/kaggle/input/...` to a local
    writeable directory (e.g. `/content/speciesnet_model`) before loading the
    models.
*   **Example:**

    ```python
    import os
    import shutil

    local_dir = "/content/speciesnet_model"
    mounted_dir = "/kaggle/input/speciesnet/pytorch/v4.0.3a/1"
    if os.path.exists(mounted_dir) and not os.path.exists(local_dir):
        shutil.copytree(mounted_dir, local_dir)

    detector = SpeciesNetDetector(local_dir)
    classifier = SpeciesNetClassifier(local_dir)
    ```

#### 3. Slow or Timeout Issues with git+https Pip Installs

*   **Issue:** Installing heavy package lists combined with Git repository
    packages (e.g., `colab install -s session package_name "git+https://..."`)
    in a single step can trigger timeouts.
*   **Workaround:** Install standard PyPI packages first (which are fast), and
    install Git repository packages in a separate command.

### Session Creation

*   **CPU Session:**

    ```bash
    colab new -s my_session
    ```

*   **GPU Session (options: T4, L4, A100, H100):**

    ```bash
    colab new --gpu T4 -s my_session
    ```

### Mounting Google Drive

Mount Google Drive inside the session virtual machine at `/content/drive`:

```bash
colab drivemount -s my_session
```

### Package Installation

Install standard PyPI packages inside the VM:

```bash
colab install -s my_session package_name
colab install -s my_session -r requirements.txt
```

### Script Execution

Execute a local Python script on the remote session:

```bash
colab exec -s my_session -f local_script.py
```

### Session Termination

Terminate and clean up the session immediately:

```bash
colab stop -s my_session
```


# Agentic ecology

Agentic ecology provides AI-driven tools to assist researchers and practitioners
processing, analyzing, and annotating their data. It currently focuses on
passive acoustic monitoring (PAM) datasets (bioacoustics), with more
capabilities to be introduced over time.

Using Google's **Antigravity IDE**, you can generate custom python scripts,
build interactive web interfaces to listen to and search your recordings, and
train classifier models—all by describing what you want to do in plain English.

## Why?

Setting up an entire agentic development framework for you to build a data
processing app sounds like a whole lot of trouble when we could simply develop
the app itself and distribute it to you, doesn't it?

The reason for that is *empowerment*: we want *you* to have the ability to build
software tools that address *your* needs. We recognize that doing so requires
its own skillset, but we firmly believe that agentic coding, paired with the
right set of agent skills, can eliminate that barrier.

Think of this project not as a tool, but as a tool-building tool.

--------------------------------------------------------------------------------

## 🚀 Getting started

Follow these step-by-step instructions to set up the environment and run your AI
assistant.

### 1. Install prerequisites

You need three tools installed on your computer: **Git**, **uv**, and the
**Antigravity IDE**.

#### A. Install Git (if you don't have it)

Git is used to download the repository.

*   **macOS / Linux**: Git is usually pre-installed. You can check by running
    `git --version` in your terminal.
*   **Windows**: Download and install it from
    [git-scm.com](https://git-scm.com/).

#### B. Install `uv`

`uv` is a tool that installs the correct version of Python and manages all
package dependencies automatically. Please refer to the
[official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
to install it on your operating system (macOS, Linux, or Windows).

#### C. Install Antigravity IDE

The Antigravity IDE is Google's agentic development environment. Please follow
the instructions on the
[official Antigravity website](https://antigravity.google/download) to download
and install the program on your machine.

--------------------------------------------------------------------------------

### 2. Set up the agentic ecology workspace in Antigravity IDE

1.  Launch the **Antigravity IDE** application.
2.  On the welcome screen, click **Clone Repository**.
3.  Paste the repository URL:
    `https://github.com/google-deepmind/agentic_ecology` and select a folder on
    your computer.
4.  Once cloned, you will see a prompt asking if you would like to open the
    repository; click **Open** to open the project.

You are now ready to set up the workspace.

> **Warning:** Unless you have extensive experience with agentic software
> engineering, we recommend that you set up your workspace so that the agent
> asks your permission before performing any operation on your machine on your
> behalf or to interact with files outside of the workspace's root directory.
> Vet every command the agent intends to run on your behalf and carefully
> inspect the code it requests to execute, as agent mistakes can happen. Avoid
> letting the agent perform operations on your data without backing it up and
> taking other relevant precautions.

#### Ask the agent

> Set up the environment for this project.

#### What the agent does

The agent runs the necessary commands to set up Python and install package
dependencies.

--------------------------------------------------------------------------------

## 🛠️ Demo workflow

To help you get a feel of what's possible with agentic ecology, let's simulate a
task in which we are trying to bootstrap the annotation of a passive acoustic
monitoring dataset for a targeted bird species. We will work with the
[Powdermill](https://zenodo.org/records/4656848) dataset published by Chronister
et al. (2022) and search for [Hooded Warbler](https://ebird.org/species/hoowar)
songs.

#### Ask the agent

> I would like to analyze the contents of the Powdermill PAM dataset. The data
> is hosted here: https://zenodo.org/records/4656848/files/mp3_Files.zip.
> Download and extract the files into data/powdermill, then help me get started.

> **Note:** We are downloading MP3 files rather than WAV files to save on
> download time. If you would rather work with WAV files, feel free to adapt the
> URL in the prompt above.

#### What the agent does

The agent:

1.  Downloads and unzips the Powdermill audio data in `data/powdermill`.
2.  Identifies all audio files in that directory.
3.  Runs the Perch 2.0 model over them to build an audio index.
4.  Creates a vector database in the `databases/` folder of the project to store
    everything. *(This step may take some time, especially if executing on CPU.
    Expect around 15 minutes.)*
5.  Builds a web application for you to browse, search, and annotate the
    database.
6.  Presents you with instructions on how to access and use the web application.

#### Try out the web app

Use the agent instructions to access the web app, and try searching for Hooded
Warbler songs in the database. You can use a
[recording from Xeno-Canto](https://xeno-canto.org/565524) to get started by
entering `xc565524` into the query URI bar.

#### Ask the agent to make changes to the web app

> I would like to be able to filter the audio windows by the recording from
> which they are taken. Add a UI element for that.

The agent will autonomously figure how to modify the existing code to accomplish
that, restart the backend server, and prompt you to reload the webpage.

--------------------------------------------------------------------------------

## 💡 Key takeaway

This repository is very minimal: it contains a
[`pyproject.toml`](pyproject.toml) configuration file declaring the Python
dependencies the agent needs along with a handful of agent instruction files
([`GEMINI.md`](GEMINI.md), [`.agents/AGENTS.md`](.agents/AGENTS.md)) and skills
(stored in [`.agents/skills`](.agents/skills)), all of which are human-readable.

The agent instructions and skills are nothing more than a shortcut that reliably
sends the agent in the right direction: they were themselves constructed by
prompting the agent to achieve a particular outcome, interactively solving the
problem with it, and asking it to write instructions for its future self to
arrive at to the solution right away. When the agent made a mistake, it was
asked to reflect on it and to amend the instructions and skills so that it
doesn't make the same mistake again in the future.

Any ecological analysis capability that the agent currently has could just as
well be achieved without any skill or pre-written instructions by working with
it iteratively and interactively. In fact, this is exactly how we intend to
expand the agent capabilities in this project!

This means that building the right tool for your needs is within your reach:
don't hesitate to state your needs; to question the agent; to ask it to clarify,
self-correct its mistakes, and amend its instructions and skills; to nudge it in
the right direction if it starts veering down the wrong path.

--------------------------------------------------------------------------------

## 📁 Repository Structure

*   `databases/`: This is where the databases created by the agent are stored.
*   `agent_workspace/`: This is where all files generated by the agent are
    saved. Look here for your web app, python scripts, and HTML pages.
*   `.agents/skills/`: Custom capability instructions (skills) that tell the
    agent how to build databases and web apps safely and correctly.

*This is not an officially supported Google product.*

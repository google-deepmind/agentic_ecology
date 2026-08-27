
# Agentic ecology

Agentic ecology provides AI-driven tools to assist researchers and practitioners
processing, analyzing, and annotating their data. It currently focuses on
passive acoustic monitoring (PAM) datasets (bioacoustics), with more
capabilities to be introduced over time.

Using an agentic development platform such as [**Google
Antigravity**](https://antigravity.google/), you can generate custom python
scripts, build interactive web interfaces to listen to and search your
recordings, and train classifier models—all by describing what you want to do in
plain English.

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

To get started, open your agentic development platform and prompt your agent:

> Help me get started with agentic ecology by following the README agent
> instructions in https://github.com/google-deepmind/agentic_ecology.

### A word of caution

Unless you have extensive experience with agentic software engineering, we
recommend that you set up your agentic coding environment so that the agent asks
your permission before performing any operation on your machine on your behalf
or to interact with files outside of the workspace's root directory. Vet every
command the agent intends to run on your behalf and carefully inspect the code
it requests to execute, as agent mistakes can happen. Avoid letting the agent
perform operations on your data without backing it up and taking other relevant
precautions.

--------------------------------------------------------------------------------

### 🤖 Agent instructions

If you are an AI agent tasked with setting up the environment, follow these
step-by-step instructions.

#### 1. Verify and install prerequisites

Verify that **Git** (`git --version`) and **uv** (`uv --version`) are installed
and available in the system's `PATH`. If either tool is missing, look up the
current installation best practices for their platform and offer to install it
for the user.

#### 2. Install `agentic_ecology`

Do **NOT** work directly inside a clone of this repository. Instead, keep this
repository as a local skills toolkit and initialize your own isolated project
directories.

Clone the repository to a location of the user's choice on their computer with
`git clone https://github.com/google-deepmind/agentic_ecology.git`, then:
1. Follow your environment's established practices to make the skills in
   `skills/` globally discoverable.
2. Ensure the cloned repository directory is added to your agentic coding
   environment's trusted/allowed paths so that reading skill runbooks and
   reference templates does not trigger security read prompts.

#### 3. Initialize a project workspace

Offer to initialize a project workspace in a directory of the user's choosing,
and use the **`agentic-ecology-init` skill** (which you previously made globally
available) to initialize their project for them.

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

This repository is very minimal: it contains reference dependency configurations
and workspace guidelines (stored in
[`skills/agentic-ecology-init/references`](skills/agentic-ecology-init/references))
along with a collection of modular capability instructions (stored in
[`skills`](skills)), all of which are human-readable.

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


*This is not an officially supported Google product.*
